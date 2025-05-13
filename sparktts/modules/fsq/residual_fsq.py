import random
import torch
import torch.nn.functional as F
import torch.distributed as dist

from typing import List
from torch import nn
from torch.nn import Module
from torch.amp import autocast
from einops import rearrange, reduce, pack, unpack

from sparktts.modules.fsq.finite_scalar_quantization import FSQ


def exists(val):
    return val is not None


def first(l):
    return l[0]


def default(val, d):
    return val if exists(val) else d


def round_up_multiple(num, mult):
    from math import ceil
    return ceil(num / mult) * mult


# distributed helpers


def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def get_maybe_sync_seed(device, max_size=10_000):
    rand_int = torch.randint(0, max_size, (), device=device)

    if is_distributed():
        dist.all_reduce(rand_int)

    return rand_int.item()


class ResidualFSQ(Module):
    """Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf"""

    def __init__(
        self,
        *,
        levels: List[int],
        num_quantizers,
        dim=None,
        is_channel_first=False,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        quantize_dropout_multiple_of=1,
        **kwargs,
    ):
        super().__init__()
        codebook_dim = len(levels)
        dim = default(dim, codebook_dim)

        requires_projection = codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self.has_projections = requires_projection

        self.is_channel_first = is_channel_first
        self.num_quantizers = num_quantizers

        self.levels = levels
        self.layers = nn.ModuleList([])

        levels_tensor = torch.Tensor(levels)

        scales = []

        for ind in range(num_quantizers):
            scales.append((levels_tensor - 1) ** -ind)

            fsq = FSQ(levels=levels, dim=codebook_dim, **kwargs)

            self.layers.append(fsq)

        assert all([not fsq.has_projections for fsq in self.layers])

        self.codebook_size = self.layers[0].codebook_size

        self.register_buffer("scales", torch.stack(scales), persistent=False)

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

    @property
    def codebooks(self):
        codebooks = [layer.implicit_codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks

    def get_codes_from_indices(self, indices, onnx_export_mode=False):
        # indices shape: [B, N, Q] (batch, seq_len, num_quantizers)
        #   or after pack: [B*N, Q]
        # codebooks shape: [Q, C, D] (num_quantizers, codes_per_quantizer, dim)

        if onnx_export_mode:
            # Simplified path for ONNX export, assuming num_quantizers = 1 for speaker_encoder context
            # and no quantize_dropout related padding is active for this inference path.
            assert self.num_quantizers == 1, "ONNX export mode for get_codes_from_indices currently assumes num_quantizers == 1"
            
            # indices are expected to be [B, N, 1] or [B*N, 1] if packed by caller.
            # For speaker_encoder.detokenize -> self.quantizer.get_output_from_indices(indices.transpose(1,2))
            # If original indices to detokenize is [B, T, Nq], transpose makes it [B, Nq, T]
            # So, the 'indices' input here is [B, Nq_actual, T_actual]

            # Let B_in, Nq_in, T_in = indices.shape
            # For speaker_encoder context, Nq_in is self.num_quantizers (which we assert is 1)
            # T_in is the sequence length of tokens for that quantizer.
            
            # Original `pack` logic: indices, ps = pack([indices], "b * q") - this would make it [B*N, Q]
            # We need to ensure the input `indices` shape for ONNX mode matches what the embedding lookup expects.
            # If indices comes in as [B, N, Q], then: 
            B, N, Q_indices = indices.shape
            assert Q_indices == self.num_quantizers and self.num_quantizers == 1, \
                f"Shape mismatch or unexpected num_quantizers for ONNX. Indices: {indices.shape}, self.num_quantizers: {self.num_quantizers}"

            # Squeeze out the Q dimension from indices as it's 1
            indices_squeezed = indices.squeeze(-1) # Shape: [B, N]
            
            # Get the single codebook
            # self.codebooks is [Q, C, D]. For Q=1, it's [1, C, D]
            single_codebook = self.codebooks.squeeze(0) # Shape: [C, D]

            # Flatten batch and sequence for embedding lookup
            indices_flat = indices_squeezed.reshape(-1) # Shape: [B*N]
            
            # Perform embedding lookup
            selected_codes_flat = F.embedding(indices_flat, single_codebook) # Shape: [B*N, D]
            
            # Reshape back to [B, N, D]
            D_code = single_codebook.shape[-1]
            all_codes_unstacked = selected_codes_flat.reshape(B, N, D_code)
            
            # Add back the Q dimension: [1, B, N, D]
            all_codes = all_codes_unstacked.unsqueeze(0)
            
            # Masking for onnx_export_mode: assume no dropout, so no -1 indices.
            # If indices could legitimately be -1 and require masking, this needs to be handled.
            # For now, assuming valid indices for the vocoder path.

            # Scaling
            scales_reshaped = rearrange(self.scales, "q d -> q 1 1 d") # scales is [Q,D] -> [1,1,1,D] if Q=1
            all_codes = all_codes * scales_reshaped
            return all_codes

        else: # Original einx based implementation
            from einx import get_at # Import locally for non-ONNX mode
            _batch, _quantize_dim_orig = indices.shape[0], indices.shape[-1]
            indices_packed, ps = pack([indices], "b * q")
            _quantize_dim_packed = indices_packed.shape[-1]

            current_indices = indices_packed
            if _quantize_dim_packed < self.num_quantizers:
                assert (
                    self.quantize_dropout > 0.0
                ), "quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations"
                current_indices = F.pad(current_indices, (0, self.num_quantizers - _quantize_dim_packed), value=-1)

            mask = current_indices == -1
            current_indices = current_indices.masked_fill(mask, 0)

            all_codes = get_at("q [c] d, b n q -> q b n d", self.codebooks, current_indices)
            all_codes = all_codes.masked_fill(rearrange(mask, "b n q -> q b n 1"), 0.0)
            scales_rearranged = rearrange(self.scales, "q d -> q 1 1 d")
            all_codes = all_codes * scales_rearranged
            (all_codes_unpacked,) = unpack(all_codes, ps, "q b * d")
            return all_codes_unpacked


    def get_output_from_indices(self, indices, onnx_export_mode=False): # Thread the flag
        codes = self.get_codes_from_indices(indices, onnx_export_mode=onnx_export_mode)
        codes_summed = reduce(codes, "q ... -> ...", "sum") # This reduce might also need review for ONNX if Q > 1
                                                            # If Q=1, reduce("1 ... -> ...") is essentially a squeeze.
        if onnx_export_mode and self.num_quantizers == 1:
            # If Q=1, codes is [1, B, N, D], reduce results in [B, N, D]
            codes_summed = codes.squeeze(0)
            
        return self.project_out(codes_summed)

    def tokenize(self, x, **kwargs): # Added tokenize for completeness, mirroring typical quantizer API
        # Assuming forward pass is the way to get tokens (indices)
        # The 'forward' method already accepts onnx_export_mode
        _, indices = self.forward(x, return_all_codes=False, **kwargs)
        return indices

    def detokenize(self, indices, onnx_export_mode=False): # Added detokenize
        # This is an alias for get_output_from_indices
        return self.get_output_from_indices(indices, onnx_export_mode=onnx_export_mode)

    def forward(self, x, return_all_codes=False, rand_quantize_dropout_fixed_seed=None, onnx_export_mode=False): # Thread the flag
        num_quant, quant_dropout_multiple_of, device = (
            self.num_quantizers,
            self.quantize_dropout_multiple_of,
            x.device,
        )

        if self.is_channel_first:
            x = rearrange(x, "b d ... -> b ... d")
            x, ps = pack([x], "b * d")

        x = self.project_in(x)
        quantized_out = 0.0
        residual = x
        all_indices = []
        should_quantize_dropout = self.training and self.quantize_dropout and not onnx_export_mode # No dropout in ONNX export mode

        if should_quantize_dropout:
            if not exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)
            rand = random.Random(rand_quantize_dropout_fixed_seed)
            rand_quantize_dropout_index = rand.randrange(
                self.quantize_dropout_cutoff_index, num_quant
            )
            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = (
                    round_up_multiple(
                        rand_quantize_dropout_index + 1, quant_dropout_multiple_of
                    )
                    - 1
                )
            null_indices = torch.full(
                x.shape[:2], -1, device=device, dtype=torch.long # Ensure -1 is int for indices
            )

        with autocast("cuda", enabled=False):
            for quantizer_index, (layer, scale) in enumerate(
                zip(self.layers, self.scales)
            ):
                if (
                    should_quantize_dropout
                    and quantizer_index > rand_quantize_dropout_index
                ):
                    all_indices.append(null_indices)
                    continue
                
                # Pass onnx_export_mode to FSQ.forward if it also needs to adapt
                # Assuming FSQ.forward is ONNX-friendly or doesn't need this flag for now
                quantized, indices_layer = layer(residual / scale)
                quantized = quantized * scale
                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized
                all_indices.append(indices_layer)

        quantized_out = self.project_out(quantized_out)
        all_indices = torch.stack(all_indices, dim=-1)

        if self.is_channel_first:
            (quantized_out,) = unpack(quantized_out, ps, "b * d")
            (all_indices,) = unpack(all_indices, ps, "b * d")
            quantized_out = rearrange(quantized_out, "b ... d -> b d ...")
            all_indices = rearrange(all_indices, "b ... d -> b d ...")

        ret = (quantized_out, all_indices)
        if not return_all_codes:
            return ret
        
        # For return_all_codes path, ensure onnx_export_mode is passed
        all_codes = self.get_codes_from_indices(all_indices, onnx_export_mode=onnx_export_mode)
        return (*ret, all_codes)


# grouped residual fsq
# ... (GroupedResidualFSQ might also need onnx_export_mode if it calls these methods)

class GroupedResidualFSQ(Module):
    def __init__(self, *, dim, groups=1, accept_image_fmap=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.groups = groups
        assert (dim % groups) == 0
        dim_per_group = dim // groups

        self.accept_image_fmap = accept_image_fmap

        self.rvqs = nn.ModuleList([])

        for _ in range(groups):
            # Note: If ResidualFSQ's __init__ is changed, this might need adjustment
            # or kwargs should carry the onnx_export_mode if it were an init-time property.
            # For now, onnx_export_mode is a method-time property.
            self.rvqs.append(ResidualFSQ(dim=dim_per_group, **kwargs))

        self.codebook_size = self.rvqs[0].codebook_size

    @property
    def codebooks(self):
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    def get_codes_from_indices(self, indices, onnx_export_mode=False): # Thread flag
        # indices for GroupedResidualFSQ is expected to be a list/tuple of index tensors, one per group
        codes = tuple(
            rvq.get_codes_from_indices(chunk_indices, onnx_export_mode=onnx_export_mode)
            for rvq, chunk_indices in zip(self.rvqs, indices)
        )
        return torch.stack(codes)

    def get_output_from_indices(self, indices, onnx_export_mode=False): # Thread flag
        # indices for GroupedResidualFSQ is expected to be a list/tuple of index tensors, one per group
        outputs = tuple(
            rvq.get_output_from_indices(chunk_indices, onnx_export_mode=onnx_export_mode)
            for rvq, chunk_indices in zip(self.rvqs, indices)
        )
        return torch.cat(outputs, dim=self.split_dim)

    def tokenize(self, x, **kwargs): # Added tokenize for GroupedResidualFSQ
        # The 'forward' method already accepts onnx_export_mode
        # It returns (quantized, all_indices_stacked_across_groups)
        # For tokenization, we need all_indices
        _, all_indices = self.forward(x, return_all_codes=False, **kwargs)
        return all_indices # all_indices is shape [Groups, B, N_packed, Q_per_rvq] or similar

    def detokenize(self, indices, onnx_export_mode=False): # Added detokenize for GroupedResidualFSQ
        # This is an alias for get_output_from_indices
        # `indices` for GroupedResidualFSQ's get_output_from_indices is a list/tuple of index tensors
        return self.get_output_from_indices(indices, onnx_export_mode=onnx_export_mode)

    def forward(self, x, return_all_codes=False, onnx_export_mode=False): # Thread flag
        shape, split_dim, device = x.shape, self.split_dim, x.device
        assert shape[split_dim] == self.dim

        x_chunks = x.chunk(self.groups, dim=split_dim)

        forward_kwargs = dict(
            return_all_codes=return_all_codes,
            # Pass onnx_export_mode to the underlying ResidualFSQ forward calls
            onnx_export_mode=onnx_export_mode, 
            rand_quantize_dropout_fixed_seed=(
                get_maybe_sync_seed(device) if self.training and not onnx_export_mode else None
            ),
        )

        out_chunks = []
        for rvq, chunk in zip(self.rvqs, x_chunks):
            out_chunks.append(rvq(chunk, **forward_kwargs))
        
        out = tuple(zip(*out_chunks))

        quantized, all_indices, *maybe_all_codes = out
        quantized = torch.cat(quantized, dim=split_dim)
        all_indices = torch.stack(all_indices) # This assumes indices from each RVQ are compatible for stacking

        ret = (quantized, all_indices)

        if return_all_codes and maybe_all_codes:
            # If all_codes were returned, they are in maybe_all_codes[0], which is a tuple of code tensors from each group
            # We need to stack them appropriately, similar to how all_indices is handled by get_codes_from_indices for GroupedResidualFSQ
            # This part needs careful handling if return_all_codes is True for GroupedResidualFSQ ONNX export.
            # For now, focusing on the main path for vocoder which might not use return_all_codes=True here.
            all_codes_grouped = maybe_all_codes[0] # This is a tuple of code tensors
            # To be consistent with `get_codes_from_indices` for GroupedResidualFSQ, we should stack them.
            # The shape from ResidualFSQ.get_codes_from_indices is [Q, B, N, D_group]
            # So all_codes_grouped would be a tuple of such tensors.
            # Stacking them along a new 'group' dimension: [Groups, Q, B, N, D_group]
            all_codes_stacked = torch.stack(all_codes_grouped, dim=0) 
            return (*ret, all_codes_stacked)
        
        return ret


if __name__ == "__main__":
    from math import ceil # Ensure ceil is available
    model = ResidualFSQ(
        levels=[4, 4, 4, 4, 4, 4],
        num_quantizers=1,
        dim=30,
        is_channel_first=True,
        quantize_dropout=False,
    )
    x = torch.randn(2, 30, 10)
    quantize, embed_ind = model(x)

    # Test with onnx_export_mode if num_quantizers is 1
    if model.num_quantizers == 1:
        # Indices need to be in the shape [B, N, Q] for the ONNX path in get_codes_from_indices
        # model(x) returns embed_ind as [B, (D if channel_first else N), (N if channel_first else D)]
        # If is_channel_first=True, embed_ind is [B, D, N_sequence_elements_after_pack]
        # This needs to be reshaped or ensured to be [B, N, Q] for the ONNX path
        # For the test case: x [2,30,10] -> (is_channel_first=True) -> x_packed [2,10,30] (B,N,D)
        # all_indices after stack is [B,N,Q]. Here Q=1. So embed_ind should be [B,N,1] or [B,N] compatible.
        # The current `model(x)` returns `embed_ind` which is [B, D_channel_packed, N_spatial_packed]
        # This is not directly [B,N,Q]. The `transpose(1,2)` in speaker_encoder `get_output_from_indices` is key.

        # Let's simulate the input 'indices' to get_output_from_indices as it would be from speaker_encoder
        # speaker_encoder.quantizer(x) returns indices of shape e.g. [B, N_seq, Q_num_quantizers]
        # then in get_output_from_indices, it calls self.quantizer.get_codes_from_indices(indices.transpose(1,2))
        # So if original indices = [B, N, Q], transposed is [B, Q, N]
        # The ONNX path expects [B, N_new, Q_new] where Q_new must be 1.
        # So, for speaker_encoder context where Q=1, input to get_codes_from_indices is [B,1,N]
        
        # For this specific test, embed_ind is [2, 30, 10] (B, D_packed, N_packed from model forward)
        # We need to get it to the form that get_codes_from_indices ONNX path expects after transpose in SpeakerEncoder
        # In SpeakerEncoder, indices = quantizer(x) -> [B, N_seq, Q=1]
        # Then it calls get_output_from_indices(indices.transpose(1,2)) -> input to get_codes_from_indices is [B, Q=1, N_seq]
        test_indices_for_onnx = embed_ind.reshape(embed_ind.shape[0], 1, -1) # Make it [B, 1, N_elements]
        # This matches the expectation if num_quantizers is 1 for the ONNX path.
        emb_from_ind_onnx = model.get_output_from_indices(test_indices_for_onnx, onnx_export_mode=True)
        print("ONNX mode emb_from_ind shape:", emb_from_ind_onnx.shape)
        # emb_from_ind_onnx is [B, N_elements, D_proj_out]. We need to transpose it back for comparison if needed.
        # For this test, quantize is [B, D_proj_out, N_elements]
        # So comparison should be with emb_from_ind_onnx.transpose(1,2)
        print("Comparison with ONNX mode (quantize vs emb_from_ind_onnx.transpose(1,2)):")
        print(quantize == emb_from_ind_onnx.transpose(1,2))


    emb_from_ind = model.get_output_from_indices(embed_ind.transpose(1, 2))

    print(quantize == emb_from_ind.transpose(1, 2))

    print("quantize shape", quantize.shape)
    print("embed_ind", embed_ind)
