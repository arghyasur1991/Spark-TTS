#!/usr/bin/env python
import sys, pathlib, torch, argparse, inspect
# make "sparktts" importable when run from scripts/
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from sparktts.models.bicodec import BiCodec            # noqa
from torch.onnx import ExportOptions

import torch, types
from sparktts.modules.fsq import residual_fsq as _rfsq

def _plain_gather(self, indices):
    # indices: (q, b, n)  int32
    q, b, n = indices.shape
    codes = self.codebooks                    # (q, c, d)
    flat = indices.permute(0, 2, 1).reshape(q, -1, 1)   # (q, b*n, 1)
    gathered = torch.gather(codes, 1, flat.expand(-1, -1, codes.size(-1)))
    gathered = gathered.view(q, n, b, -1).permute(2, 1, 0, 3)  # (b,n,q,d)
    return gathered

_rfsq.ResidualFSQ.get_codes_from_indices = types.MethodType(
    _plain_gather, _rfsq.ResidualFSQ)

def load_bicodec(ckpt_root: pathlib.Path, device="cpu"):
    return BiCodec.load_from_checkpoint(ckpt_root, device=torch.device(device)).eval()


def make_export_options(**kwargs):
    """Build ExportOptions with the right field names for *this* torch build."""
    sig = inspect.signature(ExportOptions)
    params = sig.parameters
    mapped = {}
    # map aliases → canonical names that exist in this version
    if "opset_version" in params:
        mapped["opset_version"] = kwargs.get("opset", kwargs.get("opset_version", 18))
    else:
        mapped["opset"] = kwargs.get("opset_version", kwargs.get("opset", 18))

    if "dynamic_shapes" in params:
        mapped["dynamic_shapes"] = kwargs.get(
            "dynamic_axes", kwargs.get("dynamic_shapes")
        )
    else:
        mapped["dynamic_axes"] = kwargs.get(
            "dynamic_shapes", kwargs.get("dynamic_axes")
        )

    mapped["input_names"] = kwargs["input_names"]
    mapped["output_names"] = kwargs["output_names"]
    return ExportOptions(**mapped)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True,
                    help="Folder that contains model.safetensors + config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    ckpt_root = pathlib.Path(args.ckpt_root)
    model = load_bicodec(ckpt_root, device="cpu")      # keep CPU for export

    # ------------------------------------------------------------------
    # Build dummy *indices* by calling model.tokenize on one-second noise
    # ------------------------------------------------------------------
    sr         = 16_000
    duration_s = 1.0
    wav        = torch.randn(1, int(sr * duration_s))
    feat_len   = int(duration_s * 50)                  # feats are 20 ms hop
    feat       = torch.randn(1, feat_len, 1024)

    sem_tok, glob_tok = model.tokenize(
        {"feat": feat, "ref_wav": wav}
    )

    import torch.nn as nn

    class Detokenizer(nn.Module):
        def __init__(self, bic):
            super().__init__()
            self.bic = bic
        def forward(self, semantic_tokens, global_tokens):
            return self.bic.detokenize(semantic_tokens, global_tokens)

    detok = Detokenizer(model)      # `model` is your loaded BiCodec

    # Wrapper with argument order stable for ONNX
    def detokenize_wrap(semantic_tokens, global_tokens):
        return model.detokenize(semantic_tokens, global_tokens)

    print("Semantic-token shape:", tuple(sem_tok.shape))
    print("Global-token   shape:", tuple(glob_tok.shape))

    export_out = torch.onnx.dynamo_export(detok, sem_tok, glob_tok)
    export_out.save("onnx_models/bicodec.onnx")
    print(f"✅  BiCodec detokenizer exported ➜ {args.out}.onnx")

if __name__ == "__main__":
    main()
