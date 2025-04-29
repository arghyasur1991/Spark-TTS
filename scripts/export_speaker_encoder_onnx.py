#!/usr/bin/env python
"""
Speaker-embedding network used for zero-shot cloning.
We export it as a simple forward pass that maps (B,1,T) audio ➜ (B,256) embedding
"""
import argparse, pathlib, torch
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from sparktts.modules.speaker.speaker_encoder import SpeakerEncoder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    model = SpeakerEncoder.load_from_checkpoint(
        str(pathlib.Path(args.ckpt_root) / "SpeakerEncoder")
    ).eval().cpu()

    # 1-sec of 16 kHz mono
    dummy = torch.zeros(1, 1, 16_000)

    torch.onnx.dynamo_export(
        model,
        dummy,
        opset_version=args.opset,
        input_names=["audio"],
        output_names=["embedding"],
        dynamic_axes={"audio": {2: "T"}}
    ).save(pathlib.Path(args.out).with_suffix(".onnx"))

    print(f"✅  Speaker encoder exported to {args.out}.onnx")

if __name__ == "__main__":
    main()
