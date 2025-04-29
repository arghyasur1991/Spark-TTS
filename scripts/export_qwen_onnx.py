#!/usr/bin/env python
"""
Export the causal-LM (Qwen-2.5) to ONNX with past-key-values so we keep
generation speed. Requires 🤗Optimum ≥ 1.19.

Usage:
python export_qwen_onnx.py \
       --model Qwen/Qwen2.5-0.5B --out onnx_models/qwen
"""
import argparse, pathlib, os, shutil, subprocess, sys, tempfile
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from optimum.exporters.onnx import main_export  # high-level API
from transformers import AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF repo id or local dir containing the weights.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Optimum’s CLI exposed as python: we want text-generation with past
    main_export(
        model_name_or_path=args.model,
        output=args.out,
        task="text-generation-with-past",
        device="cpu",               # ONNX graph is device-agnostic
        opset=args.opset,
        fp16=False
    )

    # save tokenizer alongside for convenience
    AutoTokenizer.from_pretrained(args.model).save_pretrained(out)
    print(f"✅  Qwen exported to {out}")

if __name__ == "__main__":
    main()
