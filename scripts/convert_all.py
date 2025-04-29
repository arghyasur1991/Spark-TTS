#!/usr/bin/env python
"""
Drive all individual export scripts so that one call gives you a full ONNX
drop-in replacement for Spark-TTS inference.

Example:
python convert_all.py \
    --spark_model_dir pretrained_models/Spark-TTS-0.5B \
    --qwen_repo Qwen/Qwen2.5-0.5B \
    --output_dir onnx_models
"""
import argparse, pathlib, subprocess, sys, shutil

ROOT = pathlib.Path(__file__).resolve().parent
PY = sys.executable

def run(cmd):
    print(" ".join(cmd)); subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark_model_dir", required=True,
                    help="Path to Spark-TTS checkpoint folder (same as you "
                         "pass to BiCodecTokenizer).")
    ap.add_argument("--qwen_repo", required=True,
                    help="HF hub id or local path of Qwen2.5 model.")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    run([PY, str(ROOT/"export_qwen_onnx.py"),
         "--model", args.qwen_repo,
         "--out", str(out/"qwen")])

    run([PY, str(ROOT/"export_bicodec_onnx.py"),
        "--ckpt_root", str(pathlib.Path(args.spark_model_dir) / "BiCodec"),
        "--out", str(out/"bicodec")])

    run([PY, str(ROOT/"export_speaker_encoder_onnx.py"),
         "--ckpt_root", args.spark_model_dir,
         "--out", str(out/"speaker_encoder")])

    print(f"\n✅  All ONNX files are under {out}")

if __name__ == "__main__":
    main()
