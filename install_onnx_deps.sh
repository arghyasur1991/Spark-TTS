#!/bin/bash

echo "Installing ONNX dependencies for Spark-TTS on macOS with Apple Silicon..."

# First, ensure pip is up to date
pip install --upgrade pip

# Install numpy, torch and other basic dependencies first
pip install numpy==2.2.3 torch==2.5.1 safetensors==0.5.2 omegaconf==2.3.0

# Install ONNX with no build isolation to avoid CMake issues
pip install "onnx>=1.15.0" --no-build-isolation

# Install onnxruntime for Apple Silicon
# Recent versions of onnxruntime have M1/M2/M3 support built-in
pip install "onnxruntime>=1.17.0"

echo "Installation complete! You can now run the conversion script:"
echo "python convert_to_onnx.py --skip_verification"

# Show the installed ONNX Runtime version and available providers
echo ""
echo "ONNX Runtime Information:"
python -c "import onnxruntime as ort; print(f'ONNX Runtime version: {ort.__version__}'); print(f'Available providers: {ort.get_available_providers()}')" 