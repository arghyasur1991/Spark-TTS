"""
SparkTTS REST API Server

This module implements a FastAPI web server that provides REST APIs for Spark-TTS.
"""

import os
import base64
import tempfile
import logging
import uuid
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cli.SparkTTS import SparkTTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Spark-TTS API",
    description="REST API for Spark Text-to-Speech Generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
TTS_MODEL = None
OUTPUT_DIR = "api_outputs"
MODEL_DIR = "pretrained_models/Spark-TTS-0.5B"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Request/Response models
class StyleGenerationRequest(BaseModel):
    text: str
    gender: str
    pitch: Optional[str] = "moderate"
    speed: Optional[str] = "moderate"

class PromptGenerationRequest(BaseModel):
    text: str
    prompt_text: Optional[str] = None

class AudioResponse(BaseModel):
    audio_base64: str
    file_path: str
    sample_rate: int = 16000

def get_device():
    """Determine the best available device for inference."""
    if torch.backends.mps.is_available():
        # macOS with MPS support (Apple Silicon)
        device = torch.device("mps:0")
        logger.info("Using MPS device")
    elif torch.cuda.is_available():
        # System with CUDA support
        device = torch.device("cuda:0")
        logger.info("Using CUDA device")
    else:
        # Fall back to CPU
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device

def initialize_model():
    """Initialize the TTS model if not already initialized."""
    global TTS_MODEL
    if TTS_MODEL is None:
        device = get_device()
        logger.info(f"Initializing Spark-TTS model from {MODEL_DIR}")
        TTS_MODEL = SparkTTS(MODEL_DIR, device)
    return TTS_MODEL

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the server starts."""
    initialize_model()

@app.post("/api/generate-audio-style", response_model=AudioResponse)
async def generate_audio_style(request: StyleGenerationRequest):
    """
    Generate audio from text with specified style parameters.
    
    Args:
        text: The text to convert to speech
        gender: "male" or "female"
        pitch: "very_low", "low", "moderate", "high", or "very_high"
        speed: "very_low", "low", "moderate", "high", or "very_high"
    
    Returns:
        AudioResponse containing base64-encoded audio and file path
    """
    try:
        model = initialize_model()
        
        # Validate parameters
        if request.gender not in ["male", "female"]:
            raise HTTPException(status_code=400, detail="Gender must be 'male' or 'female'")
        
        pitch_options = ["very_low", "low", "moderate", "high", "very_high"]
        if request.pitch not in pitch_options:
            raise HTTPException(status_code=400, detail=f"Pitch must be one of {pitch_options}")
            
        speed_options = ["very_low", "low", "moderate", "high", "very_high"]
        if request.speed not in speed_options:
            raise HTTPException(status_code=400, detail=f"Speed must be one of {speed_options}")
        
        # Generate audio
        logger.info(f"Generating audio with style parameters: gender={request.gender}, pitch={request.pitch}, speed={request.speed}")
        with torch.no_grad():
            wav = model.inference(
                text=request.text,
                gender=request.gender,
                pitch=request.pitch,
                speed=request.speed
            )
        
        # Save audio file
        filename = f"{uuid.uuid4()}.wav"
        file_path = os.path.join(OUTPUT_DIR, filename)
        sf.write(file_path, wav, samplerate=16000)
        
        # Create response with base64-encoded audio
        with open(file_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
        
        return AudioResponse(audio_base64=audio_base64, file_path=file_path)
        
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-audio-prompt", response_model=AudioResponse)
async def generate_audio_prompt(
    text: str = Form(...),
    prompt_text: Optional[str] = Form(None),
    prompt_audio: UploadFile = File(...)
):
    """
    Generate audio from text with a prompt audio file.
    
    Args:
        text: The text to convert to speech
        prompt_text: Optional transcript of the prompt audio
        prompt_audio: Audio file to use as a prompt
    
    Returns:
        AudioResponse containing base64-encoded audio and file path
    """
    try:
        model = initialize_model()
        
        # Save the uploaded prompt audio to a temporary file
        temp_dir = tempfile.mkdtemp()
        prompt_path = os.path.join(temp_dir, prompt_audio.filename)
        
        with open(prompt_path, "wb") as temp_file:
            content = await prompt_audio.read()
            temp_file.write(content)
        
        # Generate audio
        logger.info(f"Generating audio with prompt audio: {prompt_path}")
        with torch.no_grad():
            wav = model.inference(
                text=text,
                prompt_speech_path=prompt_path,
                prompt_text=prompt_text
            )
        
        # Save generated audio file
        filename = f"{uuid.uuid4()}.wav"
        file_path = os.path.join(OUTPUT_DIR, filename)
        sf.write(file_path, wav, samplerate=16000)
        
        # Create response with base64-encoded audio
        with open(file_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
        
        # Clean up temporary file
        os.unlink(prompt_path)
        os.rmdir(temp_dir)
        
        return AudioResponse(audio_base64=audio_base64, file_path=file_path)
        
    except Exception as e:
        logger.error(f"Error generating audio with prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """API health check endpoint."""
    return {"status": "healthy", "model_loaded": TTS_MODEL is not None}

if __name__ == "__main__":
    import uvicorn
    
    # Define server port
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True) 