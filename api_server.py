"""
SparkTTS REST API Server

This module implements a FastAPI web server that provides REST APIs for Spark-TTS.
"""

import os
import base64
import tempfile
import logging
import uuid
import time
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

class BatchStyleGenerationRequest(BaseModel):
    texts: list[str]
    gender: str
    pitch: Optional[str] = "moderate"
    speed: Optional[str] = "moderate"

class BatchPromptGenerationRequest(BaseModel):
    texts: list[str]
    prompt_text: Optional[str] = None

class AudioResponse(BaseModel):
    audio_base64: str
    file_path: str
    sample_rate: int = 16000
    
class BatchAudioResponse(BaseModel):
    responses: list[AudioResponse]
    total_processing_time: float

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

def validate_style_parameters(gender, pitch, speed):
    """
    Validate style parameters for audio generation.
    
    Args:
        gender: Voice gender (female, male)
        pitch: Voice pitch (low, moderate, high)
        speed: Voice speed (slow, moderate, fast)
        
    Raises:
        HTTPException: If any parameter is invalid
    """
    if gender not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="Gender must be 'male' or 'female'")
    
    pitch_options = ["very_low", "low", "moderate", "high", "very_high"]
    if pitch not in pitch_options:
        raise HTTPException(status_code=400, detail=f"Pitch must be one of {pitch_options}")
        
    speed_options = ["very_low", "low", "moderate", "high", "very_high"]
    if speed not in speed_options:
        raise HTTPException(status_code=400, detail=f"Speed must be one of {speed_options}")

@app.post("/api/generate-audio-style", response_model=AudioResponse)
async def generate_audio_style(
    text: str = Form(...),
    gender: Optional[str] = Form("female"),
    pitch: Optional[str] = Form("moderate"),
    speed: Optional[str] = Form("moderate")
):
    """
    Generate audio from text with specified style parameters.
    
    Args:
        text: Text to convert to speech
        gender: Voice gender (female, male)
        pitch: Voice pitch (low, moderate, high)
        speed: Voice speed (slow, moderate, fast)
    
    Returns:
        AudioResponse containing base64-encoded audio and file path
    """
    try:
        model = initialize_model()
        start_time = time.time()
        
        logger.info(f"Generating audio with parameters: text='{text}', gender={gender}, pitch={pitch}, speed={speed}")
        
        with torch.no_grad():
            # Check text is not empty
            if not text or text.strip() == "":
                raise HTTPException(status_code=400, detail="Text cannot be empty")
                
            # Validate style parameters
            validate_style_parameters(gender, pitch, speed)
            
            # Generate audio
            try:
                wav, _, _, _ = model.inference(
                    text=text,
                    gender=gender,
                    pitch=pitch,
                    speed=speed
                )
                
                # Validate output
                if wav is None or len(wav) < 100:  # Arbitrary minimum length check
                    raise ValueError(f"Generated audio is invalid (length: {len(wav) if wav is not None else 0})")
                
                # Save to file
                filename = f"{uuid.uuid4()}.wav"
                file_path = os.path.join(OUTPUT_DIR, filename)
                sf.write(file_path, wav, samplerate=16000)
                
                # Create response with base64-encoded audio
                with open(file_path, "rb") as audio_file:
                    audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
                
                processing_time = time.time() - start_time
                logger.info(f"Audio generated successfully in {processing_time:.2f} seconds")
                
                return AudioResponse(
                    audio_base64=audio_base64,
                    file_path=file_path
                )
            except Exception as inner_e:
                logger.error(f"Error during audio generation: {str(inner_e)}")
                raise ValueError(f"Failed to generate audio: {str(inner_e)}")
        
    except HTTPException as http_e:
        # Re-raise HTTP exceptions
        raise http_e
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
            # The inference method returns (wav, model_inputs, global_token_ids, timing_data)
            wav, _, _, _ = model.inference(
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

@app.post("/api/batch-generate-audio-style", response_model=BatchAudioResponse)
async def batch_generate_audio_style(request: BatchStyleGenerationRequest):
    """
    Generate multiple audio files from texts with same style parameters, reusing tokenization.
    
    Args:
        texts: List of texts to convert to speech
        gender: "male" or "female"
        pitch: "very_low", "low", "moderate", "high", or "very_high"
        speed: "very_low", "low", "moderate", "high", or "very_high"
    
    Returns:
        BatchAudioResponse containing list of audio responses and total processing time
    """
    try:
        model = initialize_model()
        start_time = time.time()
        
        # Validate parameters
        if request.gender not in ["male", "female"]:
            raise HTTPException(status_code=400, detail="Gender must be 'male' or 'female'")
        
        pitch_options = ["very_low", "low", "moderate", "high", "very_high"]
        if request.pitch not in pitch_options:
            raise HTTPException(status_code=400, detail=f"Pitch must be one of {pitch_options}")
            
        speed_options = ["very_low", "low", "moderate", "high", "very_high"]
        if request.speed not in speed_options:
            raise HTTPException(status_code=400, detail=f"Speed must be one of {speed_options}")
        
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        responses = []
        
        # Process first text
        logger.info(f"Generating first audio with style parameters: gender={request.gender}, pitch={request.pitch}, speed={request.speed}")
        try:
            with torch.no_grad():
                # For style-based generation, we need to initialize global_token_ids appropriately
                # First, generate audio directly from text and style parameters
                wav, model_inputs, global_token_ids, _ = model.inference(
                    text=request.texts[0],
                    gender=request.gender,
                    pitch=request.pitch,
                    speed=request.speed
                )
                
                # Validate global_token_ids
                if global_token_ids is None or global_token_ids.numel() == 0:
                    raise ValueError("Failed to extract global token IDs from first inference")
                
                # Validate wav output
                if wav is None or len(wav) < 100:  # Arbitrary minimum length check
                    raise ValueError(f"Generated audio for text 1 is invalid (length: {len(wav) if wav is not None else 0})")
                
                # Only now tokenize the inputs with the extracted global_token_ids
                model_inputs, _ = model.tokenize_inputs(
                    text=request.texts[0],
                    gender=request.gender,
                    pitch=request.pitch,
                    speed=request.speed,
                )
                
                # Save first audio
                filename = f"{uuid.uuid4()}.wav"
                file_path = os.path.join(OUTPUT_DIR, filename)
                sf.write(file_path, wav, samplerate=16000)
                
                # Create response with base64-encoded audio
                with open(file_path, "rb") as audio_file:
                    audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
                
                responses.append(AudioResponse(audio_base64=audio_base64, file_path=file_path))
        except Exception as inner_e:
            logger.error(f"Error generating first audio: {str(inner_e)}")
            raise HTTPException(status_code=500, detail=f"Error generating first audio: {str(inner_e)}")
        
        # Process remaining texts
        for i, text in enumerate(request.texts[1:], 1):
            logger.info(f"Generating audio {i+1} with reused tokenization")
            try:
                with torch.no_grad():
                    # Update text in existing tokenized inputs
                    updated_inputs = model.update_text_in_tokenized_inputs(
                        model_inputs, text, True, request.gender, request.pitch, request.speed
                    )
                    
                    # Ensure global_token_ids is still valid
                    if global_token_ids is None or global_token_ids.numel() == 0:
                        logger.warning(f"Invalid global_token_ids for text {i+1}, regenerating with full inference")
                        # Regenerate with full inference if global_token_ids is invalid
                        wav, _, global_token_ids, _ = model.inference(
                            text=text,
                            gender=request.gender,
                            pitch=request.pitch,
                            speed=request.speed
                        )
                    else:
                        # Generate with updated inputs - unpack all return values
                        wav, model_inputs, new_global_tokens, _ = model.inference(
                            model_inputs=updated_inputs,
                            global_token_ids=global_token_ids,
                        )
                        
                        # Update global_token_ids if new ones are available
                        if new_global_tokens is not None and new_global_tokens.numel() > 0:
                            global_token_ids = new_global_tokens
                    
                    # Validate wav output
                    if wav is None or len(wav) < 100:  # Arbitrary minimum length check
                        raise ValueError(f"Generated audio for text {i+1} is invalid (length: {len(wav) if wav is not None else 0})")
                    
                    # Save audio
                    filename = f"{uuid.uuid4()}.wav"
                    file_path = os.path.join(OUTPUT_DIR, filename)
                    sf.write(file_path, wav, samplerate=16000)
                    
                    # Create response with base64-encoded audio
                    with open(file_path, "rb") as audio_file:
                        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
                    
                    responses.append(AudioResponse(audio_base64=audio_base64, file_path=file_path))
                    
                    # Force garbage collection to free memory
                    if i % 5 == 0:  # Every 5 iterations
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as inner_e:
                logger.error(f"Error generating audio for text {i+1}: {str(inner_e)}")
                # Continue with next text instead of failing the entire batch
        
        total_time = time.time() - start_time
        return BatchAudioResponse(responses=responses, total_processing_time=total_time)
        
    except Exception as e:
        logger.error(f"Error generating batch audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-generate-audio-prompt", response_model=BatchAudioResponse)
async def batch_generate_audio_prompt(
    texts: list[str] = Form(...),
    prompt_text: Optional[str] = Form(None),
    prompt_audio: UploadFile = File(...)
):
    """
    Generate multiple audio files from texts with same prompt audio, reusing tokenization.
    
    Args:
        texts: List of texts to convert to speech
        prompt_text: Optional transcript of the prompt audio
        prompt_audio: Audio file to use as a prompt
    
    Returns:
        BatchAudioResponse containing list of audio responses and total processing time
    """
    temp_dir = None
    try:
        model = initialize_model()
        start_time = time.time()
        
        # Validate inputs
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if not prompt_audio.filename:
            raise HTTPException(status_code=400, detail="No prompt audio file provided")
            
        # Check file extension
        file_ext = os.path.splitext(prompt_audio.filename)[1].lower()
        if file_ext not in ['.wav', '.mp3', '.ogg', '.flac']:
            raise HTTPException(status_code=400, detail=f"Unsupported audio format: {file_ext}. Please use WAV, MP3, OGG, or FLAC")
        
        # Save the uploaded prompt audio to a temporary file
        temp_dir = tempfile.mkdtemp()
        prompt_path = os.path.join(temp_dir, prompt_audio.filename)
        
        # Read audio file content with error handling
        try:
            content = await prompt_audio.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
                
            with open(prompt_path, "wb") as temp_file:
                temp_file.write(content)
                
            # Verify the audio file is valid
            try:
                audio_info = sf.info(prompt_path)
                logger.info(f"Prompt audio: {prompt_path}, duration: {audio_info.duration:.2f}s, sample rate: {audio_info.samplerate}Hz")
                if audio_info.duration < 0.1:  # Less than 100ms is probably invalid
                    raise HTTPException(status_code=400, detail="Prompt audio is too short or invalid")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error reading prompt audio: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to read prompt audio: {str(e)}")
        
        responses = []
        successful_texts = 0
        failed_texts = 0
        
        # Process first text
        logger.info(f"Generating first audio with prompt audio: {prompt_path}")
        try:
            with torch.no_grad():
                # Initial tokenization
                try:
                    model_inputs, global_token_ids = model.tokenize_inputs(
                        text=texts[0],
                        prompt_speech_path=prompt_path
                    )
                except Exception as e:
                    logger.error(f"Error during tokenization: {str(e)}")
                    raise ValueError(f"Failed to tokenize inputs: {str(e)}")
                
                # Process first text - unpack all return values
                try:
                    wav, model_inputs, global_token_ids, timing_info = model.inference(
                        model_inputs=model_inputs,
                        global_token_ids=global_token_ids,
                    )
                except Exception as e:
                    logger.error(f"Error during inference: {str(e)}")
                    raise ValueError(f"Failed to generate audio: {str(e)}")
                
                # Validate the output
                if wav is None or len(wav) < 100:
                    logger.error(f"Generated audio is too short: {len(wav) if wav is not None else 0} samples")
                    raise ValueError("Generated audio is invalid or too short")
                
                # Save first audio
                filename = f"{uuid.uuid4()}.wav"
                file_path = os.path.join(OUTPUT_DIR, filename)
                sf.write(file_path, wav, samplerate=16000)
                
                # Create response with base64-encoded audio
                with open(file_path, "rb") as audio_file:
                    audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
                
                responses.append(AudioResponse(audio_base64=audio_base64, file_path=file_path))
                successful_texts += 1
                
                logger.info(f"First audio generation successful, length: {len(wav)} samples, duration: {len(wav)/16000:.2f}s")
                
                # Check if global_token_ids is valid for subsequent processing
                if global_token_ids is None:
                    logger.error("Global token IDs not available after first inference")
                    raise ValueError("Failed to extract global token IDs for voice cloning")
        
        except Exception as e:
            logger.error(f"Error processing first text: {str(e)}")
            failed_texts += 1
            responses.append(AudioResponse(audio_base64="", file_path="", error=str(e)))
            raise HTTPException(status_code=500, detail=f"Failed to process first text: {str(e)}")
        
        # Process remaining texts
        for i, text in enumerate(texts[1:], 1):
            try:
                logger.info(f"Generating audio {i+1} with reused tokenization")
                with torch.no_grad():
                    # Update text in existing tokenized inputs
                    try:
                        updated_inputs = model.update_text_in_tokenized_inputs(
                            model_inputs, text
                        )
                    except Exception as e:
                        logger.error(f"Error updating text in tokenized inputs: {str(e)}")
                        raise ValueError(f"Failed to update inputs with new text: {str(e)}")
                    
                    # Generate with updated inputs
                    try:
                        wav, model_inputs, _, timing_info = model.inference(
                            model_inputs=updated_inputs,
                            global_token_ids=global_token_ids,
                        )
                    except Exception as e:
                        logger.error(f"Error during inference for text {i+1}: {str(e)}")
                        raise ValueError(f"Failed to generate audio: {str(e)}")
                    
                    # Validate the output
                    if wav is None or len(wav) < 100:
                        logger.error(f"Generated audio is too short: {len(wav) if wav is not None else 0} samples")
                        raise ValueError("Generated audio is invalid or too short")
                    
                    # Save audio
                    filename = f"{uuid.uuid4()}.wav"
                    file_path = os.path.join(OUTPUT_DIR, filename)
                    sf.write(file_path, wav, samplerate=16000)
                    
                    # Create response with base64-encoded audio
                    with open(file_path, "rb") as audio_file:
                        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
                    
                    responses.append(AudioResponse(audio_base64=audio_base64, file_path=file_path))
                    successful_texts += 1
                    
                    logger.info(f"Audio generation {i+1} successful, length: {len(wav)} samples, duration: {len(wav)/16000:.2f}s")
            
            except Exception as e:
                logger.error(f"Error processing text {i+1}: {str(e)}")
                failed_texts += 1
                responses.append(AudioResponse(audio_base64="", file_path="", error=str(e)))
                # Continue processing other texts even if one fails
        
        total_time = time.time() - start_time
        logger.info(f"Batch processing completed in {total_time:.2f}s. Successful: {successful_texts}, Failed: {failed_texts}")
        
        return BatchAudioResponse(responses=responses, total_processing_time=total_time)
        
    except Exception as e:
        logger.error(f"Error in batch audio generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            try:
                prompt_path = os.path.join(temp_dir, prompt_audio.filename)
                if os.path.exists(prompt_path):
                    os.unlink(prompt_path)
                os.rmdir(temp_dir)
                logger.debug("Temporary files cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {str(e)}")

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