# Spark-TTS REST API

This is a REST API for the Spark-TTS text-to-speech system. It provides endpoints for generating speech from text with style parameters or voice cloning with a prompt audio file.

## Setup

1. Install the required dependencies:

```bash
pip install -r api_requirements.txt
```

2. Make sure you have the Spark-TTS model downloaded in the `pretrained_models/Spark-TTS-0.5B` directory.

3. Start the API server:

```bash
python api_server.py
```

The server will start on http://localhost:8000. You can access the API documentation at http://localhost:8000/docs.

## API Endpoints

### Health Check

```
GET /api/health
```

Returns the health status of the API and whether the TTS model is loaded.

### Generate Audio with Style Parameters

```
POST /api/generate-audio-style
```

Generate speech from text with specified style parameters.

**Request Body (JSON):**

```json
{
  "text": "Your text to convert to speech",
  "gender": "male|female",
  "pitch": "very_low|low|moderate|high|very_high",
  "speed": "very_low|low|moderate|high|very_high"
}
```

**Response (JSON):**

```json
{
  "audio_base64": "base64-encoded WAV audio",
  "file_path": "path/to/generated/audio.wav",
  "sample_rate": 16000
}
```

### Generate Audio with Prompt Audio

```
POST /api/generate-audio-prompt
```

Generate speech from text with voice cloning using a prompt audio file.

**Request (Multipart Form):**

- `text`: The text to convert to speech
- `prompt_text` (optional): Transcript of the prompt audio
- `prompt_audio`: Audio file to use for voice cloning

**Response (JSON):**

```json
{
  "audio_base64": "base64-encoded WAV audio",
  "file_path": "path/to/generated/audio.wav",
  "sample_rate": 16000
}
```

## Client Examples

See `api_client_examples.py` for examples of how to use the API from Python.

### Using the API from Python

```python
import requests
import base64
import json

# Generate audio with style parameters
response = requests.post(
    "http://localhost:8000/api/generate-audio-style",
    json={
        "text": "Hello, this is a test of the Spark TTS API.",
        "gender": "female",
        "pitch": "high",
        "speed": "moderate"
    }
)

if response.status_code == 200:
    data = response.json()
    # Decode and save the audio
    audio_data = base64.b64decode(data["audio_base64"])
    with open("output.wav", "wb") as f:
        f.write(audio_data)
```

### Using the API from cURL

Generate audio with style parameters:

```bash
curl -X POST "http://localhost:8000/api/generate-audio-style" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the Spark TTS API.",
    "gender": "female",
    "pitch": "high",
    "speed": "moderate"
  }' > response.json
```

Generate audio with prompt audio file:

```bash
curl -X POST "http://localhost:8000/api/generate-audio-prompt" \
  -F "text=This is voice cloning with a prompt audio file." \
  -F "prompt_text=This is a prompt recording." \
  -F "prompt_audio=@/path/to/prompt.wav" > response.json
```

## Configuration

The API server configuration can be adjusted in `api_server.py`:

- `MODEL_DIR`: Path to the TTS model directory
- `OUTPUT_DIR`: Directory to save generated audio files
- `PORT`: Server port (can be set via the PORT environment variable)

## Output Files

Generated audio files are saved in the `api_outputs` directory with unique filenames. The base64-encoded audio is also included in the API response for immediate playback. 