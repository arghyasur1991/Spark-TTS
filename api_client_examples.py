"""
Spark-TTS API Client Examples

This script contains examples of how to use the Spark-TTS REST API.
"""

import requests
import base64
import os
import json

# API endpoint base URL
API_BASE_URL = "http://localhost:8000"

def save_audio_from_response(response_data, output_folder="example_outputs"):
    """Save the base64-encoded audio from the API response to a WAV file."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the audio data and file path from the response
    audio_base64 = response_data["audio_base64"]
    remote_file_path = response_data["file_path"]
    filename = os.path.basename(remote_file_path)
    local_file_path = os.path.join(output_folder, filename)
    
    # Decode and save the audio file
    audio_data = base64.b64decode(audio_base64)
    with open(local_file_path, "wb") as f:
        f.write(audio_data)
    
    print(f"Audio saved to: {local_file_path}")
    return local_file_path

def example_generate_with_style():
    """Example of generating audio with style parameters."""
    print("\n--- Example: Generate Audio with Style Parameters ---")
    
    # Endpoint URL
    url = f"{API_BASE_URL}/api/generate-audio-style"
    
    # Request data
    data = {
        "text": "Hello, this is a test of the Spark TTS API with style parameters.",
        "gender": "female",
        "pitch": "high",
        "speed": "moderate"
    }
    
    # Send request
    print(f"Sending request to {url} with data: {json.dumps(data, indent=2)}")
    response = requests.post(url, json=data)
    
    # Process response
    if response.status_code == 200:
        response_data = response.json()
        save_audio_from_response(response_data)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def example_generate_with_prompt(prompt_file_path):
    """Example of generating audio with a prompt audio file."""
    print("\n--- Example: Generate Audio with Prompt Audio ---")
    
    # Endpoint URL
    url = f"{API_BASE_URL}/api/generate-audio-prompt"
    
    # Request data
    text = "This is an example of voice cloning using a prompt audio file."
    prompt_text = "This is a prompt recording for voice cloning."
    
    # Prepare files and form data
    files = {
        "prompt_audio": (os.path.basename(prompt_file_path), open(prompt_file_path, "rb"), "audio/wav")
    }
    form_data = {
        "text": text,
        "prompt_text": prompt_text
    }
    
    # Send request
    print(f"Sending request to {url}")
    print(f"Text: {text}")
    print(f"Prompt text: {prompt_text}")
    print(f"Prompt audio: {prompt_file_path}")
    
    response = requests.post(url, files=files, data=form_data)
    
    # Process response
    if response.status_code == 200:
        response_data = response.json()
        save_audio_from_response(response_data)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def check_api_health():
    """Check if the API is healthy and ready to use."""
    url = f"{API_BASE_URL}/api/health"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy" and data.get("model_loaded"):
                print("API is healthy and the model is loaded.")
                return True
            else:
                print("API is responding but may not be fully initialized.")
                return False
        else:
            print(f"API health check failed with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to the API at {url}. Is the server running?")
        return False

if __name__ == "__main__":
    print("Spark-TTS API Client Examples")
    
    # Check if the API is healthy before running examples
    if check_api_health():
        # Example 1: Generate audio with style parameters
        example_generate_with_style()
        
        # Example 2: Generate audio with a prompt audio file
        # Replace with the path to your own prompt audio file
        prompt_file_path = "example/prompt.wav"
        
        # Check if the prompt file exists
        if os.path.exists(prompt_file_path):
            example_generate_with_prompt(prompt_file_path)
        else:
            print(f"Prompt file not found: {prompt_file_path}")
            print("Skipping the prompt-based audio generation example.")
            print("To run this example, provide a valid path to a WAV audio file for voice cloning.")
    else:
        print("API health check failed. Make sure the API server is running.")
        print("To start the server, run: python api_server.py") 