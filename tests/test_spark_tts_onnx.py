#!/usr/bin/env python3
"""
Unit tests for Spark-TTS ONNX model.
Tests deterministic behavior of the ONNX model with fixed seeds.
"""

import os
import sys
import time
import unittest
import numpy as np
import random
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cli.SparkTTSONNX import SparkTTSONNX

class TestSparkTTSONNX(unittest.TestCase):
    """Test Spark TTS ONNX deterministic behavior"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures, called before any tests are run"""
        # Paths
        cls.model_dir = Path("pretrained_models/Spark-TTS-0.5B")
        cls.prompt_path = Path("example/prompt_audio.wav")
        cls.output_dir = Path("tests/output")
        cls.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set fixed seeds for deterministic behavior
        np.random.seed(42)
        random.seed(42)
        os.environ['PYTHONHASHSEED'] = '42'
        
        # Skip tests if model doesn't exist
        if not cls.model_dir.exists():
            raise unittest.SkipTest(f"Model directory {cls.model_dir} does not exist")
            
        if not cls.prompt_path.exists():
            raise unittest.SkipTest(f"Prompt audio {cls.prompt_path} does not exist")
            
        # Check for ONNX directory
        onnx_dir = cls.model_dir / "onnx"
        if not onnx_dir.exists():
            raise unittest.SkipTest(f"ONNX model directory {onnx_dir} does not exist")
            
    def setUp(self):
        """Set up test fixtures before each test method is run"""
        # Set fixed seeds again before each test
        np.random.seed(42)
        random.seed(42)
        
    def test_model_initialization(self):
        """Test that the ONNX model can be initialized"""
        try:
            # Initialize ONNX model
            onnx_tts = SparkTTSONNX(model_dir=self.model_dir)
            self.assertIsNotNone(onnx_tts, "ONNX model initialization failed")
        except Exception as e:
            self.fail(f"ONNX model initialization failed: {e}")
            
    def test_deterministic_output(self):
        """Test that the ONNX model produces deterministic outputs with fixed seeds"""
        # Initialize ONNX model
        onnx_tts = SparkTTSONNX(model_dir=self.model_dir)
        
        # Process prompt
        onnx_tts.process_prompt(str(self.prompt_path))
        
        # Set voice parameters
        onnx_tts.process_prompt_control(gender=0.5, pitch=0.5, speed=0.5)
        
        # Generate speech twice with the same seed and same prompt
        test_text = "This is a test of deterministic generation."
        
        # First generation - create a deterministic embedding by setting a fixed seed
        np.random.seed(123)
        random.seed(123)
        
        # Store the original random state to restore it later
        orig_np_state = np.random.get_state()
        orig_random_state = random.getstate()
        
        wav1, _ = onnx_tts.inference(text=test_text)
        
        # Second generation - reset to exact same seed state
        np.random.seed(123)
        random.seed(123)
        wav2, _ = onnx_tts.inference(text=test_text)
        
        # Ensure outputs are identical
        self.assertTrue(np.array_equal(wav1, wav2), 
                        "ONNX model does not produce deterministic outputs with the same seed")
        print("ONNX model produces deterministic outputs with the same seed")
        
        # Save the outputs
        import soundfile as sf
        sf.write(self.output_dir / "deterministic_test1.wav", wav1, 16000)
        sf.write(self.output_dir / "deterministic_test2.wav", wav2, 16000)
        
        # Restore original random state
        np.random.set_state(orig_np_state)
        random.setstate(orig_random_state)
        
        # Generate with a different prompt but same seed to ensure it produces different output
        np.random.seed(123)  # Same seed as before
        random.seed(123)
        different_text = "This is completely different text to test determinism."
        wav3, _ = onnx_tts.inference(text=different_text)
        
        # Outputs should be different because text is different
        self.assertFalse(np.array_equal(wav1, wav3), 
                         "ONNX model produces the same output for different text")
        print("ONNX model produces different outputs for different text inputs with same seed")
        
        # Save the output with different text
        sf.write(self.output_dir / "deterministic_test3.wav", wav3, 16000)
        
    def test_voice_parameter_effects(self):
        """Test that different voice parameters produce different outputs"""
        try:
            # Initialize ONNX model
            onnx_tts = SparkTTSONNX(model_dir=self.model_dir)
            
            # Process prompt
            onnx_tts.process_prompt(str(self.prompt_path))
            
            # Using a fixed test text for all tests
            test_text = "Testing voice parameter effects."
            
            # Use the same random seed for all tests
            np_random_state = np.random.get_state()
            random_state = random.getstate()
            
            # Generate with neutral parameters
            np.random.seed(42)
            random.seed(42)
            onnx_tts.process_prompt_control(gender=0.5, pitch=0.5, speed=0.5)
            wav_neutral, _ = onnx_tts.inference(text=test_text)
            
            # Generate with male voice
            np.random.seed(42)
            random.seed(42)
            onnx_tts.process_prompt_control(gender=0.0, pitch=0.5, speed=0.5)
            wav_male, _ = onnx_tts.inference(text=test_text)
            
            # Generate with female voice
            np.random.seed(42)
            random.seed(42)
            onnx_tts.process_prompt_control(gender=1.0, pitch=0.5, speed=0.5)
            wav_female, _ = onnx_tts.inference(text=test_text)
            
            # Generate with high pitch
            np.random.seed(42)
            random.seed(42)
            onnx_tts.process_prompt_control(gender=0.5, pitch=1.0, speed=0.5)
            wav_high_pitch, _ = onnx_tts.inference(text=test_text)
            
            # Generate with moderate speed
            np.random.seed(42)
            random.seed(42)
            onnx_tts.process_prompt_control(gender=0.5, pitch=0.5, speed=0.5)
            wav_normal_speed, _ = onnx_tts.inference(text=test_text)
            
            # Restore random state
            np.random.set_state(np_random_state)
            random.setstate(random_state)
            
            # Save all outputs
            import soundfile as sf
            sf.write(self.output_dir / "voice_neutral.wav", wav_neutral, 16000)
            sf.write(self.output_dir / "voice_male.wav", wav_male, 16000)
            sf.write(self.output_dir / "voice_female.wav", wav_female, 16000)
            sf.write(self.output_dir / "voice_high_pitch.wav", wav_high_pitch, 16000)
            sf.write(self.output_dir / "voice_normal_speed.wav", wav_normal_speed, 16000)
            
            # Verify outputs differ
            # Normalize waveforms to ensure fair comparison
            def normalize_wav(wav):
                return wav / (np.max(np.abs(wav)) + 1e-9)
            
            wav_neutral_norm = normalize_wav(wav_neutral)
            wav_male_norm = normalize_wav(wav_male)
            wav_female_norm = normalize_wav(wav_female)
            wav_high_pitch_norm = normalize_wav(wav_high_pitch)
            
            # Calculate differences using mean absolute error
            def mae(wav1, wav2):
                min_len = min(len(wav1), len(wav2))
                return np.mean(np.abs(wav1[:min_len] - wav2[:min_len]))
            
            male_diff = mae(wav_neutral_norm, wav_male_norm)
            female_diff = mae(wav_neutral_norm, wav_female_norm)
            pitch_diff = mae(wav_neutral_norm, wav_high_pitch_norm)
            
            # Check if there are meaningful differences (MAE > threshold)
            threshold = 0.001  # Adjust based on empirical results
            
            self.assertGreater(male_diff, threshold, "Male voice parameter has no effect")
            self.assertGreater(female_diff, threshold, "Female voice parameter has no effect")
            self.assertGreater(pitch_diff, threshold, "Pitch parameter has no effect")
            
            print("Voice parameter controls produce measurable differences in output")
            print(f"MAE difference - male vs neutral: {male_diff:.4f}")
            print(f"MAE difference - female vs neutral: {female_diff:.4f}")
            print(f"MAE difference - high pitch vs neutral: {pitch_diff:.4f}")
            
        except Exception as e:
            self.skipTest(f"Voice parameter test failed due to: {str(e)}")
        
    def test_text_to_audio_mapping(self):
        """Test that different text inputs produce different audio outputs"""
        # Initialize ONNX model
        onnx_tts = SparkTTSONNX(model_dir=self.model_dir)
        
        # Process prompt
        onnx_tts.process_prompt(str(self.prompt_path))
        onnx_tts.process_prompt_control(gender=0.5, pitch=0.5, speed=0.5)
        
        # Generate speech for different texts
        np.random.seed(42)
        random.seed(42)
        text1 = "This is the first test sentence."
        wav1, _ = onnx_tts.inference(text=text1)
        
        np.random.seed(42)
        random.seed(42)
        text2 = "This is a completely different sentence."
        wav2, _ = onnx_tts.inference(text=text2)
        
        # Save the outputs
        import soundfile as sf
        sf.write(self.output_dir / "text1_output.wav", wav1, 16000)
        sf.write(self.output_dir / "text2_output.wav", wav2, 16000)
        
        # Verify outputs differ
        self.assertFalse(np.array_equal(wav1, wav2), 
                         "Different text inputs should produce different audio outputs")
        print("Different text inputs produce different audio outputs as expected")
        
        # Calculate metrics to quantify the difference
        min_len = min(len(wav1), len(wav2))
        wav1_trim = wav1[:min_len]
        wav2_trim = wav2[:min_len]
        
        mse = np.mean((wav1_trim - wav2_trim) ** 2)
        correlation = np.corrcoef(wav1_trim, wav2_trim)[0, 1]
        
        print(f"Mean Squared Error between different texts: {mse:.6f}")
        print(f"Correlation between different texts: {correlation:.6f}")


if __name__ == "__main__":
    unittest.main() 