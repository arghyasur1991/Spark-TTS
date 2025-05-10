# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import torch
import time
from typing import Tuple, Dict, Optional, Literal
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP


class SparkTTS:
    """
    Spark-TTS for text-to-speech generation.
    """

    def __init__(
        self, 
        model_dir: Path, 
        device: torch.device = torch.device("cuda:0"),
        quantization: Optional[Literal["int8", "fp16", "none"]] = None,
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        use_wav2vec2_onnx: bool = False,
    ):
        """
        Initializes the SparkTTS model with the provided configurations and device.

        Args:
            model_dir (Path): Directory containing the model and config files.
            device (torch.device): The device (CPU/GPU) to run the model on.
            quantization (str, optional): Quantization method to use ('int8', 'fp16', or None).
            use_compile (bool, optional): Whether to use torch.compile for acceleration (PyTorch 2.0+).
            compile_mode (str, optional): Compilation mode for torch.compile (default, reduce-overhead, max-autotune).
            use_wav2vec2_onnx (bool, optional): Whether to use ONNX for Wav2Vec2 feature extraction in BiCodecTokenizer.
        """
        self.device = device
        self.model_dir = model_dir
        self.configs = load_config(f"{model_dir}/config.yaml")
        self.sample_rate = self.configs["sample_rate"]
        self.quantization = quantization
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.use_wav2vec2_onnx = use_wav2vec2_onnx
        self._initialize_inference()

    def _initialize_inference(self):
        """Initializes the tokenizer, model, and audio tokenizer for inference."""
        print("\n==== Initializing SparkTTS Model ====")
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_dir}/LLM")
        
        # Load base model
        if self.quantization == "fp16" and self.device.type in ["cuda", "mps"]:
            # Load with FP16 precision for GPU
            print("Using FP16 precision for model")
            self.model = AutoModelForCausalLM.from_pretrained(
                f"{self.model_dir}/LLM",
                torch_dtype=torch.float16,
            )
        else:
            # Load with default precision
            self.model = AutoModelForCausalLM.from_pretrained(f"{self.model_dir}/LLM")
        
        # Apply quantization if requested
        if self.quantization == "int8":
            try:
                # Use dynamic quantization for INT8
                print("Applying INT8 dynamic quantization...")
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                print("Applied INT8 dynamic quantization to model")
            except Exception as e:
                print(f"Failed to apply INT8 quantization: {e}")
                print("Using original model without quantization")
        
        # Move model to device
        self.model.to(self.device)
        
        # Apply torch.compile if requested (and available)
        if self.use_compile:
            if hasattr(torch, 'compile'):
                try:
                    print(f"Applying torch.compile with mode '{self.compile_mode}'...")
                    # Available modes: default, reduce-overhead, max-autotune
                    self.model = torch.compile(self.model, mode=self.compile_mode)
                    print("✓ Model successfully compiled with torch.compile")
                except Exception as e:
                    print(f"✗ Failed to compile model: {e}")
                    print("Using original model without compilation")
            else:
                print("✗ torch.compile not available - requires PyTorch 2.0+")
                print("Please upgrade PyTorch to use this feature")
        
        # Initialize audio tokenizer
        print(f"Initializing BiCodecTokenizer with use_onnx_wav2vec2={self.use_wav2vec2_onnx}")
        self.audio_tokenizer = BiCodecTokenizer(
            self.model_dir, 
            device=self.device,
            use_onnx_wav2vec2=self.use_wav2vec2_onnx
        )
        print("==== Model Initialization Complete ====\n")

    def _measure_time(self, func, *args, **kwargs):
        """Helper method to measure execution time of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time

    def process_prompt(
        self,
        text: str,
        prompt_speech_path: Path,
        prompt_text: str = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.

        Return:
            Tuple[str, torch.Tensor]: Input prompt; global tokens
        """

        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
            prompt_speech_path
        )
        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
        )

        # Prepare the input tokens for the model
        if prompt_text is not None:
            semantic_tokens = "".join(
                [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
            )
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                prompt_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)

        return inputs, global_token_ids

    def process_prompt_control(
        self,
        gender: str,
        pitch: str,
        speed: str,
        text: str,
    ):
        """
        Process input for voice creation.

        Args:
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            text (str): The text input to be converted to speech.

        Return:
            str: Input prompt
        """
        assert gender in GENDER_MAP.keys()
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join(
            [gender_tokens, pitch_label_tokens, speed_label_tokens]
        )

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)

    @torch.no_grad()
    def tokenize_inputs(
        self,
        text: str,
        prompt_speech_path: Path = None,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes inputs without performing inference.
        
        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing (model_inputs, global_token_ids)
        """
        global_token_ids = None
        
        if gender is not None:
            prompt = self.process_prompt_control(gender, pitch, speed, text)
        else:
            prompt, global_token_ids = self.process_prompt(
                text, prompt_speech_path, prompt_text
            )
            
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        return model_inputs, global_token_ids

    def update_text_in_tokenized_inputs(
        self,
        model_inputs: torch.Tensor,
        new_text: str,
        is_control_mode: bool = False,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
    ) -> torch.Tensor:
        """
        Updates only the text portion in pre-tokenized inputs.
        
        Args:
            model_inputs: Previously tokenized inputs
            new_text: New text to inject
            is_control_mode: Whether inputs were created with control parameters
            gender, pitch, speed: Control parameters (needed only if is_control_mode=True)
            
        Returns:
            torch.Tensor: Updated model inputs
        """
        # Decode the tokenized inputs back to text
        prompt = self.tokenizer.decode(model_inputs.input_ids[0], skip_special_tokens=False)
        # print(prompt)
        
        # Replace the text between content markers
        if is_control_mode:
            new_prompt = self.process_prompt_control(gender, pitch, speed, new_text)
        else:
            # Extract parts before and after content
            start_content_idx = prompt.find("<|start_content|>")
            end_content_idx = prompt.find("<|end_content|>")
            
            if start_content_idx != -1 and end_content_idx != -1:
                prefix = prompt[:start_content_idx + len("<|start_content|>")]
                suffix = prompt[end_content_idx:]
                new_prompt = prefix + new_text + suffix
        
        # Re-tokenize with the new text
        return self.tokenizer([new_prompt], return_tensors="pt").to(self.device)

    @torch.no_grad()
    def inference(
        self,
        text: str = None,
        prompt_speech_path: Path = None,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 3000,
        do_sample: bool = True,
        model_inputs: torch.Tensor = None,
        global_token_ids: torch.Tensor = None,
        collect_timing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str, optional): The text input to be converted to speech.
            prompt_speech_path (Path, optional): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str, optional): female | male.
            pitch (str, optional): very_low | low | moderate | high | very_high
            speed (str, optional): very_low | low | moderate | high | very_high
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Default is 3000.
            do_sample (bool, optional): Whether to use sampling for generation. Default is True.
            model_inputs (torch.Tensor, optional): Pre-tokenized inputs to use instead of text.
            global_token_ids (torch.Tensor, optional): Pre-generated global token IDs.
            collect_timing (bool, optional): Whether to collect and return timing data.

        Returns:
            Tuple: Containing (waveform, model_inputs, global_token_ids, timing_data)
        """
        timing_data = {} if collect_timing else None
        is_control_mode = gender is not None
        
        # Use pre-built tokenized inputs or generate new ones
        if model_inputs is None:
            if text is None:
                raise ValueError("Either text or model_inputs must be provided")
                
            if collect_timing:
                tokenization_start = time.time()
                
            model_inputs, global_token_ids = self.tokenize_inputs(
                text, prompt_speech_path, prompt_text, gender, pitch, speed
            )
            
            if collect_timing:
                timing_data["tokenization_time"] = time.time() - tokenization_start

        # Generate speech using the model
        if collect_timing:
            generation_start = time.time()
            
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        
        if collect_timing:
            timing_data["model_generation_time"] = time.time() - generation_start
            token_extraction_start = time.time()

        # Trim the output tokens to remove the input tokens
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the generated tokens into text
        predicts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Extract semantic token IDs from the generated text
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
            .long()
            .unsqueeze(0)
        )

        if is_control_mode and global_token_ids is None:
            global_token_ids = (
                torch.tensor([int(token) for token in re.findall(r"bicodec_global_(\d+)", predicts)])
                .long()
                .unsqueeze(0)
                .unsqueeze(0)
            )
            
        if collect_timing:
            timing_data["token_extraction_time"] = time.time() - token_extraction_start
            audio_conversion_start = time.time()

        # Convert semantic tokens back to waveform
        wav = self.audio_tokenizer.detokenize(
            global_token_ids.to(self.device).squeeze(0),
            pred_semantic_ids.to(self.device),
        )
        
        if collect_timing:
            timing_data["audio_conversion_time"] = time.time() - audio_conversion_start
            
            # Add total time and calculate percentages
            total_time = sum(timing_data.values())
            timing_data["total_time"] = total_time
            
            for key in list(timing_data.keys()):
                if key != "total_time":
                    timing_data[f"{key}_percent"] = (timing_data[key] / total_time) * 100
            
            # Identify the heaviest operation
            heaviest_op = max([(k, v) for k, v in timing_data.items() 
                              if k not in ["total_time"] and not k.endswith("_percent")], 
                              key=lambda x: x[1])
            timing_data["heaviest_operation"] = heaviest_op[0]
            timing_data["heaviest_operation_time"] = heaviest_op[1]
            timing_data["heaviest_operation_percent"] = (heaviest_op[1] / total_time) * 100

        return wav, model_inputs, global_token_ids, timing_data

    def benchmark_performance(
        self,
        text_list: list,
        prompt_speech_path: Path = None,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
        num_runs: int = 1,
    ) -> dict:
        """
        Benchmark performance comparison between standard and optimized inference flows.
        
        Args:
            text_list (list): List of text strings to synthesize
            prompt_speech_path (Path): Path to the audio file used as a prompt
            prompt_text (str, optional): Transcript of the prompt audio
            gender (str, optional): female | male
            pitch (str, optional): very_low | low | moderate | high | very_high
            speed (str, optional): very_low | low | moderate | high | very_high
            temperature (float, optional): Sampling temperature
            top_k (float, optional): Top-k sampling parameter
            top_p (float, optional): Top-p sampling parameter
            num_runs (int, optional): Number of benchmark runs to average
            
        Returns:
            dict: Dictionary containing timing results
        """
        import time
        
        # Benchmark results
        results = {
            "standard_flow": {"total_time": 0, "per_utterance": [], "timing_details": []},
            "optimized_flow": {"total_time": 0, "per_utterance": [], "timing_details": []},
            "tokenization_time": 0,
            "speedup_factor": 0,
        }
        
        for run in range(num_runs):
            # Standard flow (tokenize each time)
            standard_start = time.time()
            for text in text_list:
                utterance_start = time.time()
                _, _, _, timing_data = self.inference(
                    text=text,
                    prompt_speech_path=prompt_speech_path,
                    prompt_text=prompt_text,
                    gender=gender,
                    pitch=pitch,
                    speed=speed,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    collect_timing=True,
                )
                utterance_time = time.time() - utterance_start
                results["standard_flow"]["per_utterance"].append(utterance_time)
                results["standard_flow"]["timing_details"].append(timing_data)
            standard_total = time.time() - standard_start
            results["standard_flow"]["total_time"] += standard_total / num_runs
            
            # Optimized flow (reuse tokenized inputs)
            optimized_start = time.time()
            
            # Initial tokenization
            tokenize_start = time.time()
            model_inputs, global_token_ids = self.tokenize_inputs(
                text=text_list[0],
                prompt_speech_path=prompt_speech_path,
                prompt_text=prompt_text,
                gender=gender,
                pitch=pitch,
                speed=speed,
            )
            tokenization_time = time.time() - tokenize_start
            results["tokenization_time"] += tokenization_time / num_runs
            
            # First utterance
            utterance_start = time.time()
            _, model_inputs, global_token_ids, timing_data = self.inference(
                model_inputs=model_inputs,
                global_token_ids=global_token_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                collect_timing=True,
            )
            utterance_time = time.time() - utterance_start
            results["optimized_flow"]["per_utterance"].append(utterance_time)
            results["optimized_flow"]["timing_details"].append(timing_data)
            
            # Subsequent utterances
            for text in text_list[1:]:
                utterance_start = time.time()
                
                # Update text in existing tokenized inputs if needed
                is_control_mode = gender is not None
                if is_control_mode:
                    updated_inputs = self.update_text_in_tokenized_inputs(
                        model_inputs, text, True, gender, pitch, speed
                    )
                else:
                    updated_inputs = self.update_text_in_tokenized_inputs(
                        model_inputs, text
                    )
                
                # Generate with updated inputs
                _, updated_inputs, _, timing_data = self.inference(
                    model_inputs=updated_inputs,
                    global_token_ids=global_token_ids,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    collect_timing=True,
                )
                
                utterance_time = time.time() - utterance_start
                results["optimized_flow"]["per_utterance"].append(utterance_time)
                results["optimized_flow"]["timing_details"].append(timing_data)
                model_inputs = updated_inputs
                
            optimized_total = time.time() - optimized_start
            results["optimized_flow"]["total_time"] += optimized_total / num_runs
        
        # Calculate average per-utterance times
        results["standard_flow"]["avg_per_utterance"] = sum(results["standard_flow"]["per_utterance"]) / len(results["standard_flow"]["per_utterance"])
        results["optimized_flow"]["avg_per_utterance"] = sum(results["optimized_flow"]["per_utterance"]) / len(results["optimized_flow"]["per_utterance"])
        
        # Calculate speedup factor
        if results["standard_flow"]["total_time"] > 0:
            results["speedup_factor"] = results["standard_flow"]["total_time"] / results["optimized_flow"]["total_time"]
        
        # Include some detailed metrics
        results["first_utterance_overhead"] = results["tokenization_time"]
        results["subsequent_utterance_speedup"] = (
            results["standard_flow"]["avg_per_utterance"] / 
            (sum(results["optimized_flow"]["per_utterance"][1:]) / max(1, len(results["optimized_flow"]["per_utterance"])-1))
            if len(results["optimized_flow"]["per_utterance"]) > 1 else 0
        )
        
        # Analyze operations to determine the heaviest
        results["operation_analysis"] = self._analyze_operations(results)
        
        return results
        
    def _analyze_operations(self, benchmark_results):
        """Analyze the benchmark results to determine the heaviest operations."""
        # Extract timing details from the benchmark
        standard_timing = benchmark_results["standard_flow"]["timing_details"]
        optimized_timing = benchmark_results["optimized_flow"]["timing_details"]
        
        # Initialize analysis results
        analysis = {
            "standard_flow": self._aggregate_operation_timings(standard_timing),
            "optimized_flow": self._aggregate_operation_timings(optimized_timing),
        }
        
        return analysis
        
    def _aggregate_operation_timings(self, timing_details):
        """Aggregate operation timings across multiple runs."""
        if not timing_details:
            return {}
            
        # Initialize aggregation
        aggregated = {}
        operation_count = {}
        
        # Sum up all times for each operation
        for timing in timing_details:
            if not timing:
                continue
                
            for op, time_value in timing.items():
                if op.endswith("_percent") or op in ["total_time", "heaviest_operation", "heaviest_operation_time", "heaviest_operation_percent"]:
                    continue
                    
                if op not in aggregated:
                    aggregated[op] = 0
                    operation_count[op] = 0
                    
                aggregated[op] += time_value
                operation_count[op] += 1
        
        # Calculate averages
        for op in aggregated:
            if operation_count[op] > 0:
                aggregated[op] /= operation_count[op]
        
        # Calculate total time and percentages
        if aggregated:
            total_time = sum(aggregated.values())
            aggregated["total_time"] = total_time
            
            for op in list(aggregated.keys()):
                if op != "total_time":
                    aggregated[f"{op}_percent"] = (aggregated[op] / total_time) * 100
            
            # Find heaviest operation
            heaviest_op = max([(k, v) for k, v in aggregated.items() 
                              if k not in ["total_time"] and not k.endswith("_percent")], 
                              key=lambda x: x[1])
            aggregated["heaviest_operation"] = heaviest_op[0]
            aggregated["heaviest_operation_time"] = heaviest_op[1]
            aggregated["heaviest_operation_percent"] = (heaviest_op[1] / total_time) * 100
        
        return aggregated