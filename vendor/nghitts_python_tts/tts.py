"""
Vietnamese TTS Engine
Simple wrapper around Piper TTS with Vietnamese text preprocessing
"""

import wave
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple
import sys

# Import Piper TTS
try:
    from piper import PiperVoice
except ImportError:
    print("Error: 'piper-tts' not installed. Install with: pip install piper-tts")
    sys.exit(1)

from text_cleaner import process_text_for_tts, chunk_text


class VietnameseTTS:
    """
    Vietnamese Text-to-Speech engine using Piper TTS.
    
    Features:
    - Automatic Vietnamese text preprocessing (numbers, dates, times, etc.)
    - Optional transliteration of non-Vietnamese words
    - Support for chunking long texts
    - Easy-to-use interface for both file output and audio arrays
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        use_cuda: bool = False,
        enable_transliteration: bool = True
    ):
        """
        Initialize Vietnamese TTS engine.
        
        Args:
            model_path: Path to ONNX model file (.onnx)
            config_path: Path to config file (.json). If None, assumes model_path + ".json"
            use_cuda: Whether to use CUDA (GPU) for inference
            enable_transliteration: Whether to transliterate non-Vietnamese words
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else Path(str(model_path) + ".json")
        self.enable_transliteration = enable_transliteration
        
        # Load Piper voice
        try:
            self.voice = PiperVoice.load(
                model_path=str(self.model_path),
                config_path=str(self.config_path),
                use_cuda=use_cuda
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Piper model: {e}")
        
        print(f"✓ Loaded model: {self.model_path.name}")
        print(f"✓ Sample rate: {self.voice.config.sample_rate} Hz")
        print(f"✓ Transliteration: {'enabled' if enable_transliteration else 'disabled'}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess Vietnamese text for TTS.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        return process_text_for_tts(
            text,
            enable_transliteration=self.enable_transliteration
        )
    
    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        preprocess: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text and return audio array.
        
        Args:
            text: Text to synthesize
            speaker_id: Speaker ID for multi-speaker models (optional)
            length_scale: Speech speed (< 1 = faster, > 1 = slower)
            noise_scale: Amount of noise to add (default: 0.667)
            preprocess: Whether to preprocess text (default: True)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Preprocess text if enabled
        if preprocess:
            processed_text = self.preprocess_text(text)
        else:
            processed_text = text
        
        print(f"Input: {text}")
        if preprocess and processed_text != text:
            print(f"Processed: {processed_text}")
        
        # Synthesize
        audio_arrays = []
        
        from piper import SynthesisConfig
        syn_config = SynthesisConfig(
            speaker_id=speaker_id,
            length_scale=length_scale,
            noise_scale=noise_scale
        )
        
        for audio_chunk in self.voice.synthesize(processed_text, syn_config):
            audio_arrays.append(audio_chunk.audio_float_array)
        
        # Concatenate all audio chunks
        if audio_arrays:
            audio = np.concatenate(audio_arrays)
        else:
            audio = np.array([], dtype=np.float32)
        
        return audio, self.voice.config.sample_rate
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: Union[str, Path],
        speaker_id: Optional[int] = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        preprocess: bool = True
    ) -> None:
        """
        Synthesize speech from text and save to WAV file.
        
        Args:
            text: Text to synthesize
            output_path: Path to output WAV file
            speaker_id: Speaker ID for multi-speaker models (optional)
            length_scale: Speech speed (< 1 = faster, > 1 = slower)
            noise_scale: Amount of noise to add (default: 0.667)
            preprocess: Whether to preprocess text (default: True)
        """
        audio, sample_rate = self.synthesize(
            text,
            speaker_id=speaker_id,
            length_scale=length_scale,
            noise_scale=noise_scale,
            preprocess=preprocess
        )
        
        # Convert to int16
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        
        # Save to WAV file
        output_path = Path(output_path)
        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        print(f"✓ Saved audio to: {output_path}")
    
    def speak(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        speaker_id: Optional[int] = None,
        length_scale: float = 1.0,
        preprocess: bool = True
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        High-level function to speak text. Saves to file if output_path given,
        otherwise returns audio array.
        
        Args:
            text: Text to speak
            output_path: Optional path to save audio file
            speaker_id: Speaker ID for multi-speaker models
            length_scale: Speech speed
            preprocess: Whether to preprocess text
            
        Returns:
            (audio_array, sample_rate) if no output_path, None otherwise
        """
        if output_path:
            self.synthesize_to_file(
                text,
                output_path,
                speaker_id=speaker_id,
                length_scale=length_scale,
                preprocess=preprocess
            )
            return None
        else:
            return self.synthesize(
                text,
                speaker_id=speaker_id,
                length_scale=length_scale,
                preprocess=preprocess
            )
    
    def list_speakers(self) -> dict:
        """
        Get available speakers for multi-speaker models.
        
        Returns:
            Dictionary mapping speaker names to IDs
        """
        return self.voice.config.speaker_id_map if self.voice.config.speaker_id_map else {}
