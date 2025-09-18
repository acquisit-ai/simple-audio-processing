#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Whisper 模型配置文件
包含所有可用的 Replicate 模型及其参数配置
"""

MODELS = {
    1: {
        "name": "WhisperX - Faster Whisper",
        "version": "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb",
        "audio_param": "audio_file",
        "default_params": {
            "debug": False,
            "vad_onset": 0.5,
            "batch_size": 64,
            "vad_offset": 0.363,
            "diarization": False,
            "temperature": 0,
            "align_output": True,
            "language_detection_min_prob": 0,
            "language_detection_max_tries": 5,
        }
    },
    2: {
        "name": "OpenAI Whisper",
        "version": "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e",
        "audio_param": "audio",
        "default_params": {
            "language": "auto",
            "translate": False,
            "temperature": 0,
            "transcription": "plain text",
            "suppress_tokens": "-1",
            "logprob_threshold": -1,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 2.4,
            "temperature_increment_on_fallback": 0.2
        }
    },
    3: {
        "name": "Whisper Diarization",
        "version": "thomasmol/whisper-diarization:1495a9cddc83b2203b0d8d3516e38b80fd1572ebc4bc5700ac1da56a9b3ed886",
        "audio_param": "file",
        "default_params": {
            "prompt": "",
            "file_url": "",
            "language": "en",
            "translate": False,
            "num_speakers": 1,
            "group_segments": True
        }
    },
    4: {
        "name": "Whisper Timestamped",
        "version": "villesau/whisper-timestamped:c5b122b7e513b1b5a6ef849891c538869b77cc932cbd0f8203e11d3b357553b8",
        "audio_param": "audio_file",
        "default_params": {
            "vad": True,
            "task": "transcribe",
            "verbose": False,
            "language": "en",
            "temperature": 0,
            "suppress_tokens": "-1",
            "logprob_threshold": -1,
            "detect_disfluencies": False,
            "no_speech_threshold": 0.6,
            "compute_word_confidence": True,
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 2.4
        }
    },
    5: {
        "name": "Whisper Diarization Advanced",
        "version": "rafaelgalle/whisper-diarization-advanced:a910390c4ee857e01b94c75132594ec740da51cefadf89c109f2ba55f5b1a7b7",
        "audio_param": "file_path",
        "default_params": {
            "prompt": "",
            "num_speakers": 1,
            "translate": False,
            "preprocess": 0,
            "stationary": True,
            "target_dBFS": -18,
            "lowpass_freq": 8000,
            "highpass_freq": 45,
            "prop_decrease": 0.3
        }
    }
}