"""Zero-weight-update assistant engine: activation & attention interventions on frozen base LLMs."""

from .artifact import SteeringArtifact
from .calibration import CalibConfig, calibrate
from .generator import EncodedPrompt, GenerationResult, encode_prompt, generate
from .hooks import AssistantEngine, EngineConfig
from .model_loader import LoadedModel, free_memory, load_model

__all__ = [
    "AssistantEngine", "CalibConfig", "EncodedPrompt", "EngineConfig", "GenerationResult", "LoadedModel",
    "SteeringArtifact", "calibrate", "encode_prompt", "free_memory", "generate", "load_model",
]
