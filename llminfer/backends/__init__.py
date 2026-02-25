"""
llminfer/backends/__init__.py
"""

from llminfer.backends.base import BaseBackend
from llminfer.backends.eager import EagerBackend
from llminfer.backends.compiled import CompiledBackend

__all__ = ["BaseBackend", "EagerBackend", "CompiledBackend"]

try:
    from llminfer.backends.vllm_backend import VLLMBackend
    __all__.append("VLLMBackend")
except ImportError:
    pass  # vllm optional
