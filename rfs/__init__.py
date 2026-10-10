"""Optional PaperBanana scientific figure exchange. No additional agent."""
from .contract import RFSValidationError, SCHEMA_VERSION, validate_figure_spec, scientific_fingerprint

__all__ = ["RFSValidationError", "SCHEMA_VERSION", "validate_figure_spec", "scientific_fingerprint"]
from .artifacts import export_bundle, verify_bundle
from .prompt_compiler import CompiledPrompt, compile_prompt, compile_comparison_prompt

__all__ += ["export_bundle", "verify_bundle", "CompiledPrompt", "compile_prompt", "compile_comparison_prompt"]
