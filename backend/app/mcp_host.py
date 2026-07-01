# Backward compatibility wrapper - use tool_executor.py directly for new code
from backend.app.tool_executor import execute_tool as execute_tool_locally, resolve_safe_path

__all__ = ["execute_tool_locally", "resolve_safe_path"]
