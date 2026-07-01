import os
import subprocess
import logging
from typing import Optional
from smolagents import Tool
from backend.app.tool_executor import resolve_safe_path

logger = logging.getLogger("portal-backend.agent_tools")


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = "List files and subdirectories in the workspace."
    inputs = {
        "path": {
            "type": "string",
            "description": "Relative path inside the workspace. Defaults to root ('.').",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, path: str = ".") -> str:
        safe_path = resolve_safe_path(path)

        if not os.path.exists(safe_path):
            return f"Error: Path '{path}' does not exist."

        if not os.path.isdir(safe_path):
            return f"Error: Path '{path}' is a file, not a directory. Use read_file to view its contents."

        entries = []
        for item in os.listdir(safe_path):
            full_item_path = os.path.join(safe_path, item)
            is_dir = os.path.isdir(full_item_path)
            size = 0 if is_dir else os.path.getsize(full_item_path)
            entries.append(f"{'[DIR] ' if is_dir else ''}{item} ({size} bytes)")

        return "\n".join(entries) if entries else "Directory is empty."


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the full content of a file from the workspace."
    inputs = {
        "path": {
            "type": "string",
            "description": "Relative path of the file to read."
        }
    }
    output_type = "string"

    def forward(self, path: str) -> str:
        if not path:
            return "Error: Missing parameter 'path'."

        safe_path = resolve_safe_path(path)

        if not os.path.exists(safe_path):
            return f"Error: File '{path}' does not exist."

        if os.path.isdir(safe_path):
            return f"Error: '{path}' is a directory, not a file. Use list_directory to see its contents."

        with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create a new file or completely overwrite an existing file in the workspace."
    inputs = {
        "path": {
            "type": "string",
            "description": "Relative path of the file to write."
        },
        "content": {
            "type": "string",
            "description": "Full content to write to the file."
        }
    }
    output_type = "string"

    def forward(self, path: str, content: str) -> str:
        if not path:
            return "Error: Missing parameter 'path'."

        safe_path = resolve_safe_path(path)
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)

        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully wrote {len(content)} characters to '{path}'."


class EditFileTool(Tool):
    name = "edit_file"
    description = "Apply a specific text replacement inside an existing file in the workspace."
    inputs = {
        "path": {
            "type": "string",
            "description": "Relative path of the file to edit."
        },
        "old_string": {
            "type": "string",
            "description": "The exact block of text to find."
        },
        "new_string": {
            "type": "string",
            "description": "The text block to replace it with."
        }
    }
    output_type = "string"

    def forward(self, path: str, old_string: str, new_string: str) -> str:
        if not path or old_string is None or new_string is None:
            return "Error: Missing 'path', 'old_string', or 'new_string'."

        safe_path = resolve_safe_path(path)

        if not os.path.exists(safe_path):
            return f"Error: File '{path}' does not exist."

        with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        matches = content.count(old_string)
        if matches == 0:
            return f"Error: Could not find exact match for 'old_string' in '{path}'."
        elif matches > 1:
            return f"Error: Found {matches} matches for 'old_string' in '{path}'. Please provide more context to uniquely identify the match."

        updated_content = content.replace(old_string, new_string)
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(updated_content)

        return f"Successfully updated '{path}'."


class ExecuteCommandTool(Tool):
    name = "execute_command"
    description = "Execute a terminal command in the workspace directory. Use this to run scripts, build files, run tests, or check status."
    inputs = {
        "command": {
            "type": "string",
            "description": "The exact terminal command to run."
        }
    }
    output_type = "string"

    def forward(self, command: str) -> str:
        if not command:
            return "Error: Missing parameter 'command'."

        workspace_root = os.environ.get("WORKSPACE_DIR", "/workspace")
        logger.info(f"Running command in {workspace_root}: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=workspace_root
            )

            output = f"Exit code: {result.returncode}\n"
            if result.stdout:
                output += f"Stdout:\n{result.stdout}\n"
            if result.stderr:
                output += f"Stderr:\n{result.stderr}\n"
            return output

        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds."
        except Exception as e:
            return f"Error executing command: {str(e)}"


def get_agent_tools() -> list[Tool]:
    """Return all available agent tools."""
    return [
        ListDirectoryTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        ExecuteCommandTool(),
    ]
