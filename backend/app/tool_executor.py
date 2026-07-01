import os
import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger("portal-backend.tool_executor")

def resolve_safe_path(relative_path: str) -> str:
    """Resolves a relative path against the mounted workspace root and checks for traversal."""
    workspace_root = os.environ.get("WORKSPACE_DIR", "/workspace")
    if not relative_path:
        relative_path = "."
        
    joined_path = os.path.join(workspace_root, relative_path)
    normalized_path = os.path.abspath(joined_path)
    
    normalized_root = os.path.abspath(workspace_root)
    if not normalized_path.startswith(normalized_root):
        raise PermissionError(f"Access denied: Path '{relative_path}' is outside workspace boundaries.")
        
    return normalized_path

async def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Routes and executes tool calls inside the Docker container."""
    logger.info(f"Executing tool: {name} with args {args}")
    try:
        if name == "list_directory":
            path = args.get("path", ".")
            safe_path = resolve_safe_path(path)
            
            if not os.path.exists(safe_path):
                return {"error": f"Path '{path}' does not exist."}
                
            if not os.path.isdir(safe_path):
                return {"error": f"Path '{path}' is a file, not a directory. Use read_file to view its contents."}
                
            entries = []
            for item in os.listdir(safe_path):
                full_item_path = os.path.join(safe_path, item)
                is_dir = os.path.isdir(full_item_path)
                size = 0 if is_dir else os.path.getsize(full_item_path)
                entries.append({
                    "name": item,
                    "type": "directory" if is_dir else "file",
                    "size": size
                })
            return {"files": entries}

        elif name == "read_file":
            path = args.get("path")
            if not path:
                return {"error": "Missing parameter 'path'."}
            safe_path = resolve_safe_path(path)
            
            if not os.path.exists(safe_path):
                return {"error": f"File '{path}' does not exist."}
                
            if os.path.isdir(safe_path):
                return {"error": f"'{path}' is a directory, not a file. Use list_directory to see its contents."}
                
            with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return {"path": path, "content": content}

        elif name == "write_file":
            path = args.get("path")
            content = args.get("content", "")
            if not path:
                return {"error": "Missing parameter 'path'."}
            safe_path = resolve_safe_path(path)
            
            os.makedirs(os.path.dirname(safe_path), exist_ok=True)
            
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(content)
                
            return {"success": True, "message": f"Successfully wrote {len(content)} characters to '{path}'."}

        elif name == "edit_file":
            path = args.get("path")
            old_string = args.get("old_string")
            new_string = args.get("new_string")
            
            if not path or old_string is None or new_string is None:
                return {"error": "Missing 'path', 'old_string', or 'new_string'."}
                
            safe_path = resolve_safe_path(path)
            
            if not os.path.exists(safe_path):
                return {"error": f"File '{path}' does not exist."}
                
            with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                
            matches = content.count(old_string)
            if matches == 0:
                return {"error": f"Could not find exact match for 'old_string' in '{path}'."}
            elif matches > 1:
                return {"error": f"Found {matches} matches for 'old_string' in '{path}'. Please provide more context to uniquely identify the match."}
                
            updated_content = content.replace(old_string, new_string)
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(updated_content)
                
            return {"success": True, "message": f"Successfully updated '{path}'."}

        elif name == "execute_command":
            command = args.get("command")
            if not command:
                return {"error": "Missing parameter 'command'."}
                
            workspace_root = os.environ.get("WORKSPACE_DIR", "/workspace")
            logger.info(f"Running command in {workspace_root}: {command}")
            
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace_root
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
                stdout_str = stdout.decode("utf-8", errors="replace")
                stderr_str = stderr.decode("utf-8", errors="replace")
                return {
                    "exit_code": proc.returncode,
                    "stdout": stdout_str,
                    "stderr": stderr_str
                }
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                return {"error": "Command timed out after 30 seconds."}

        elif name == "create_directory":
            path = args.get("path")
            if not path:
                return {"error": "Missing parameter 'path'."}
                
            safe_path = resolve_safe_path(path)
            
            if os.path.exists(safe_path):
                return {"error": f"Directory '{path}' already exists."}
                
            os.makedirs(safe_path, exist_ok=True)
            return {"success": True, "message": f"Successfully created directory '{path}'."}

        elif name == "delete_file":
            path = args.get("path")
            if not path:
                return {"error": "Missing parameter 'path'."}
                
            safe_path = resolve_safe_path(path)
            
            if not os.path.exists(safe_path):
                return {"error": f"Path '{path}' does not exist."}
                
            if os.path.isdir(safe_path):
                if os.listdir(safe_path):
                    return {"error": f"Directory '{path}' is not empty. Use execute_command to remove with 'rm -rf'."}
                os.rmdir(safe_path)
            else:
                os.remove(safe_path)
                
            return {"success": True, "message": f"Successfully deleted '{path}'."}

        elif name == "move_file":
            source = args.get("source")
            destination = args.get("destination")
            
            if not source or not destination:
                return {"error": "Missing 'source' or 'destination'."}
                
            safe_source = resolve_safe_path(source)
            safe_dest = resolve_safe_path(destination)
            
            if not os.path.exists(safe_source):
                return {"error": f"Source path '{source}' does not exist."}
                
            if os.path.exists(safe_dest):
                return {"error": f"Destination '{destination}' already exists."}
                
            os.rename(safe_source, safe_dest)
            return {"success": True, "message": f"Successfully moved '{source}' to '{destination}'."}

        elif name == "spawn_agent":
            description = args.get("description")
            if not description:
                return {"error": "Missing parameter 'description'."}
                
            from backend.app.agents import spawn_job
            job_id = spawn_job(description)
            return {
                "success": True,
                "job_id": job_id,
                "message": f"Background agent spawned with ID: {job_id}. "
                           "You can check its status using get_agent_status."
            }

        elif name == "get_agent_status":
            job_id = args.get("job_id")
            if not job_id:
                return {"error": "Missing parameter 'job_id'."}
                
            from backend.app.agents import get_job_status
            status = get_job_status(job_id)
            if not status:
                return {"error": f"No background job found with ID '{job_id}'."}
            return status

        elif name == "list_agents":
            from backend.app.agents import get_all_jobs
            jobs = get_all_jobs()
            active = [j for j in jobs if j["status"] in ("pending", "running")]
            completed = [j for j in jobs if j["status"] == "completed"]
            failed = [j for j in jobs if j["status"] == "failed"]
            return {
                "active": active,
                "completed": completed,
                "failed": failed,
                "summary": {
                    "active_count": len(active),
                    "completed_count": len(completed),
                    "failed_count": len(failed)
                }
            }

        else:
            return {"error": f"Unknown tool: '{name}'"}

    except Exception as e:
        logger.error(f"Error executing tool '{name}': {e}", exc_info=True)
        return {"error": f"Tool execution failed: {str(e)}"}
