import os
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List
import httpx

# In-memory dictionary tracking background jobs
JOBS: Dict[str, Dict[str, Any]] = {}

logger = logging.getLogger("portal-backend.agents")

# Define tools for background agents
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path inside the workspace. Defaults to root ('.')."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full content of a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path of the file to read."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or completely overwrite an existing file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path of the file to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write to the file."
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Apply a specific text replacement inside an existing file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
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
                },
                "required": ["path", "old_string", "new_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a terminal command in the workspace directory. Use this to run scripts, build files, run tests, or check status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact terminal command to run."
                    }
                },
                "required": ["command"]
            }
        }
    }
]

AGENT_SYSTEM_PROMPT = (
    "You are an autonomous background software engineering agent called 'portal-agent'.\n"
    "Your objective is to complete the user's requested task thoroughly and correctly by using your tools.\n"
    "You run asynchronously, which means you can take multiple steps (up to 12) to implement code, "
    "read files, verify with bash commands/tests, and refine implementation.\n\n"
    "OPERATIONAL INSTRUCTIONS:\n"
    "1. Read relevant code first using 'read_file' to understand existing structure.\n"
    "2. Execute bash tests or run check scripts to verify your changes actually compile or pass tests.\n"
    "3. When you have completed the task, output a final message summarizing exactly what was done, "
    "which files were modified, and how they were verified.\n"
    "4. End your response with the marker: 'JOB_COMPLETED: [summary]' to signal completion."
)

def spawn_job(description: str) -> str:
    """Spawns an async background agent and returns its unique ID."""
    job_id = f"job_{uuid.uuid4().hex[:6]}"
    now_str = datetime.utcnow().isoformat()
    
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "description": description,
        "logs": ["Job initialized in background queue."],
        "result": "",
        "created_at": now_str,
        "updated_at": now_str
    }
    
    # Start the async runner in the background
    asyncio.create_task(run_agent_loop(job_id, description))
    return job_id

def get_job_status(job_id: str) -> Dict[str, Any]:
    """Returns the status and logs of a specific background job."""
    return JOBS.get(job_id, {})

def get_all_jobs() -> List[Dict[str, Any]]:
    """Returns a list of all background jobs sorted by creation date."""
    return sorted(JOBS.values(), key=lambda j: j["created_at"], reverse=True)

async def run_agent_loop(job_id: str, description: str):
    """Executes the autonomous agent loop by querying OpenAI and running local tools."""
    import json
    from backend.app.tool_executor import execute_tool
    
    # Sub-agent model configuration via environment variables
    agent_model = os.environ.get("AGENT_MODEL", "xiaomi/mimo-v2.5")
    agent_api_key = os.environ.get("AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    agent_base_url = os.environ.get("AGENT_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
    
    if not agent_api_key or agent_api_key in ("your_api_key", "your_api_key_here"):
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["logs"].append("Error: No API key configured. Set AGENT_API_KEY or OPENAI_API_KEY.")
        return

    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["updated_at"] = datetime.utcnow().isoformat()
    JOBS[job_id]["logs"].append(f"Background agent started (model={agent_model}).")

    headers = {
        "Authorization": f"Bearer {agent_api_key}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Please complete the following task:\n\n{description}"}
    ]
    
    max_steps = 12
    step = 0
    
    async with httpx.AsyncClient() as client:
        while step < max_steps:
            step += 1
            JOBS[job_id]["updated_at"] = datetime.utcnow().isoformat()
            
            payload = {
                "model": agent_model,
                "messages": messages,
                "tools": AGENT_TOOLS,
                "tool_choice": "auto"
            }
            
            try:
                response = await client.post(agent_base_url, json=payload, headers=headers, timeout=45.0)
                if response.status_code != 200:
                    error_msg = f"OpenAI API returned error code {response.status_code}: {response.text}"
                    JOBS[job_id]["logs"].append(f"Error: {error_msg}")
                    JOBS[job_id]["status"] = "failed"
                    return
                
                resp_json = response.json()
                choices = resp_json.get("choices", [])
                if not choices:
                    JOBS[job_id]["logs"].append("Warning: Model returned an empty response. Stopping loop.")
                    break
                    
                message = choices[0].get("message", {})
                
                messages.append(message)
                
                content = message.get("content") or ""
                tool_calls = message.get("tool_calls") or []

                if content:
                    clean_text = content.strip()
                    JOBS[job_id]["logs"].append(f"Agent thought:\n{clean_text}")
                    
                    if "JOB_COMPLETED" in clean_text:
                        JOBS[job_id]["status"] = "completed"
                        JOBS[job_id]["result"] = clean_text
                        JOBS[job_id]["logs"].append("Job completed successfully!")
                        return

                if tool_calls:
                    for t_call in tool_calls:
                        tc_id = t_call.get("id")
                        func = t_call.get("function", {})
                        name = func.get("name")
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str)
                        except Exception:
                            args = {}
                        
                        JOBS[job_id]["logs"].append(f"Executing tool: {name}({args})")
                        
                        result = await execute_tool(name, args)
                        
                        JOBS[job_id]["logs"].append(f"Tool result: {str(result)[:200]}...")
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": name,
                            "content": json.dumps(result)
                        })
                else:
                    if step < max_steps:
                        messages.append({
                            "role": "user",
                            "content": "Please continue with the next action or output 'JOB_COMPLETED: [summary]' if you are finished."
                        })
                        JOBS[job_id]["logs"].append("Prompting agent to continue...")
                    
            except Exception as e:
                logger.error(f"Error in job {job_id} step {step}: {e}", exc_info=True)
                JOBS[job_id]["logs"].append(f"Exception during execution: {str(e)}")
                JOBS[job_id]["status"] = "failed"
                return
                
        if JOBS[job_id]["status"] == "running":
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["result"] = "Agent reached maximum execution steps without explicit JOB_COMPLETED marker."
            JOBS[job_id]["logs"].append("Agent loop finished (reached max steps limit).")
