import os
import json
import random
import asyncio
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List
from smolagents import ToolCallingAgent, OpenAIModel

from backend.app.agent_tools import get_agent_tools

# In-memory dictionary tracking background jobs
JOBS: Dict[str, Dict[str, Any]] = {}

# Threading events for job cancellation
_CANCEL_EVENTS: Dict[str, threading.Event] = {}

logger = logging.getLogger("portal-backend.agents")

# Default system prompt (used as fallback if no profile matches)
DEFAULT_AGENT_PROMPT = (
    "You are an autonomous background software engineering agent called 'portal-agent'.\n"
    "Your objective is to complete the user's requested task thoroughly and correctly by using your tools.\n"
    "You run asynchronously, which means you can take multiple steps to implement code, "
    "read files, verify with bash commands/tests, and refine implementation.\n\n"
    "OPERATIONAL INSTRUCTIONS:\n"
    "1. Read relevant code first using 'read_file' to understand existing structure.\n"
    "2. Execute bash tests or run check scripts to verify your changes actually compile or pass tests.\n"
    "3. When you have completed the task, output a final message summarizing exactly what was done.\n"
    "4. End your response with the marker: 'JOB_COMPLETED: [summary]' to signal completion."
)

# Word lists for friendly job names
ADJECTIVES = [
    "happy", "brave", "calm", "eager", "fair", "grand", "kind", "bold",
    "cool", "dawn", "east", "fast", "gold", "high", "keen", "light",
    "moon", "neat", "past", "quick", "rare", "safe", "true", "warm",
    "blue", "dark", "deep", "early", "free", "green", "hint", "just",
    "keen", "last", "mild", "noble", "open", "prime", "real", "sharp",
    "thick", "vast", "white", "young", "amber", "coral", "ivory", "azure"
]

NOUNS = [
    "bear", "bird", "cloud", "dream", "flame", "grove", "hawk", "isle",
    "jade", "kite", "lake", "moon", "nest", "oak", "pine", "rain",
    "star", "tide", "vale", "wave", "wind", "arch", "beam", "bolt",
    "cave", "dawn", "edge", "fern", "gate", "hill", "iron", "jewel",
    "knot", "leaf", "mask", "opal", "port", "reef", "sage", "tusk",
    "vine", "wolf", "yarn", "zinc", "maple", "cedar", "ridge", "stone"
]


def _generate_friendly_name() -> str:
    """Generate a random two-word friendly name like 'brave-bear'."""
    return f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"


def _read_opencode_config() -> Dict[str, Any]:
    """Read the full opencode.json config."""
    config_path = os.path.join(os.environ.get("WORKSPACE_DIR", "/workspace"), "opencode.json")
    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read opencode.json: {e}")
        return {}


def get_model_output_limit(model: str) -> int | None:
    """Read opencode.json and return the output token limit for the given model."""
    config = _read_opencode_config()
    models = config.get("provider", {}).get("openrouter", {}).get("models", {})
    return models.get(model, {}).get("limit", {}).get("output")


def get_agent_profile(mode: str) -> Dict[str, Any]:
    """Get agent profile from opencode.json by mode name.
    
    Returns dict with 'instructions' and 'max_steps' keys.
    Falls back to defaults if profile not found.
    """
    config = _read_opencode_config()
    profiles = config.get("agents", {})
    
    # Try to get the requested profile
    profile = profiles.get(mode, {})
    
    if profile:
        return {
            "instructions": profile.get("instructions", DEFAULT_AGENT_PROMPT),
            "max_steps": profile.get("max_steps", 8),
        }
    
    # Fallback to standard profile if it exists, otherwise use defaults
    standard = profiles.get("standard", {})
    return {
        "instructions": standard.get("instructions", DEFAULT_AGENT_PROMPT),
        "max_steps": standard.get("max_steps", 8),
    }


def spawn_job(description: str, mode: str = "standard") -> str:
    """Spawns an async background agent and returns its friendly name ID.
    
    Args:
        description: The task description for the agent
        mode: Agent profile mode ('quick', 'standard', 'thorough')
    """
    job_id = _generate_friendly_name()
    now_str = datetime.utcnow().isoformat()

    # Create cancellation event for this job
    cancel_event = threading.Event()
    _CANCEL_EVENTS[job_id] = cancel_event

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "description": description,
        "mode": mode,
        "logs": ["Job initialized in background queue."],
        "result": "",
        "created_at": now_str,
        "updated_at": now_str
    }

    # Start the async runner in the background
    asyncio.create_task(run_agent_loop(job_id, description, mode))
    return job_id


def cancel_job(job_id: str) -> bool:
    """Cancel a running job. Returns True if successful."""
    if job_id not in JOBS:
        return False

    job = JOBS[job_id]
    if job["status"] not in ("pending", "running"):
        return False

    # Signal the cancellation event
    if job_id in _CANCEL_EVENTS:
        _CANCEL_EVENTS[job_id].set()

    job["status"] = "cancelled"
    job["updated_at"] = datetime.utcnow().isoformat()
    job["logs"].append({"type": "cancelled", "text": "Job cancelled by user."})
    return True


def get_job_status(job_id: str) -> Dict[str, Any]:
    """Returns the status and logs of a specific background job."""
    return JOBS.get(job_id, {})


def get_all_jobs() -> List[Dict[str, Any]]:
    """Returns a list of all background jobs sorted by creation date."""
    return sorted(JOBS.values(), key=lambda j: j["created_at"], reverse=True)


def _check_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled."""
    if job_id in _CANCEL_EVENTS:
        return _CANCEL_EVENTS[job_id].is_set()
    return False


def _cleanup_cancel_event(job_id: str):
    """Clean up the cancellation event for a completed job."""
    if job_id in _CANCEL_EVENTS:
        del _CANCEL_EVENTS[job_id]


async def run_agent_loop(job_id: str, description: str, mode: str = "standard"):
    """Execute agent loop using smolagents ToolCallingAgent."""
    try:
        # Sub-agent model configuration via environment variables
        agent_model = os.environ.get("AGENT_MODEL", "xiaomi/mimo-v2.5")
        agent_api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        agent_base_url = os.environ.get("AGENT_BASE_URL", "https://openrouter.ai/api/v1")

        if not agent_api_key or agent_api_key in ("your_api_key", "your_api_key_here"):
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["logs"].append({"type": "error", "text": "No API key configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env"})
            return

        # Check if cancelled before starting
        if _check_cancelled(job_id):
            return

        # Get agent profile from opencode.json
        profile = get_agent_profile(mode)
        agent_instructions = profile["instructions"]
        max_steps = profile["max_steps"]

        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["updated_at"] = datetime.utcnow().isoformat()
        JOBS[job_id]["logs"].append({"type": "start", "model": agent_model, "mode": mode})

        # Get output limit from opencode.json
        output_limit = get_model_output_limit(agent_model)

        # Initialize model with config from opencode.json
        model_kwargs = {
            "model_id": agent_model,
            "api_base": agent_base_url,
            "api_key": agent_api_key,
        }
        if output_limit:
            model_kwargs["max_tokens"] = output_limit
            JOBS[job_id]["logs"].append({"type": "config", "text": f"Applied output limit from opencode.json: {output_limit}"})

        model = OpenAIModel(**model_kwargs)

        # Initialize agent with tools
        tools = get_agent_tools()
        agent = ToolCallingAgent(
            tools=tools,
            model=model,
            max_steps=max_steps,
            instructions=agent_instructions,
        )

        # Run agent in a thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(agent.run, description)

        # Check if cancelled during execution
        if _check_cancelled(job_id):
            return

        # Check for JOB_COMPLETED marker in result
        if "JOB_COMPLETED" in str(result):
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["result"] = str(result)
            job_completed_idx = str(result).find("JOB_COMPLETED:")
            if job_completed_idx != -1:
                summary = str(result)[job_completed_idx + len("JOB_COMPLETED:"):].strip()
                JOBS[job_id]["logs"].append({"type": "summary", "text": summary})
        else:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["result"] = str(result)
            JOBS[job_id]["logs"].append({"type": "summary", "text": str(result)})

    except Exception as e:
        # Don't mark as failed if cancelled
        if _check_cancelled(job_id):
            return

        logger.error(f"Error in job {job_id}: {e}", exc_info=True)
        JOBS[job_id]["logs"].append({"type": "error", "text": str(e)})
        JOBS[job_id]["status"] = "failed"

    finally:
        JOBS[job_id]["updated_at"] = datetime.utcnow().isoformat()
        _cleanup_cancel_event(job_id)
