import os
import json
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file from workspace root if it exists
load_dotenv(os.path.join(os.environ.get("WORKSPACE_DIR", "/workspace"), ".env"))

from backend.app.agents import get_all_jobs, cancel_job
from backend.app.tool_executor import execute_tool

# Load tool definitions from shared config
TOOLS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "tools.json")
with open(TOOLS_PATH) as f:
    TOOLS = json.load(f)

# System instruction for streaming mode
SYSTEM_INSTRUCTION = (
    "You are 'portal', a low-latency, real-time AI software developer. "
    "You have access to the user's workspace via a mounted volume, "
    "allowing you to view and edit files, run bash commands, and spawn background agents.\n\n"
    "YOU HAVE FULL ACCESS TO SHELL COMMANDS. When the user asks you to run a command, "
    "you MUST use the 'execute_command' tool to run it and return the output.\n\n"
    "RESPONSE FORMAT RULES:\n"
    "- When the user asks you to do something that requires a tool, give a SHORT 5-10 word acknowledgment "
    "like 'Running that now.' or 'Let me check.' THEN immediately call the tool.\n"
    "- After the tool completes, give a SHORT result like 'The output is: ...' or 'Done.'\n"
    "- Keep responses concise unless the user explicitly asks for detail.\n\n"
    "AGENT SPAWN RULES:\n"
    "- When spawning background agents, just spawn them and confirm. Do NOT automatically check their status.\n"
    "- Do NOT call get_agent_status or list_agents unless the user explicitly asks.\n"
    "- The user will check agent status themselves via the Background Agents panel or by asking.\n"
    "- After spawning, just say something like 'Spawned [job_id] in [mode] mode.' and move on.\n\n"
    "Your workspace is mounted at '/workspace'. All paths must be resolved relative to this root."
)

# Convert tools to OpenAI function calling format
OPENAI_TOOLS = []
for tool_def in TOOLS:
    OPENAI_TOOLS.append({
        "type": "function",
        "function": {
            "name": tool_def["name"],
            "description": tool_def["description"],
            "parameters": tool_def["parameters"]
        }
    })

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("portal-backend")

app = FastAPI(title="portal", description="AI Developer Portal with Real-Time Audio")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes for inspecting workspace and background agents
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "workspace": os.environ.get("WORKSPACE_DIR", "/workspace")}

@app.get("/api/jobs")
async def list_jobs():
    """Retrieve all background async jobs and their statuses."""
    return get_all_jobs()

@app.get("/api/tasks")
async def list_jobs_legacy():
    """Legacy endpoint - use /api/jobs instead."""
    return get_all_jobs()

@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job_endpoint(job_id: str):
    """Cancel a running background job."""
    success = cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found or cannot be cancelled.")
    return {"status": "cancelled", "job_id": job_id}

class ToolCall(BaseModel):
    name: str
    arguments: dict

@app.post("/api/session/token")
async def create_session_token(request: Request):
    """Return the validated OpenAI API key so the frontend @openai/agents SDK can connect directly
    to OpenAI's Realtime WebRTC endpoint. portal is a local developer tool, so passing the raw key
    to the browser (via useInsecureApiKey on OpenAIRealtimeWebRTC) is acceptable and avoids the
    ephemeral-token round-trip that OpenAI's /v1/realtime/sessions does not yet expose publicly."""
    api_key = request.query_params.get("apiKey")
    if not api_key or api_key in ("your_api_key", "your_api_key_here"):
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key or api_key in ("your_api_key", "your_api_key_here"):
        logger.error("OpenAI API Key missing or invalid")
        raise HTTPException(
            status_code=400,
            detail="OpenAI API Key missing or invalid. Please set OPENAI_API_KEY environment variable or enter a valid key in the Web UI."
        )

    # Persist key so background agents spawned later can also use it
    os.environ["OPENAI_API_KEY"] = api_key

    logger.info("API key validated and returned for client-side WebRTC session")
    return {"client_secret": api_key}

@app.post("/api/session")
async def create_session(request: Request):
    """SDP proxy for the @openai/agents SDK WebRTC transport.
    The SDK POSTs a raw SDP offer (Content-Type: application/sdp). We wrap it in the
    multipart/form-data format that OpenAI's /v1/realtime/calls endpoint requires, forward
    it server-side (avoiding CORS and browser key restrictions), and return the SDP answer."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API Key not set. Call /api/session/token first.")

    sdp_offer_text = (await request.body()).decode("utf-8")
    if not sdp_offer_text.strip().startswith("v="):
        logger.error(f"Received non-SDP body (first 80 chars): {sdp_offer_text[:80]}")
        raise HTTPException(status_code=400, detail="Expected raw SDP offer in request body.")

    # Include tools and input_audio_transcription in the initial call so the model
    # knows about tools immediately at connection time.
    session_config = {
        "type": "realtime",
        "model": "gpt-realtime-2",
        "instructions": "You are 'portal', a low-latency, real-time AI software developer.",
        "tools": TOOLS,
        "audio": {
            "input": {
                "transcription": {
                    "model": "gpt-4o-mini-transcribe"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            },
            "output": {
                "voice": "alloy"
            }
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            openai_resp = await client.post(
                "https://api.openai.com/v1/realtime/calls",
                headers={"Authorization": f"Bearer {api_key}"},
                files={
                    "sdp": (None, sdp_offer_text, "text/plain"),
                    "session": (None, json.dumps(session_config), "application/json"),
                },
                timeout=30.0,
            )

            if openai_resp.status_code not in (200, 201):
                logger.error(f"OpenAI SDP exchange failed {openai_resp.status_code}: {openai_resp.text}")
                raise HTTPException(status_code=openai_resp.status_code, detail=f"OpenAI API Error: {openai_resp.text}")

            logger.info("SDP exchange with OpenAI Realtime API successful")
            return Response(content=openai_resp.text, media_type="application/sdp")

    except Exception as e:
        logger.error(f"Failed SDP proxy request: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"SDP proxy failed: {str(e)}")

# ─── Streaming Mode Endpoints ────────────────────────────────────────────────

@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert audio to text using Whisper API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API Key not set.")
    
    audio_bytes = await audio.read()
    client = OpenAI(api_key=api_key)
    
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.webm", audio_bytes, "audio/webm")
        )
        return {"text": transcript.text}
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

class TTSRequest(BaseModel):
    text: str

@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using OpenAI TTS."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API Key not set.")
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=request.text
        )
        return Response(content=response.content, media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"TTS failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

class ChatMessage(BaseModel):
    role: str
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list] = None

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: Optional[str] = "gpt-4o-mini"
    base_url: Optional[str] = None
    reasoning_effort: Optional[str] = None  # "low", "medium", "high"

# OpenAI models (static list)
OPENAI_MODELS = [
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "OpenAI"},
    {"id": "gpt-4o", "name": "GPT-4o", "provider": "OpenAI"},
    {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini", "provider": "OpenAI"},
    {"id": "gpt-4.1-nano", "name": "GPT-4.1 Nano", "provider": "OpenAI"},
]

@app.get("/api/models")
async def list_models():
    """List available models for streaming chat."""
    openrouter_models = []
    
    # Fetch OpenRouter models if API key is available
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {openrouter_key}"},
                    timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("data", []):
                        # Extract just the model name without provider prefix
                        model_id = m.get("id", "")
                        display_name = m.get("name", model_id)
                        pricing = m.get("pricing", {})
                        
                        openrouter_models.append({
                            "id": model_id,
                            "name": display_name,
                            "provider": "OpenRouter",
                            "pricing": {
                                "prompt": float(pricing.get("prompt", 0)),
                                "completion": float(pricing.get("completion", 0))
                            }
                        })
        except Exception as e:
            logger.error(f"Failed to fetch OpenRouter models: {e}")
    
    return {"models": OPENAI_MODELS + openrouter_models}

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """Stream chat completions with tool execution loop."""
    model = request.model or "gpt-4o-mini"
    
    # Determine API key and base URL based on provider
    if request.base_url:
        # OpenRouter or custom endpoint - require OPENROUTER_API_KEY
        api_key = os.environ.get("OPENROUTER_API_KEY")
        base_url = request.base_url
        if not api_key:
            raise HTTPException(
                status_code=400, 
                detail="OPENROUTER_API_KEY not set. Add it to your .env file to use OpenRouter models."
            )
    else:
        # OpenAI direct
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = None
        if not api_key:
            raise HTTPException(
                status_code=400, 
                detail="OPENAI_API_KEY not set. Add it to your .env file or enter it in the UI."
            )
    
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    
    # Build messages array with system prompt
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})
    
    async def event_generator():
        max_iterations = 10  # Prevent infinite tool call loops
        
        # Build request body
        request_body = {
            "model": model,
            "messages": messages,
            "tools": OPENAI_TOOLS,
            "stream": True
        }
        
        # Add reasoning effort for models that support it
        if request.reasoning_effort:
            request_body["reasoning"] = {"effort": request.reasoning_effort}
        
        for _ in range(max_iterations):
            try:
                # Use httpx for OpenRouter to support extra parameters like reasoning
                if base_url:
                    async with httpx.AsyncClient() as http_client:
                        response = await http_client.post(
                            f"{base_url}/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}"},
                            json=request_body,
                            timeout=60.0
                        )
                        response.raise_for_status()
                        
                        # Parse SSE stream manually
                        tool_calls = []
                        current_tool_call = None
                        text_content = []
                        
                        for line in response.text.split("\n"):
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                
                                # Text content
                                if delta.get("content"):
                                    content = delta["content"]
                                    text_content.append(content)
                                    yield f"data: {json.dumps({'type': 'text_delta', 'content': content})}\n\n"
                                
                                # Tool calls
                                if delta.get("tool_calls"):
                                    for tc_delta in delta["tool_calls"]:
                                        if tc_delta.get("index") is not None:
                                            if current_tool_call is None or tc_delta["index"] != current_tool_call.get("index"):
                                                if current_tool_call:
                                                    tool_calls.append(current_tool_call)
                                                current_tool_call = {
                                                    "index": tc_delta["index"],
                                                    "id": tc_delta.get("id", ""),
                                                    "name": "",
                                                    "arguments": ""
                                                }
                                            
                                            if tc_delta.get("id"):
                                                current_tool_call["id"] = tc_delta["id"]
                                            if tc_delta.get("function"):
                                                if tc_delta["function"].get("name"):
                                                    current_tool_call["name"] = tc_delta["function"]["name"]
                                                if tc_delta["function"].get("arguments"):
                                                    current_tool_call["arguments"] += tc_delta["function"]["arguments"]
                            except json.JSONDecodeError:
                                continue
                else:
                    # Use OpenAI SDK for direct OpenAI calls
                    stream = client.chat.completions.create(**request_body)
                    
                    tool_calls = []
                    current_tool_call = None
                    text_content = []
                    
                    for chunk in stream:
                        delta = chunk.choices[0].delta if chunk.choices else None
                        if not delta:
                            continue
                        
                        # Text content
                        if delta.content:
                            text_content.append(delta.content)
                            yield f"data: {json.dumps({'type': 'text_delta', 'content': delta.content})}\n\n"
                        
                        # Tool calls
                        if delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                if tc_delta.index is not None:
                                    # New tool call or continuation
                                    if current_tool_call is None or tc_delta.index != current_tool_call.get("index"):
                                        if current_tool_call:
                                            tool_calls.append(current_tool_call)
                                        current_tool_call = {
                                            "index": tc_delta.index,
                                            "id": tc_delta.id or "",
                                            "name": "",
                                            "arguments": ""
                                        }
                                    
                                    if tc_delta.id:
                                        current_tool_call["id"] = tc_delta.id
                                    if tc_delta.function:
                                        if tc_delta.function.name:
                                            current_tool_call["name"] = tc_delta.function.name
                                        if tc_delta.function.arguments:
                                            current_tool_call["arguments"] += tc_delta.function.arguments
                
                # Add last tool call
                if current_tool_call:
                    tool_calls.append(current_tool_call)
                
                # If no tool calls, we're done
                if not tool_calls:
                    yield "data: [DONE]\n\n"
                    break
                
                # Add assistant message with tool calls to history
                assistant_tool_calls = []
                for tc in tool_calls:
                    assistant_tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    })
                    yield f"data: {json.dumps({'type': 'tool_call', 'id': tc['id'], 'name': tc['name'], 'arguments': json.loads(tc['arguments'])})}\n\n"
                
                messages.append({
                    "role": "assistant",
                    "content": "".join(text_content) if text_content else None,
                    "tool_calls": assistant_tool_calls
                })
                
                # Execute each tool call
                for tc in tool_calls:
                    try:
                        args = json.loads(tc["arguments"])
                        result = await execute_tool(tc["name"], args)
                        result_str = json.dumps(result)
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_str
                    })
                    
                    yield f"data: {json.dumps({'type': 'tool_result', 'id': tc['id'], 'result': json.loads(result_str)})}\n\n"
                
                # Continue loop for next response after tool execution
                
            except Exception as e:
                logger.error(f"Chat stream error: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
                break
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/execute_tool")
async def execute_tool_endpoint(req: ToolCall):
    logger.info(f"Executing tool {req.name} via HTTP endpoint")
    result = await execute_tool(req.name, req.arguments)
    return result

# ─── Setup / Onboarding Endpoints ────────────────────────────────────────────

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")

@app.get("/api/setup/status")
async def setup_status():
    """Check if .portal/ directory exists in the workspace."""
    portal_dir = os.path.join(WORKSPACE_DIR, ".portal")
    graph_path = os.path.join(portal_dir, "graph.json")
    specs_dir = os.path.join(portal_dir, "specs")

    has_portal_dir = os.path.isdir(portal_dir)
    has_graph = os.path.isfile(graph_path)
    has_specs = os.path.isdir(specs_dir) and bool(os.listdir(specs_dir)) if os.path.isdir(specs_dir) else False

    graph = None
    specs = []
    if has_graph:
        try:
            with open(graph_path) as f:
                graph = json.load(f)
        except Exception:
            graph = None

    if has_specs:
        try:
            for fname in os.listdir(specs_dir):
                fpath = os.path.join(specs_dir, fname)
                if os.path.isfile(fpath) and fname.endswith(".md"):
                    with open(fpath) as f:
                        content = f.read()
                    specs.append({"filename": fname, "content": content, "lines": content.count("\n") + 1})
        except Exception:
            specs = []

    return {
        "requires_setup": not has_portal_dir,
        "has_graph": has_graph,
        "has_specs": has_specs,
        "graph": graph,
        "specs": specs,
    }


class SetupGenerateRequest(BaseModel):
    scan_src_only: bool = False
    max_depth: int = 3
    generate_specs: bool = True
    include_readme: bool = True


@app.post("/api/setup/generate")
async def setup_generate(request: SetupGenerateRequest):
    """Trigger the onboarding agent to scan the workspace and generate .portal/ artifacts.
    This runs synchronously and returns the generated graph and specs."""
    from backend.app.agents import spawn_onboarding_agent

    portal_dir = os.path.join(WORKSPACE_DIR, ".portal")
    specs_dir = os.path.join(portal_dir, "specs")
    os.makedirs(specs_dir, exist_ok=True)

    result = await spawn_onboarding_agent(
        workspace=WORKSPACE_DIR,
        scan_src_only=request.scan_src_only,
        max_depth=request.max_depth,
        generate_specs=request.generate_specs,
        include_readme=request.include_readme,
    )

    # Write graph.json
    graph_path = os.path.join(portal_dir, "graph.json")
    with open(graph_path, "w") as f:
        json.dump(result.get("graph", {"concepts": [], "specs": {}, "relationships": []}), f, indent=2)

    # Write spec files
    for spec in result.get("specs", []):
        spec_path = os.path.join(specs_dir, spec["filename"])
        with open(spec_path, "w") as f:
            f.write(spec["content"])

    return {
        "status": "complete",
        "graph": result.get("graph"),
        "specs": result.get("specs", []),
    }


@app.get("/api/graph")
async def get_graph():
    """Read the concept graph from .portal/graph.json."""
    graph_path = os.path.join(WORKSPACE_DIR, ".portal", "graph.json")
    if not os.path.isfile(graph_path):
        raise HTTPException(status_code=404, detail="No concept graph found. Run setup first.")
    with open(graph_path) as f:
        return json.load(f)


@app.put("/api/graph")
async def update_graph(request: Request):
    """Write the concept graph to .portal/graph.json."""
    body = await request.json()
    portal_dir = os.path.join(WORKSPACE_DIR, ".portal")
    os.makedirs(portal_dir, exist_ok=True)
    graph_path = os.path.join(portal_dir, "graph.json")
    with open(graph_path, "w") as f:
        json.dump(body, f, indent=2)
    return {"status": "ok"}


@app.get("/api/specs")
async def list_specs():
    """List all spec files in .portal/specs/."""
    specs_dir = os.path.join(WORKSPACE_DIR, ".portal", "specs")
    if not os.path.isdir(specs_dir):
        return {"specs": []}
    specs = []
    for fname in sorted(os.listdir(specs_dir)):
        fpath = os.path.join(specs_dir, fname)
        if os.path.isfile(fpath) and fname.endswith(".md"):
            with open(fpath) as f:
                content = f.read()
            specs.append({"filename": fname, "content": content, "lines": content.count("\n") + 1})
    return {"specs": specs}


@app.get("/api/specs/{filename}")
async def get_spec(filename: str):
    """Read a single spec file."""
    spec_path = os.path.join(WORKSPACE_DIR, ".portal", "specs", filename)
    if not os.path.isfile(spec_path):
        raise HTTPException(status_code=404, detail=f"Spec '{filename}' not found.")
    with open(spec_path) as f:
        content = f.read()
    return {"filename": filename, "content": content, "lines": content.count("\n") + 1}


@app.put("/api/specs/{filename}")
async def update_spec(filename: str, request: Request):
    """Write/update a spec file."""
    body = await request.json()
    specs_dir = os.path.join(WORKSPACE_DIR, ".portal", "specs")
    os.makedirs(specs_dir, exist_ok=True)
    spec_path = os.path.join(specs_dir, filename)
    with open(spec_path, "w") as f:
        f.write(body.get("content", ""))
    return {"status": "ok", "filename": filename}

@app.websocket("/ws")
async def legacy_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Rejected legacy WebSocket connection on /ws (WebRTC session is active)")
    try:
        await websocket.close(code=1000)
    except Exception:
        pass

# Mount static frontend files if they exist (built by Stage 1 of Dockerfile)
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir) and os.listdir(static_dir):
    logger.info(f"Mounting static frontend files from: {static_dir}")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    logger.warning(f"Static frontend files not found at: {static_dir}. Serving API only.")
    @app.get("/")
    async def index_fallback():
        return {
            "message": "portal backend running. Web frontend files not built or mounted.",
            "api_status": "healthy",
            "voice_endpoint": "/api/session"
        }
