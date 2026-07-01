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

AVAILABLE_MODELS = [
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "OpenAI"},
    {"id": "gpt-4o", "name": "GPT-4o", "provider": "OpenAI"},
    {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini", "provider": "OpenAI"},
    {"id": "gpt-4.1-nano", "name": "GPT-4.1 Nano", "provider": "OpenAI"},
    {"id": "xiaomi/mimo-v2.5", "name": "Xiaomi MiMo V2.5", "provider": "OpenRouter"},
    {"id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet 4", "provider": "OpenRouter"},
    {"id": "anthropic/claude-3.5-haiku", "name": "Claude 3.5 Haiku", "provider": "OpenRouter"},
    {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "OpenRouter"},
    {"id": "meta-llama/llama-4-scout", "name": "Llama 4 Scout", "provider": "OpenRouter"},
    {"id": "deepseek/deepseek-chat-v3-0324", "name": "DeepSeek V3", "provider": "OpenRouter"},
]

@app.get("/api/models")
async def list_models():
    """List available models for streaming chat."""
    return {"models": AVAILABLE_MODELS}

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """Stream chat completions with tool execution loop."""
    model = request.model or "gpt-4o-mini"
    
    # Determine API key and base URL based on provider
    if request.base_url:
        # OpenRouter or custom endpoint
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = request.base_url
    else:
        # OpenAI direct
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = None
    
    if not api_key:
        raise HTTPException(status_code=400, detail="API key not set.")
    
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    
    # Build messages array with system prompt
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})
    
    async def event_generator():
        max_iterations = 10  # Prevent infinite tool call loops
        
        for _ in range(max_iterations):
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=OPENAI_TOOLS,
                    stream=True
                )
                
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
