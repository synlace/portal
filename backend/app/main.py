import os
import json
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

# Load .env file from workspace root if it exists
load_dotenv(os.path.join(os.environ.get("WORKSPACE_DIR", "/workspace"), ".env"))

from backend.app.agents import get_all_jobs
from backend.app.tool_executor import execute_tool

# Load tool definitions from shared config
TOOLS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "tools.json")
with open(TOOLS_PATH) as f:
    TOOLS = json.load(f)

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
