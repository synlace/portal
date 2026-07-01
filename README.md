# portal
Call your AI assistant.

## Core Features
- **Low-Latency Voice Chat:** Directly integrated with OpenAI's Realtime API using standard WebRTC peer connections for instantaneous, high-fidelity bidirectional spoken conversation with minimal latency.
- **Local Workspace Mount:** Users mount any directory into the container, giving the AI immediate tool access (`list_directory`, `read_file`, `write_file`, `edit_file`, `execute_command`) to view, modify, and test files locally.
- **File System Operations:** Full CRUD capabilities on files and directories including create, read, update, delete, move, and rename operations.
- **Autonomous Async Agents:** Delegates complex, long-running engineering tasks (such as refactoring files or writing comprehensive tests) to background agents who run independently, while you continue speaking in real-time.
- **Dual Text/Voice Mode:** Features a gorgeous, interactive web panel for chatting via microphone or keyboard while displaying live stdout execution codes and agent thought/action logs.

## Quickstart

### Environment Variables
- **`MODEL_NAME`**: Specify the AI model for background tasks (defaults to the standard model).

### Option 1: Direct Docker Run
Build and start the portal container while mounting your current directory into the `/workspace` mount:
```bash
# Build the container
docker build -t portal .

# Start the container
docker run -p 8000:8000 -v $(pwd):/workspace -e OPENAI_API_KEY="your_api_key_here" portal
```

### Option 2: Docker Compose
If you prefer Docker Compose:
```bash
export OPENAI_API_KEY="your_api_key_here"
docker compose up --build
```

### Accessing the Portal
Once launched, open your web browser to:
**`http://localhost:8000`**

- If `OPENAI_API_KEY` was not supplied in the environment, you can safely paste your key directly in the top-right field of the Web UI to initiate the session!
- Press **Connect** to initialize the bidirectional voice stream and start collaborating with your AI assistant.

## Tools

### Direct Execution (Synchronous)
These tools execute immediately and return results in the conversation:

| Tool | Description |
|------|-------------|
| `list_directory` | List files and subdirectories in the workspace |
| `read_file` | Read the full content of a file |
| `write_file` | Create a new file or overwrite an existing file |
| `edit_file` | Apply find-and-replace text replacement in a file |
| `execute_command` | Execute a terminal command in the workspace |
| `create_directory` | Create a new directory |
| `delete_file` | Delete a file or empty directory |
| `move_file` | Rename or move a file or directory |

### Delegated Execution (Asynchronous)
These tools spawn background agents that run independently:

| Tool | Description |
|------|-------------|
| `spawn_agent` | Start a background agent for complex tasks |
| `get_agent_status` | Check status, logs, and results of a background agent |
| `list_agents` | List all background agents with status summary |

## Architecture & Technology Decisions

### Logging Executed Commands
The system now logs raw shell commands executed via `execute_command`. This helps users track operations performed in the background agents.

### 1. Unified WebRTC Integration (REST + SDP)
To provide the lowest possible audio latency and eliminate complex WebSocket framing/re-sampling in Python, portal uses a unified WebRTC connection model:
- The client (browser) captures local audio via `getUserMedia` and generates a local SDP offer.
- It POSTs this offer to `/api/session` on our backend FastAPI server.
- The FastAPI server injects system instructions, tool declarations, server VAD configuration, and outputs a multipart form containing the SDP offer and session configurations to OpenAI's `/v1/realtime/calls` API.
- The returned SDP answer is sent back to the browser to establish an ultra-low latency, bidirectional peer-to-peer audio and data session (`oai-events` data channel).

### 2. Native VAD & Interruptions
Instead of fragile, CPU-heavy browser-side threshold calculations, portal delegates Voice Activity Detection (VAD) entirely to OpenAI's servers (`server_vad`).
- When server-side VAD detects speech start, OpenAI automatically interrupts and halts any active audio outputs.
- To ensure standard WebRTC echo-cancellation operates flawlessly, browser-native hardware constraints are specified on microphone acquisition to filter speaker bleedback.

### 3. Strict Server-Side Item ID Routing & Chronological Turn Sorting
Due to network package jitter and asynchronous audio translation (Whisper running asynchronously to prompt generation), user STT transcripts may arrive after the model has already begun responding. To keep logs perfectly in order:
- **Conversation Item Tracking:** The client tracks all messages using unique, server-assigned `item_id`s rather than positional indexes in the list, preventing independent turns or message deltas from merging into each other.
- **Timeline Splicing:** On user item creation (`conversation.item.created`), if a model response is already active, the client splices the transcribing user placeholder *before* the active model bubble.
- **Dynamic Chronological Sorting:** Before rendering, messages are sorted chronologically based on the precise timestamp the user stopped speaking (recorded via `input_audio_buffer.speech_stopped`). A tie-breaker rule guarantees user statements always render before model responses within any given turn.

### 4. Execution Models
Portal supports two distinct execution models:

**Direct Execution (Synchronous):**
- The AI performs an action and returns immediately
- Tool calls are processed in the conversation thread
- Best for quick operations: reading files, checking status, running short commands

**Delegated Execution (Asynchronous):**
- The AI spawns a background agent that runs independently
- User continues talking while the agent works
- Best for complex/long-running tasks: refactoring modules, writing tests, multi-file changes
- Progress and results visible in the Background Agents sidebar

### Model Selection
To specify a model for background tasks, set the `MODEL_NAME` environment variable in your `.env` file or in tools.json at the appropriate place:
```bash
echo "MODEL_NAME="your_desired_model_name"" >> .env
```

## Command Logging
To enable raw command logging, modify the `log_commands` flag in your configuration. This will log all shell commands executed through `execute_command`.

| Term | Definition |
|------|------------|
| **assistant** | The primary AI you interact with via voice or text |
| **agent** | A background worker spawned for complex tasks |
| **job** | A unit of work assigned to a background agent |
| **workspace** | The mounted directory (`/workspace`) that tools operate on |
| **tool** | A function callable by the AI for file/command operations |
| **direct execution** | Synchronous tool calls that return immediately |
| **delegated execution** | Asynchronous work handed off to background agents |
