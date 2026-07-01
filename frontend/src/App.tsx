import { useState, useEffect, useRef } from 'react';
import { 
  Square, Mic, MicOff,
  AlertCircle, Loader2, HelpCircle, Send, Cpu, ChevronRight, ChevronDown, Star
} from 'lucide-react';
import { RealtimeAgent, RealtimeSession, OpenAIRealtimeWebRTC, tool } from '@openai/agents/realtime';
import toolsSchema from '../../tools.json';

// ─── System prompt ───────────────────────────────────────────────────────────
const SYSTEM_INSTRUCTION =
  "You are 'portal', a low-latency, real-time AI software developer. You speak directly to the " +
  "user via a bidirectional live audio stream. You have access to their workspace via a mounted volume, " +
  "allowing you to view and edit files, run bash commands, and spawn background agents.\n\n" +
  "YOU HAVE FULL ACCESS TO SHELL COMMANDS. When the user asks you to run a command (like 'pwd', 'ls', " +
  "'cat file.txt', 'npm install', etc.), you MUST use the 'execute_command' tool to run it and return " +
  "the output. Never say you cannot run commands - you absolutely can.\n\n" +
  "RESPONSE FORMAT RULES:\n" +
  "- When the user asks you to do something that requires a tool, give a SHORT 5-10 word acknowledgment " +
  "like 'Running that now.' or 'Let me check.' THEN immediately call the tool. Do NOT give multiple responses.\n" +
  "- After the tool completes, give a SHORT result like 'The output is: ...' or 'Done. I've created the file.'\n" +
  "- NEVER generate multiple separate responses to a single user request. One acknowledgment + tool call + one result.\n" +
  "- Keep ALL spoken responses under 20 words unless the user explicitly asks for detail.\n\n" +
  "CRITICAL VOICE CONSTRAINTS:\n" +
  "1. Always keep spoken answers short, professional, and conversational. Do not ramble.\n" +
  "2. NEVER speak out loud long blocks of code, markdown tables, or extensive directories. " +
  "If the user asks you to write code, do it silently by using your tools ('write_file' or 'edit_file'), " +
  "and then briefly say: 'I've written that code to [filename]. [Brief 1-sentence summary of what it does].'\n" +
  "3. If a task is complex, multi-step, or requires running multiple bash commands, spawn a background agent " +
  "using 'spawn_agent' so we can keep talking in real-time. Give the user the job ID, and tell them you will " +
  "check back on it later.\n" +
  "4. Your workspace is mounted at '/workspace'. All paths provided to tools must be resolved relative to this root.\n" +
  "5. ALWAYS use tools to perform actions. If the user asks to read a file, use read_file. If they ask to run a command, " +
  "use execute_command. If they ask to create/modify files, use write_file or edit_file. Never claim you cannot do something " +
  "that your tools can do.";

// ─── Helper: forward tool execution to backend /api/execute_tool ─────────────
const callBackendTool = async (name: string, args: Record<string, any>): Promise<string> => {
  const resp = await fetch('/api/execute_tool', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, arguments: args }),
  });
  if (!resp.ok) throw new Error(`Tool ${name} failed with HTTP ${resp.status}`);
  const result = await resp.json();
  return JSON.stringify(result);
};

// ─── Tool definitions using SDK's tool() helper ──────────────────────────────
const TOOLS = toolsSchema.map(toolDef => 
  tool({
    name: toolDef.name,
    description: toolDef.description,
    parameters: toolDef.parameters as any,
    execute: async (args: any) => {
      return await callBackendTool(toolDef.name, args);
    },
  })
);

// ─── Group consecutive model messages into single turns ─────────────────────
interface GroupedMessage {
  type: 'system' | 'user' | 'tool' | 'model';
  messages?: Message[];
  combinedText?: string;
  message?: Message;
}

const groupMessages = (messages: Message[]): GroupedMessage[] => {
  // Deduplicate model messages with identical text (keep only the first occurrence)
  const seen = new Set<string>();
  const deduped = messages.filter(m => {
    if (m.role === 'model' && m.text) {
      const key = `model:${m.text}`;
      if (seen.has(key)) return false;
      seen.add(key);
    }
    return true;
  });

  const sorted = [...deduped].sort((a, b) => {
    const tA = a.timestamp instanceof Date ? a.timestamp.getTime() : new Date(a.timestamp).getTime();
    const tB = b.timestamp instanceof Date ? b.timestamp.getTime() : new Date(b.timestamp).getTime();
    if (Math.abs(tA - tB) < 1000) {
      if (a.role === 'user' && b.role === 'model') return -1;
      if (a.role === 'model' && b.role === 'user') return 1;
    }
    return tA - tB;
  });

  const grouped: GroupedMessage[] = [];
  let i = 0;

  while (i < sorted.length) {
    const msg = sorted[i];

    if (msg.role === 'system' || msg.role === 'tool') {
      grouped.push({ type: msg.role, message: msg });
      i++;
      continue;
    }

    if (msg.role === 'user') {
      grouped.push({ type: 'user', message: msg });
      i++;
      continue;
    }

    // For model messages, group consecutive ones within 3 seconds
    if (msg.role === 'model') {
      const modelGroup: Message[] = [msg];
      const groupStart = msg.timestamp instanceof Date ? msg.timestamp.getTime() : new Date(msg.timestamp).getTime();
      i++;

      while (i < sorted.length && sorted[i].role === 'model') {
        const nextTime = sorted[i].timestamp instanceof Date ? sorted[i].timestamp.getTime() : new Date(sorted[i].timestamp).getTime();
        if (nextTime - groupStart < 3000) {
          modelGroup.push(sorted[i]);
          i++;
        } else {
          break;
        }
      }

      // Deduplicate: only include messages whose text isn't already a substring of accumulated text
      let combinedText = '';
      for (const m of modelGroup) {
        const text = m.text || '';
        if (text && !combinedText.includes(text)) {
          combinedText += (combinedText ? ' ' : '') + text;
        }
      }
      grouped.push({ type: 'model', messages: modelGroup, combinedText });
      continue;
    }

    i++;
  }

  return grouped;
};

// ─── Types ───────────────────────────────────────────────────────────────────
interface Job {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  description: string;
  logs: any[];
  result: string;
  created_at: string;
  updated_at: string;
}

interface Message {
  role: 'user' | 'model' | 'system' | 'tool';
  text?: string;
  id?: string;
  toolName?: string;
  toolArgs?: any;
  toolStatus?: 'executing' | 'completed' | 'failed';
  toolResult?: string;
  timestamp: Date;
}

export default function App() {
  // Connection and Authentication
  const [apiKey, setApiKey] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  // Voice States
  const [isMuted] = useState(false);
  const [isModelTalking, _setIsModelTalking] = useState(false);
  const isModelTalkingRef = useRef(false);
  const setIsModelTalking = (val: boolean) => {
    _setIsModelTalking(val);
    isModelTalkingRef.current = val;
  };

  // Chat & Messages
  const [textInput, setTextInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'system',
      text: "Welcome to portal. Use the voice button or chat box to speak with your AI assistant. You can ask it to view, create, or edit files in your workspace, run commands, or spawn background agents for heavy tasks.",
      timestamp: new Date()
    }
  ]);

  // Jobs & Background agents
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [expandedAgents, setExpandedAgents] = useState<Set<string>>(new Set());

  // Connection mode
  const [connectionMode, setConnectionMode] = useState<'webrtc' | 'streaming' | 'noaudio'>('webrtc');
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Model selection for streaming mode
  const [selectedModel, setSelectedModel] = useState('gpt-4o-mini');
  const [availableModels, setAvailableModels] = useState<Array<{id: string, name: string, provider: string, pricing?: {prompt: number, completion: number}}>>([]);
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [modelSearch, setModelSearch] = useState('');
  const modelSelectorRef = useRef<HTMLDivElement>(null);
  const [favoriteModels, setFavoriteModels] = useState<string[]>(() => {
    const saved = localStorage.getItem('portal_favorite_models');
    return saved ? JSON.parse(saved) : [];
  });

  // Refs for SDK session and audio
  const sessionRef = useRef<RealtimeSession | null>(null);
  const audioElRef = useRef<HTMLAudioElement | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const userSpeechStopTimestampRef = useRef<number>(0);
  const micDrainTimeoutRef = useRef<any>(null);
  const userSpeakingRef = useRef<boolean>(false);

  // Refs for tool call tracking and text buffering
  const pendingToolCallsRef = useRef(0);
  const bufferedTextRef = useRef<{ id: string; text: string } | null>(null);
  const flushTimeoutRef = useRef<any>(null);

  // Separate mute suppression driven purely by server events, decoupled from isModelTalking visual state
  const micSuppressedRef = useRef<boolean>(false);

  const applyMuteState = () => {
    const shouldMute = isMutedRef.current || micSuppressedRef.current;
    logger(`applyMuteState: isMuted=${isMutedRef.current}, micSuppressed=${micSuppressedRef.current} -> shouldMute=${shouldMute}`);
    if (sessionRef.current) {
      sessionRef.current.mute(shouldMute);
    }
  };

  const suppressMic = () => {
    logger("suppressMic() called! Muting session to suppress microphone and prevent echo.");
    if (micDrainTimeoutRef.current) {
      clearTimeout(micDrainTimeoutRef.current);
      micDrainTimeoutRef.current = null;
    }
    micSuppressedRef.current = true;
    applyMuteState();
  };

  const unsuppressMicAfterDrain = () => {
    logger("unsuppressMicAfterDrain() called! Initiating 4-second mic drain timeout.");
    if (micDrainTimeoutRef.current) {
      clearTimeout(micDrainTimeoutRef.current);
    }
    micDrainTimeoutRef.current = setTimeout(() => {
      logger("4-second mic drain timeout completed! Re-enabling mic stream.");
      micSuppressedRef.current = false;
      micDrainTimeoutRef.current = null;
      applyMuteState();
    }, 4000);
  };

  const isMutedRef = useRef<boolean>(false);
  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  useEffect(() => {
    applyMuteState();
  }, [isMuted]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Periodically fetch background jobs
  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const res = await fetch('/api/jobs');
        if (res.ok) {
          const data = await res.json();
          setJobs(data);
        }
      } catch (err) {
        console.error('Failed to fetch background agents:', err);
      }
    };

    fetchJobs();
    const interval = setInterval(fetchJobs, 3000);
    return () => clearInterval(interval);
  }, []);

  // Fetch available models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch('/api/models');
        if (res.ok) {
          const data = await res.json();
          setAvailableModels(data.models);
        }
      } catch (err) {
        console.error('Failed to fetch models:', err);
      }
    };
    fetchModels();
  }, []);

  // Toggle model favorite
  const toggleFavoriteModel = (modelId: string) => {
    setFavoriteModels(prev => {
      const newFavorites = prev.includes(modelId)
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId];
      localStorage.setItem('portal_favorite_models', JSON.stringify(newFavorites));
      return newFavorites;
    });
  };

  // Close model selector when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (modelSelectorRef.current && !modelSelectorRef.current.contains(event.target as Node)) {
        setShowModelSelector(false);
        setModelSearch('');
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Spacebar push-to-talk in streaming mode
  useEffect(() => {
    if (connectionMode !== 'streaming') return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat && isConnected && !isRecording) {
        e.preventDefault();
        toggleRecording();
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space' && isRecording) {
        e.preventDefault();
        toggleRecording();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [connectionMode, isConnected, isRecording]);

  // Cancel a running job
  const cancelJob = async (jobId: string) => {
    try {
      const res = await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
      if (res.ok) {
        // Refresh jobs list immediately
        const jobsRes = await fetch('/api/jobs');
        if (jobsRes.ok) {
          setJobs(await jobsRes.json());
        }
      }
    } catch (err) {
      console.error('Failed to cancel job:', err);
    }
  };

  // ─── Connect via @openai/agents RealtimeSession ────────────────────────────────
  const startSession = async () => {
    if (isConnecting || isConnected) return;
    setIsConnecting(true);
    setErrorMsg('');

    // Branch on connection mode
    if (connectionMode === 'streaming') {
      await startStreamingSession();
      return;
    }

    try {
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      micStreamRef.current = stream;

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      const audioEl = document.createElement('audio');
      audioEl.autoplay = true;
      audioElRef.current = audioEl;

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const checkVolume = () => {
        if (!sessionRef.current) {
          audioCtx.close().catch(() => {});
          return;
        }
        if (audioCtx.state === 'suspended') {
          audioCtx.resume().catch(() => {});
        }

        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) sum += dataArray[i];
        const average = sum / bufferLength;
        const userSpeaking = average > 10;

        if (userSpeakingRef.current !== userSpeaking) {
          logger(`[Local Audio Transition] User speaking changed from ${userSpeakingRef.current} to ${userSpeaking} (avg: ${average.toFixed(2)})`);
          userSpeakingRef.current = userSpeaking;
        }

        if (userSpeaking && micDrainTimeoutRef.current) {
          logger("[Early Cancellation] User spoke during 4-second drain window! Cancelling and unmuting mic.");
          clearTimeout(micDrainTimeoutRef.current);
          micDrainTimeoutRef.current = null;
          micSuppressedRef.current = false;
          applyMuteState();
        }

        requestAnimationFrame(checkVolume);
      };
      checkVolume();

      const webrtcTransport = new OpenAIRealtimeWebRTC({
        baseUrl: `${window.location.origin}/api/session`,
        mediaStream: stream,
        audioElement: audioEl,
        useInsecureApiKey: true,
        changePeerConnection: (pc) => {
          const cfg = pc.getConfiguration();
          if (!cfg.iceServers?.length) {
            pc.setConfiguration({
              ...cfg,
              iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' },
              ],
            });
          }

          pc.addEventListener('track', (e) => {
            try {
              const remoteSource = audioCtx.createMediaStreamSource(e.streams[0]);
              const remoteAnalyser = audioCtx.createAnalyser();
              remoteAnalyser.fftSize = 256;
              remoteSource.connect(remoteAnalyser);

              const remoteBufferLength = remoteAnalyser.frequencyBinCount;
              const remoteDataArray = new Uint8Array(remoteBufferLength);

              const checkRemoteVolume = () => {
                if (!sessionRef.current) return;
                remoteAnalyser.getByteFrequencyData(remoteDataArray);
                let sum = 0;
                for (let i = 0; i < remoteBufferLength; i++) sum += remoteDataArray[i];
                const average = sum / remoteBufferLength;
                const modelAudible = average > 15;
                if (isModelTalkingRef.current !== modelAudible) {
                  logger(`[Remote Audio Transition] Model audible changed from ${isModelTalkingRef.current} to ${modelAudible} (avg: ${average.toFixed(2)})`);
                  setIsModelTalking(modelAudible);
                }
                requestAnimationFrame(checkRemoteVolume);
              };
              checkRemoteVolume();
            } catch (err) {
              console.error('Failed to set up remote audio analyser:', err);
            }
          });

          pc.onconnectionstatechange = () => {
            logger(`WebRTC Connection State: ${pc.connectionState}`);
            if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
              stopSession();
            }
          };

          return pc;
        },
      });

      const agent = new RealtimeAgent({
        name: 'portal',
        instructions: SYSTEM_INSTRUCTION,
        tools: TOOLS as any,
      });

      const session = new RealtimeSession(agent, {
        model: 'gpt-realtime-2',
        transport: webrtcTransport,
        config: {
          outputModalities: ['audio'],
          audio: {
            input: {
              transcription: { model: 'gpt-4o-mini-transcribe' },
              turnDetection: {
                type: 'server_vad',
                threshold: 0.5,
                prefixPaddingMs: 300,
                silenceDurationMs: 500,
              },
            },
            output: { voice: 'alloy' },
          },
        },
      });
      sessionRef.current = session;

      let modelAudioActive = false;

      // Listen directly on transport for function_call events
      session.transport.on('function_call', (event: any) => {
        logger(`transport function_call: ${event.name} callId=${event.callId}`);
        const args = JSON.parse(event.arguments || '{}');
        handleToolUpdate({ name: event.name, args, status: 'executing' });
      });

      // Also listen for item_update on transport directly
      session.transport.on('item_update', (item: any) => {
        logger(`transport item_update: ${item.type} status=${item.status} name=${item.name}`);
        if (item.type === 'function_call') {
          const args = JSON.parse(item.arguments || '{}');
          if (item.status === 'completed') {
            handleToolUpdate({ name: item.name, args, status: 'completed', result: item.output });
          }
        }
      });

      session.on('audio_stopped', () => {
        logger('SDK audio_stopped: starting 4s mic drain');
        modelAudioActive = false;
        setIsModelTalking(false);
        unsuppressMicAfterDrain();
      });

      session.on('audio_interrupted', () => {
        logger('SDK audio_interrupted: resetting mic drain');
        modelAudioActive = false;
        setIsModelTalking(false);
        unsuppressMicAfterDrain();
      });

      session.on('transport_event', (event: any) => {
        console.log('OpenAI Server Event:', event);

        if (event.type === 'input_audio_buffer.speech_stopped' || event.type === 'input_audio_buffer.committed') {
          userSpeechStopTimestampRef.current = Date.now();
        }

        if (event.type === 'response.output_audio_transcript.delta') {
          if (!modelAudioActive) {
            modelAudioActive = true;
            logger('First transcript delta — suppressing mic (WebRTC audio_start substitute)');
            setIsModelTalking(true);
            suppressMic();
          }
          const itemId = event.item_id;
          const text = event.delta;

          // If tool calls are in progress, buffer the text instead of adding to messages
          if (pendingToolCallsRef.current > 0) {
            const existing = bufferedTextRef.current;
            if (existing && existing.id === itemId) {
              // Append to existing buffer for same message
              bufferedTextRef.current = { id: itemId, text: existing.text + text };
            } else {
              // New message or different item_id - replace buffer
              bufferedTextRef.current = { id: itemId, text };
            }
            return;
          }

          // No tool calls in progress - add normally
          setMessages(prev => {
            const existingIdx = prev.findIndex(m => m.id === itemId);
            if (existingIdx !== -1) {
              const existingText = prev[existingIdx].text || '';
              // Prevent duplicate deltas - only append if delta is not already at the end
              if (existingText.endsWith(text)) {
                return prev;
              }
              const updated = [...prev];
              updated[existingIdx] = { ...updated[existingIdx], text: existingText + text };
              return updated;
            }
            return [...prev, { role: 'model', text, id: itemId, timestamp: new Date() }];
          });
        }

        else if (event.type === 'conversation.item.added' && event.item?.role === 'user') {
          const itemId = event.item.id;
          const contentPart = event.item.content?.[0];
          const isTextMsg = contentPart?.type === 'input_text';
          const textVal = contentPart?.text;

          setMessages(prev => {
            if (prev.some(m => m.id === itemId)) return prev;

            const message: Message = {
              role: 'user',
              text: isTextMsg && textVal ? textVal : '🎙️ Transcribing...',
              id: itemId,
              timestamp: isTextMsg ? new Date() : new Date(userSpeechStopTimestampRef.current || (Date.now() - 2000)),
            };

            if (isTextMsg && textVal) {
              const matchingIdx = prev.findIndex(m => m.role === 'user' && m.text === textVal && !m.id);
              if (matchingIdx !== -1) {
                const updated = [...prev];
                updated[matchingIdx] = { ...updated[matchingIdx], id: itemId };
                return updated;
              }
            }

            const lastMsg = prev[prev.length - 1];
            if (lastMsg?.role === 'model') {
              const updated = [...prev];
              updated.splice(prev.length - 1, 0, message);
              return updated;
            }
            return [...prev, message];
          });
        }

        else if (event.type === 'conversation.item.input_audio_transcription.completed') {
          const transcript = event.transcript;
          const itemId = event.item_id;
          if (transcript?.trim()) {
            setMessages(prev => {
              const idx = prev.findIndex(m => m.id === itemId);
              if (idx !== -1) {
                const updated = [...prev];
                updated[idx] = { ...updated[idx], text: transcript };
                return updated;
              }
              return [...prev, {
                role: 'user', text: transcript, id: itemId,
                timestamp: new Date(userSpeechStopTimestampRef.current || (Date.now() - 2000)),
              }];
            });
          }
        }
      });

      session.on('error', (err) => {
        logger(`Session error: ${JSON.stringify(err)}`);
        stopSession();
      });

      const tokenUrl = `/api/session/token?apiKey=${encodeURIComponent(apiKey)}`;
      const tokenRes = await fetch(tokenUrl, { method: 'POST' });
      if (!tokenRes.ok) {
        const errText = await tokenRes.text();
        throw new Error(`Failed to get session token: ${errText}`);
      }
      const { client_secret } = await tokenRes.json();

      await session.connect({ apiKey: client_secret });

      applyMuteState();

      setIsConnected(true);
      setIsConnecting(false);

    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || 'Error starting session');
      stopSession();
    }
  };

  const stopSession = () => {
    // Handle streaming mode
    if (connectionMode === 'streaming') {
      stopStreamingSession();
      return;
    }
    
    setIsConnected(false);
    setIsConnecting(false);
    setIsModelTalking(false);
    micSuppressedRef.current = false;

    if (micDrainTimeoutRef.current) {
      clearTimeout(micDrainTimeoutRef.current);
      micDrainTimeoutRef.current = null;
    }

    if (sessionRef.current) {
      try { sessionRef.current.close(); } catch {}
      sessionRef.current = null;
    }

    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(track => track.stop());
      micStreamRef.current = null;
    }

    if (audioElRef.current) {
      audioElRef.current.srcObject = null;
      audioElRef.current = null;
    }
  };

  // ─── Streaming Mode Functions ──────────────────────────────────────────────

  const startStreamingSession = async () => {
    try {
      // Verify API key works by calling token endpoint
      const tokenUrl = `/api/session/token?apiKey=${encodeURIComponent(apiKey)}`;
      const tokenRes = await fetch(tokenUrl, { method: 'POST' });
      if (!tokenRes.ok) {
        throw new Error('Invalid API key');
      }
      
      setIsConnected(true);
      setIsConnecting(false);
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || 'Error starting streaming session');
      setIsConnecting(false);
    }
  };

  const stopStreamingSession = () => {
    setIsConnected(false);
    setIsRecording(false);
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }
    audioChunksRef.current = [];
  };

  const toggleRecording = async () => {
    if (isRecording) {
      // Stop recording
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
    } else {
      // Start recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: { echoCancellation: true, noiseSuppression: true } 
        });
        
        const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];
        const recordingStartTime = Date.now();
        
        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };
        
        mediaRecorder.onstop = async () => {
          const recordingDuration = (Date.now() - recordingStartTime) / 1000;
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          stream.getTracks().forEach(track => track.stop());
          
          // Skip if recording too short or too small (likely noise/accidental)
          if (recordingDuration < 1.0 || audioBlob.size < 1000) {
            console.log(`Recording skipped: ${recordingDuration.toFixed(1)}s, ${audioBlob.size} bytes`);
            return;
          }
          
          // Send to STT endpoint
          const formData = new FormData();
          formData.append('audio', audioBlob, 'recording.webm');
          
          try {
            const response = await fetch('/api/stt', { method: 'POST', body: formData });
            const data = await response.json();
            
            if (data.text && data.text.trim().length > 1) {
              // Send transcribed text as message
              await sendStreamingMessage(data.text);
            }
          } catch (err) {
            console.error('STT failed:', err);
            setErrorMsg('Transcription failed');
          }
        };
        
        mediaRecorder.start();
        setIsRecording(true);
      } catch (err) {
        console.error('Microphone access denied:', err);
        setErrorMsg('Microphone access denied');
      }
    }
  };

  const sendStreamingMessage = async (text: string) => {
    // Add user message
    const userMsg: Message = {
      role: 'user',
      text,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMsg]);
    setTextInput('');
    
    // Build messages array for API
    const apiMessages = messages
      .filter(m => m.role === 'user' || m.role === 'model')
      .map(m => ({
        role: m.role === 'model' ? 'assistant' : 'user',
        content: m.text || ''
      }));
    apiMessages.push({ role: 'user', content: text });
    
    // Create assistant message placeholder
    const assistantMsgId = crypto.randomUUID();
    const assistantMsg: Message = {
      role: 'model',
      text: '',
      id: assistantMsgId,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, assistantMsg]);
    
    let accumulatedText = '';
    let hasToolCalls = false;
    
    try {
      // Determine base URL for OpenRouter models
      const isOpenRouter = selectedModel.includes('/');
      const baseUrl = isOpenRouter ? 'https://openrouter.ai/api/v1' : undefined;
      
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          messages: apiMessages,
          model: selectedModel,
          base_url: baseUrl
        })
      });
      
      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader');
      
      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') break;
            
            try {
              const event = JSON.parse(data);
              
              if (event.type === 'text_delta') {
                accumulatedText += event.content;
                setMessages(prev => prev.map(m => 
                  m.id === assistantMsgId 
                    ? { ...m, text: (m.text || '') + event.content }
                    : m
                ));
              } else if (event.type === 'tool_call') {
                hasToolCalls = true;
                const toolMsg: Message = {
                  role: 'tool',
                  toolName: event.name,
                  toolArgs: event.arguments,
                  toolStatus: 'executing',
                  timestamp: new Date()
                };
                setMessages(prev => [...prev, toolMsg]);
              } else if (event.type === 'tool_result') {
                setMessages(prev => {
                  // Find the last tool message that's still executing
                  let lastToolIdx = -1;
                  for (let i = prev.length - 1; i >= 0; i--) {
                    if (prev[i].role === 'tool' && prev[i].toolName && prev[i].toolStatus === 'executing') {
                      lastToolIdx = i;
                      break;
                    }
                  }
                  if (lastToolIdx !== -1) {
                    const updated = [...prev];
                    updated[lastToolIdx] = {
                      ...updated[lastToolIdx],
                      toolStatus: 'completed',
                      toolResult: JSON.stringify(event.result)
                    };
                    return updated;
                  }
                  return prev;
                });
              } else if (event.type === 'error') {
                setErrorMsg(event.message);
              }
            } catch {}
          }
        }
      }
      
      // Play TTS for the response if there's text and no tool calls
      if (accumulatedText && !hasToolCalls) {
        try {
          const ttsResponse = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: accumulatedText })
          });
          
          if (ttsResponse.ok) {
            const audioBlob = await ttsResponse.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            audio.onended = () => URL.revokeObjectURL(audioUrl);
            await audio.play();
          }
        } catch (ttsErr) {
          console.error('TTS playback failed:', ttsErr);
        }
      }
    } catch (err) {
      console.error('Streaming failed:', err);
      setErrorMsg('Streaming failed');
    }
  };

  const flushBufferedText = () => {
    if (bufferedTextRef.current) {
      const { id, text } = bufferedTextRef.current;
      setMessages(prev => {
        // Don't add if already exists
        if (prev.some(m => m.id === id)) return prev;
        return [...prev, { role: 'model', text, id, timestamp: new Date() }];
      });
      bufferedTextRef.current = null;
    }
  };

  const handleToolUpdate = (msg: any) => {
    const { name, args, status, result } = msg;
    logger(`handleToolUpdate: ${name} status=${status}`);

    // Track pending tool calls
    if (status === 'executing') {
      // First tool call in a sequence - remove any recent model messages (initial acknowledgments)
      if (pendingToolCallsRef.current === 0) {
        setMessages(prev => {
          // Find and remove model messages added in the last 2 seconds
          const now = Date.now();
          const cutoff = now - 2000;
          return prev.filter(m => {
            if (m.role === 'model') {
              const msgTime = m.timestamp instanceof Date ? m.timestamp.getTime() : new Date(m.timestamp).getTime();
              if (msgTime > cutoff) {
                logger(`Removing early model message: "${m.text?.substring(0, 50)}..."`);
                return false;
              }
            }
            return true;
          });
        });
      }
      pendingToolCallsRef.current++;
      // Clear any pending flush timeout
      if (flushTimeoutRef.current) {
        clearTimeout(flushTimeoutRef.current);
        flushTimeoutRef.current = null;
      }
    } else if (status === 'completed' || status === 'failed') {
      pendingToolCallsRef.current = Math.max(0, pendingToolCallsRef.current - 1);
      // If all tool calls are done, flush buffered text after a short delay
      // (to catch any final text deltas that arrive right after tool completion)
      if (pendingToolCallsRef.current === 0) {
        flushTimeoutRef.current = setTimeout(() => {
          flushBufferedText();
          flushTimeoutRef.current = null;
        }, 100);
      }
    }

    setMessages(prev => {
      const existingIdx = prev.findIndex(m =>
        m.role === 'tool' &&
        m.toolName === name &&
        m.toolStatus === 'executing' &&
        JSON.stringify(m.toolArgs) === JSON.stringify(args)
      );

      if (existingIdx !== -1) {
        const updated = [...prev];
        updated[existingIdx] = {
          role: 'tool', toolName: name, toolArgs: args,
          toolStatus: status, toolResult: result, timestamp: new Date(),
        };
        return updated;
      }
      return [...prev, {
        role: 'tool', toolName: name, toolArgs: args,
        toolStatus: status, toolResult: result, timestamp: new Date(),
      }];
    });
  };

  const sendTextMessage = async () => {
    if (!textInput.trim()) return;
    
    // Streaming and noaudio modes - text chat works without connection
    if (connectionMode === 'streaming' || connectionMode === 'noaudio') {
      await sendStreamingMessage(textInput);
      return;
    }
    
    // WebRTC mode
    if (!sessionRef.current) return;
    sessionRef.current.sendMessage(textInput);

    setMessages(prev => [...prev, { role: 'user', text: textInput, timestamp: new Date() }]);
    setTextInput('');
  };

  const logger = (msg: string) => {
    console.log(`[Portal] ${msg}`);
  };

  const activeJobs = jobs.filter(j => j.status === 'pending' || j.status === 'running');
  const completedJobs = jobs.filter(j => j.status === 'completed');
  const failedJobs = jobs.filter(j => j.status === 'failed');

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      pending: 'text-amber-400 bg-amber-950/30 border-amber-800/40',
      queued: 'text-amber-400 bg-amber-950/30 border-amber-800/40',
      running: 'text-blue-400 bg-blue-950/30 border-blue-800/40',
      completed: 'text-green-400 bg-green-950/30 border-green-800/40',
      failed: 'text-red-400 bg-red-950/30 border-red-800/40',
    };
    return colors[status] || colors.pending;
  };

  const getToolIcon = (toolName: string) => {
    if (toolName === 'spawn_agent') return '🚀';
    if (toolName === 'list_directory' || toolName === 'read_file') return '📄';
    if (toolName === 'write_file' || toolName === 'edit_file') return '✏️';
    if (toolName === 'execute_command') return '⚡';
    if (toolName === 'create_directory') return '📁';
    if (toolName === 'delete_file') return '🗑️';
    if (toolName === 'move_file') return '📦';
    if (toolName === 'get_agent_status') return '📋';
    if (toolName === 'list_agents') return '📋';
    return '🔧';
  };

  const parseAgentResult = (toolName: string, result: string) => {
    try {
      const data = JSON.parse(result);
      if (toolName === 'spawn_agent' && data.job_id) {
        return {
          type: 'spawn' as const,
          jobId: data.job_id,
          mode: data.mode || 'standard',
          message: data.message || '',
        };
      }
      if (toolName === 'get_agent_status' && data.job_id) {
        const startLog = data.logs?.find((l: any) => l.type === 'start');
        const summaryLog = data.logs?.find((l: any) => l.type === 'summary');
        return {
          type: 'status' as const,
          jobId: data.job_id,
          status: data.status,
          model: startLog?.model || 'unknown',
          mode: data.mode || startLog?.mode || 'standard',
          result: summaryLog?.text || data.result || '',
        };
      }
      if (toolName === 'list_agents' && data.summary) {
        return {
          type: 'list' as const,
          active: (data.active || []).map((j: any) => ({
            jobId: j.job_id,
            mode: j.mode || 'standard',
            description: j.description || '',
          })),
          completed: (data.completed || []).map((j: any) => ({
            jobId: j.job_id,
            mode: j.mode || 'standard',
            description: j.description || '',
          })),
          failed: (data.failed || []).map((j: any) => ({
            jobId: j.job_id,
            mode: j.mode || 'standard',
            description: j.description || '',
          })),
          summary: data.summary,
        };
      }
    } catch {}
    return null;
  };

  return (
    <div className="flex flex-col h-screen bg-gray-950 font-mono">
      {/* Header bar */}
      <header className="border-b border-gray-800 bg-gray-900/60 backdrop-blur px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-4 h-4 rounded-full bg-violet-500 animate-pulse"></div>
          <h1 className="text-xl font-bold bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
            portal
          </h1>
          <span className="text-xs text-gray-400 bg-gray-800 px-2.5 py-0.5 rounded-md">v1.0.0-beta</span>
        </div>

        <div className="flex items-center gap-3">
          {!isConnected && (
            <input 
              type="password" 
              placeholder="Enter OpenAI API Key (or set in env)" 
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="text-xs bg-gray-950 border border-gray-800 focus:border-violet-500 text-white rounded px-3 py-2 w-64 focus:outline-none transition-colors"
            />
          )}
        </div>
      </header>

      {/* Main dashboard grid */}
      <main className="flex-1 flex overflow-hidden">
        {/* MIDDLE COLUMN: CONVERSATION LOG */}
        <section className="flex-1 flex flex-col bg-gray-900/5">
          <div className="px-6 py-4 border-b border-gray-800/50 bg-gray-900/20 flex items-center justify-between">
            <span className="text-sm font-bold text-gray-400 uppercase tracking-widest">Conversation</span>
            <span className="text-xs text-violet-400 font-semibold">{messages.length} messages</span>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {groupMessages(messages).map((grouped, idx) => {
              if (grouped.type === 'system' && grouped.message) {
                return (
                  <div key={idx} className="bg-violet-950/15 border border-violet-900/30 text-violet-300 p-4 rounded text-xs leading-relaxed max-w-2xl">
                    {grouped.message.text}
                  </div>
                );
              }

              if (grouped.type === 'tool' && grouped.message) {
                const m = grouped.message;
                const isExec = m.toolStatus === 'executing';
                const isDelegated = m.toolName === 'spawn_agent';
                const toolIcon = getToolIcon(m.toolName || '');
                const humanReadableArgs = m.toolName === 'execute_command' ? m.toolArgs?.command :
                  m.toolName === 'read_file' || m.toolName === 'write_file' || m.toolName === 'edit_file' ? m.toolArgs?.path :
                  m.toolName === 'list_directory' ? (m.toolArgs?.path || '.') :
                  m.toolName === 'move_file' ? (m.toolArgs?.source && m.toolArgs?.destination ? `${m.toolArgs.source} → ${m.toolArgs.destination}` : '') :
                  m.toolName === 'spawn_agent' ? m.toolArgs?.description :
                  m.toolName === 'get_agent_status' ? m.toolArgs?.job_id : '';
                
                // Parse execute_command results for nice display
                let cmdOutput = null;
                if (m.toolName === 'execute_command' && m.toolResult) {
                  try {
                    cmdOutput = JSON.parse(m.toolResult);
                  } catch {}
                }
                
                // For agent tools, render flat card without parent wrapper
                if (m.toolResult && (m.toolName === 'spawn_agent' || m.toolName === 'get_agent_status' || m.toolName === 'list_agents')) {
                  const agentData = parseAgentResult(m.toolName, m.toolResult);
                  const agentName = m.toolArgs?.description?.split(' ')[0] || agentData?.jobId?.slice(0, 8) || '';
                  
                  if (agentData?.type === 'spawn') {
                    return (
                      <div key={idx} className={`bg-gray-900 border border-gray-800/80 rounded p-3 text-xs font-mono max-w-3xl shadow-md ${
                        'border-l-4 border-l-violet-500'
                      }`}>
                        <div className="flex items-center gap-2 text-[11px]">
                          <span className="text-violet-400">🚀</span>
                          <span className="text-gray-300 font-medium">{agentName}</span>
                          <span className="text-gray-600">·</span>
                          <span className="text-gray-500">{agentData.mode}</span>
                          <span className="text-gray-600">·</span>
                          <span className="text-violet-400/80">{isExec ? 'Running' : 'Spawned'}</span>
                        </div>
                        {humanReadableArgs && (
                          <div className="text-gray-500 text-[10px] mt-1 line-clamp-1">
                            {humanReadableArgs}
                          </div>
                        )}
                      </div>
                    );
                  }
                  
                  if (agentData?.type === 'list') {
                    const { active, completed, failed, summary } = agentData;
                    return (
                      <div key={idx} className="bg-gray-900 border border-gray-800/80 rounded p-3 text-xs font-mono max-w-3xl shadow-md border-l-4 border-l-indigo-500">
                        <div className="flex items-center gap-2 text-[11px]">
                          <span className="text-gray-500">📋</span>
                          <span className="text-gray-400">Background Agents</span>
                          <span className="text-gray-600">·</span>
                          <span className="text-blue-400">{summary.active_count} active</span>
                          <span className="text-green-400">{summary.completed_count} done</span>
                          {summary.failed_count > 0 && (
                            <span className="text-red-400">{summary.failed_count} failed</span>
                          )}
                        </div>
                        {active.length > 0 && active.map((j: any) => (
                          <div key={j.jobId} className="flex items-center gap-2 text-[10px] py-0.5">
                            <span className="text-blue-400">●</span>
                            <span className="text-gray-300">{j.jobId}</span>
                            <span className="text-gray-600">{j.mode}</span>
                          </div>
                        ))}
                        {completed.length > 0 && completed.map((j: any) => (
                          <div key={j.jobId} className="flex items-center gap-2 text-[10px] py-0.5">
                            <span className="text-green-400">✓</span>
                            <span className="text-gray-300">{j.jobId}</span>
                            <span className="text-gray-600">{j.mode}</span>
                          </div>
                        ))}
                        {failed.length > 0 && failed.map((j: any) => (
                          <div key={j.jobId} className="flex items-center gap-2 text-[10px] py-0.5">
                            <span className="text-red-400">✗</span>
                            <span className="text-gray-300">{j.jobId}</span>
                            <span className="text-gray-600">{j.mode}</span>
                          </div>
                        ))}
                      </div>
                    );
                  }
                  
                  // Status card (get_agent_status fallback)
                  return (
                    <div key={idx} className="bg-gray-900 border border-gray-800/80 rounded p-3 text-xs font-mono max-w-3xl shadow-md border-l-4 border-l-indigo-500">
                      <div className="flex items-center gap-2 text-[11px]">
                        <span className="text-gray-500">📋</span>
                        <span className="text-gray-300 font-medium">{agentData?.jobId || m.toolArgs?.job_id}</span>
                        <span className="text-gray-600">·</span>
                        <span className="text-gray-500">{agentData?.mode}</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded ${getStatusColor(agentData?.status || 'unknown')}`}>
                          {agentData?.status}
                        </span>
                      </div>
                      {agentData?.model && (
                        <div className="text-gray-600 text-[10px] mt-1">
                          {agentData.model}
                        </div>
                      )}
                      {agentData?.result && (
                        <div className="text-green-400/70 text-[10px] mt-1 line-clamp-1">
                          {agentData.result}
                        </div>
                      )}
                    </div>
                  );
                }
                
                return (
                  <div key={idx} className={`bg-gray-900 border border-gray-800/80 rounded p-4 text-xs font-mono space-y-2.5 max-w-3xl shadow-md ${
                    isDelegated ? 'border-l-4 border-l-violet-500' : 'border-l-4 border-l-indigo-500'
                  }`}>
                    <div className={`flex items-center justify-between font-bold border-b border-gray-800/60 pb-1.5 ${
                      isDelegated ? 'text-violet-400' : 'text-indigo-400'
                    }`}>
                      <div className="flex items-center gap-2">
                        {isExec ? <Loader2 className="w-3.5 h-4 animate-spin" /> : <span>{toolIcon}</span>}
                        <span>{m.toolName}</span>
                      </div>
                      <span className="text-[10px] text-gray-500">
                        {isExec ? 'Executing...' : 'Completed'}
                      </span>
                    </div>

                    {humanReadableArgs && (
                      <div className="text-gray-300 font-semibold">
                        {humanReadableArgs}
                      </div>
                    )}

                    {cmdOutput ? (
                      <div className="bg-gray-950 rounded border border-gray-800/50 overflow-hidden">
                        {/* Command prompt */}
                        <div className="bg-gray-900 px-3 py-1.5 text-gray-400 text-[10px] border-b border-gray-800/50">
                          $ {m.toolArgs?.command}
                        </div>
                        {/* stdout */}
                        {cmdOutput.stdout && (
                          <div className="px-3 py-2 text-green-400/90 whitespace-pre-wrap max-h-36 overflow-y-auto">
                            {cmdOutput.stdout}
                          </div>
                        )}
                        {/* stderr */}
                        {cmdOutput.stderr && (
                          <div className="px-3 py-2 text-red-400/90 whitespace-pre-wrap max-h-36 overflow-y-auto border-t border-gray-800/50">
                            {cmdOutput.stderr}
                          </div>
                        )}
                        {/* exit code */}
                        {cmdOutput.exit_code !== 0 && (
                          <div className="px-3 py-1.5 text-amber-400/90 border-t border-gray-800/50">
                            exit code: {cmdOutput.exit_code}
                          </div>
                        )}
                      </div>
                    ) : m.toolResult ? (
                      <div className="bg-gray-950 p-2.5 rounded border border-gray-800/50 max-h-36 overflow-y-auto text-green-400/90 whitespace-pre-wrap">
                        {m.toolResult}
                      </div>
                    ) : null}
                  </div>
                );
              }

              if (grouped.type === 'user' && grouped.message) {
                return (
                  <div key={idx} className="flex flex-col gap-1 max-w-[80%] ml-auto items-end">
                    <span className="text-[10px] text-gray-500 font-bold px-1 uppercase">
                      You
                    </span>
                    <div className="p-4 rounded-lg text-sm leading-relaxed shadow-sm bg-violet-600 text-white rounded-tr-none">
                      {grouped.message.text}
                    </div>
                  </div>
                );
              }

              if (grouped.type === 'model' && grouped.combinedText) {
                return (
                  <div key={idx} className="flex flex-col gap-1 max-w-[80%] mr-auto items-start">
                    <span className="text-[10px] text-gray-500 font-bold px-1 uppercase">
                      Assistant
                    </span>
                    <div className="p-4 rounded-lg text-sm leading-relaxed shadow-sm bg-gray-900 border border-gray-800/80 text-gray-200 rounded-tl-none">
                      {grouped.combinedText}
                    </div>
                  </div>
                );
              }

              return null;
            })}
            <div ref={chatEndRef}></div>
          </div>

          {/* Chat Input */}
          <div className="p-4 border-t border-gray-800 bg-gray-950">
            {/* Error message */}
            {errorMsg && (
              <div className="mb-2 bg-red-950/30 border border-red-800/50 rounded-lg p-2 text-red-300 text-xs flex gap-2 items-center">
                <AlertCircle className="w-3 h-3 shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}
            
            {/* Main input row */}
            <div className="flex gap-2 items-center">
              {/* Mic/Connect button - hidden in noaudio mode */}
              {connectionMode !== 'noaudio' && (
                <button 
                  onClick={() => {
                    if (!isConnected) {
                      startSession();
                    } else if (connectionMode === 'webrtc') {
                      // Right-click or shift-click to disconnect in webrtc mode
                      stopSession();
                    } else {
                      toggleRecording();
                    }
                  }}
                  onContextMenu={(e) => {
                    e.preventDefault();
                    if (isConnected) stopSession();
                  }}
                  disabled={isConnecting}
                  title={!isConnected ? "Click to connect" : (connectionMode === 'webrtc' ? "Click to mute/unmute, right-click to disconnect" : "Click to record")}
                  className={`p-3 rounded-lg border transition-all flex items-center justify-center ${
                    isConnecting
                      ? 'bg-gray-800 border-gray-700 text-gray-400'
                      : !isConnected
                        ? 'bg-violet-600 hover:bg-violet-700 border-violet-500 text-white'
                        : isRecording
                          ? 'bg-red-600 border-red-500 text-white animate-pulse'
                          : isMuted
                            ? 'bg-red-950/50 border-red-800/50 text-red-400'
                            : 'bg-green-950/30 border-green-800/40 text-green-400'
                  }`}
                >
                  {isConnecting ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : !isConnected ? (
                    <Mic className="w-4 h-4" />
                  ) : isRecording || isMuted ? (
                    <MicOff className="w-4 h-4" />
                  ) : (
                    <Mic className="w-4 h-4" />
                  )}
                </button>
              )}
              
              {/* Disconnect button - shown when connected */}
              {isConnected && (
                <button 
                  onClick={stopSession}
                  className="p-3 rounded-lg border bg-red-950/40 hover:bg-red-950/80 border-red-800 text-red-300 transition-all flex items-center justify-center"
                  title="Disconnect"
                >
                  <Square className="w-4 h-4 fill-red-300" />
                </button>
              )}
              
              {/* Text input */}
              <input 
                type="text" 
                placeholder="Type a message and press enter..." 
                disabled={connectionMode !== 'streaming' && connectionMode !== 'noaudio' && !isConnected}
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendTextMessage()}
                className="flex-1 bg-gray-900/80 border border-gray-800 focus:border-violet-500 text-white rounded px-4 py-3 focus:outline-none transition-colors text-sm font-sans"
              />
              
              {/* Send button */}
              <button 
                onClick={sendTextMessage}
                disabled={connectionMode !== 'streaming' && connectionMode !== 'noaudio' && !isConnected}
                className="bg-violet-600 hover:bg-violet-700 disabled:bg-gray-800/50 p-3 rounded-lg text-white disabled:text-gray-600 transition-colors shadow-md"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
            
            {/* Controls row */}
            <div className="flex items-center gap-3 mt-2">
              {/* Mode selector - always first */}
              <select 
                value={connectionMode}
                onChange={(e) => setConnectionMode(e.target.value as 'webrtc' | 'streaming' | 'noaudio')}
                disabled={isConnected}
                className="text-[10px] px-2 py-1 rounded border bg-gray-800/50 border-gray-700 text-gray-400 cursor-pointer disabled:opacity-50 focus:outline-none focus:border-violet-500"
              >
                <option value="webrtc">🎙️ Realtime</option>
                <option value="streaming">💬 Streaming</option>
                <option value="noaudio">⌨️ Text Only</option>
              </select>
              
              {/* Model Selector - shown in streaming and noaudio modes */}
              {(connectionMode === 'streaming' || connectionMode === 'noaudio') && (
                <div className="relative" ref={modelSelectorRef}>
                  <button
                    onClick={() => {
                      setShowModelSelector(!showModelSelector);
                      if (!showModelSelector) setModelSearch('');
                    }}
                    className="text-[10px] px-2 py-1 rounded border bg-gray-800/50 border-gray-700 text-gray-400 hover:text-gray-300 hover:border-gray-600 transition-colors flex items-center gap-1.5"
                  >
                    <Cpu className="w-3 h-3" />
                    {availableModels.find(m => m.id === selectedModel)?.name || selectedModel}
                    <ChevronDown className="w-3 h-3" />
                  </button>
                  
                  {showModelSelector && (
                    <div className="absolute bottom-full left-0 mb-1 bg-gray-900 border border-gray-700 rounded-lg shadow-xl z-50 min-w-[300px] max-h-[450px] flex flex-col">
                      {/* Search Field */}
                      <div className="p-2 border-b border-gray-800">
                        <input
                          type="text"
                          placeholder="Search models..."
                          value={modelSearch}
                          onChange={(e) => setModelSearch(e.target.value)}
                          className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-violet-500"
                          autoFocus
                        />
                      </div>
                      
                      {/* Models List */}
                      <div className="overflow-y-auto flex-1 py-1">
                        {/* Favorites Section */}
                        {favoriteModels.length > 0 && !modelSearch && (
                          <>
                            <div className="px-3 py-1 text-[9px] text-gray-500 uppercase font-semibold border-b border-gray-800">
                              Favorites
                            </div>
                            {availableModels
                              .filter(m => favoriteModels.includes(m.id))
                              .map((model) => (
                                <div
                                  key={model.id}
                                  className={`w-full text-left px-3 py-2 text-xs hover:bg-gray-800 flex items-center justify-between ${
                                    selectedModel === model.id ? 'text-violet-400 bg-gray-800/50' : 'text-gray-300'
                                  }`}
                                >
                                  <button
                                    onClick={() => {
                                      setSelectedModel(model.id);
                                      setShowModelSelector(false);
                                      setModelSearch('');
                                    }}
                                    className="flex-1 text-left"
                                  >
                                    <div className="flex items-center gap-2">
                                      <Star className="w-3 h-3 text-yellow-500 fill-yellow-500" />
                                      <span>{model.name}</span>
                                    </div>
                                    <span className="text-[9px] text-gray-500 ml-5">{model.provider}</span>
                                  </button>
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      toggleFavoriteModel(model.id);
                                    }}
                                    className="p-1 hover:bg-gray-700 rounded"
                                  >
                                    <Star className="w-3 h-3 text-yellow-500 fill-yellow-500" />
                                  </button>
                                </div>
                              ))}
                            <div className="border-b border-gray-800 my-1"></div>
                          </>
                        )}
                        
                        {/* All Models */}
                        <div className="px-3 py-1 text-[9px] text-gray-500 uppercase font-semibold">
                          {modelSearch ? 'Search Results' : (favoriteModels.length > 0 ? 'All Models' : 'Models')}
                        </div>
                        {availableModels
                          .filter(m => {
                            if (!modelSearch) return true;
                            const search = modelSearch.toLowerCase();
                            return m.name.toLowerCase().includes(search) || 
                                   m.id.toLowerCase().includes(search) ||
                                   m.provider.toLowerCase().includes(search);
                          })
                          .map((model) => (
                            <div
                              key={model.id}
                              className={`w-full text-left px-3 py-2 text-xs hover:bg-gray-800 flex items-center justify-between ${
                                selectedModel === model.id ? 'text-violet-400 bg-gray-800/50' : 'text-gray-300'
                              }`}
                            >
                              <button
                                onClick={() => {
                                  setSelectedModel(model.id);
                                  setShowModelSelector(false);
                                  setModelSearch('');
                                }}
                                className="flex-1 text-left"
                              >
                                <span>{model.name}</span>
                                <span className="text-[9px] text-gray-500 ml-2">{model.provider}</span>
                                {model.pricing && (
                                  <span className="text-[9px] text-gray-600 ml-2">
                                    ${model.pricing.prompt.toFixed(6)}/tok
                                  </span>
                                )}
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  toggleFavoriteModel(model.id);
                                }}
                                className="p-1 hover:bg-gray-700 rounded"
                                title={favoriteModels.includes(model.id) ? "Remove from favorites" : "Add to favorites"}
                              >
                                <Star className={`w-3 h-3 ${favoriteModels.includes(model.id) ? 'text-yellow-500 fill-yellow-500' : 'text-gray-600'}`} />
                              </button>
                            </div>
                          ))}
                        {availableModels.filter(m => {
                          if (!modelSearch) return true;
                          const search = modelSearch.toLowerCase();
                          return m.name.toLowerCase().includes(search) || 
                                 m.id.toLowerCase().includes(search) ||
                                 m.provider.toLowerCase().includes(search);
                        }).length === 0 && (
                          <div className="px-3 py-4 text-xs text-gray-500 text-center">
                            No models found
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              {/* Status indicator */}
              <span className={`text-[10px] px-2 py-0.5 rounded-full ${
                isConnected 
                  ? (isModelTalking ? 'bg-violet-950/50 text-violet-300 border border-violet-800/50' 
                    : isRecording ? 'bg-red-950/50 text-red-300 border border-red-800/50'
                    : 'bg-green-950/30 text-green-400 border border-green-800/30')
                  : 'bg-gray-900 text-gray-500 border border-gray-800'
              }`}>
                {isConnected 
                  ? (isModelTalking ? 'Speaking' 
                    : isRecording ? 'Recording' 
                    : connectionMode === 'webrtc' ? 'Live' : 'Ready')
                  : 'Offline'}
              </span>
            </div>
          </div>
        </section>

        {/* RIGHT COLUMN: BACKGROUND AGENTS */}
        <section className="w-96 border-l border-gray-800/80 bg-gray-900/10 flex flex-col">
          <div className="px-6 py-4 border-b border-gray-800/50 bg-gray-900/20 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-indigo-400 animate-pulse" />
              <span className="text-sm font-bold text-gray-400 uppercase tracking-widest">Background Agents</span>
            </div>
            <div className="flex items-center gap-2">
              {activeJobs.length > 0 && (
                <span className="text-xs bg-blue-950/50 border border-blue-900/60 text-blue-300 px-2 py-0.5 rounded-full font-bold">
                  {activeJobs.length} active
                </span>
              )}
              {completedJobs.length > 0 && (
                <span className="text-xs bg-green-950/50 border border-green-900/60 text-green-300 px-2 py-0.5 rounded-full font-bold">
                  {completedJobs.length} done
                </span>
              )}
              {failedJobs.length > 0 && (
                <span className="text-xs bg-red-950/50 border border-red-900/60 text-red-300 px-2 py-0.5 rounded-full font-bold">
                  {failedJobs.length} failed
                </span>
              )}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {jobs.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-6 text-gray-600">
                <HelpCircle className="w-10 h-10 mb-2 stroke-1" />
                <p className="text-xs">No background agents.</p>
                <p className="text-[10px] max-w-[200px] mt-1 leading-relaxed">
                  Ask the assistant to spawn a background agent for complex tasks.
                </p>
              </div>
            ) : (
              <>
                {/* Active Jobs */}
                {activeJobs.length > 0 && (
                  <div>
                    <div className="text-[10px] font-bold text-gray-500 uppercase mb-2 px-1">Active</div>
                    {activeJobs.map((j) => {
                      const isSelected = selectedJobId === j.job_id;
                      const isAgentExpanded = expandedAgents.has(j.job_id);
                      const toolLogs = j.logs.filter((l: any) => l.type === 'tool');
                      const startLog = j.logs.find((l: any) => l.type === 'start');
                      return (
                        <div 
                          key={j.job_id}
                          className={`border rounded-lg p-4 cursor-pointer transition-all hover:bg-gray-900/40 shadow-sm mb-3 ${
                            isSelected ? 'bg-gray-900 border-indigo-500/50 shadow-md' : 'bg-gray-900/20 border-gray-800'
                          }`}
                          onClick={() => setSelectedJobId(isSelected ? null : j.job_id)}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-bold text-gray-300">{j.job_id}</span>
                            <div className="flex items-center gap-2">
                              <span className={`text-[10px] px-2.5 py-0.5 rounded-full font-bold uppercase border ${getStatusColor(j.status)}`}>
                                <span className="flex items-center gap-1">
                                  <Loader2 className="w-2.5 h-3 animate-spin" />
                                  running
                                </span>
                              </span>
                              <button
                                className="p-1 rounded hover:bg-red-900/50 text-gray-500 hover:text-red-400 transition-colors"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  cancelJob(j.job_id);
                                }}
                                title="Stop job"
                              >
                                <Square className="w-3 h-3" />
                              </button>
                            </div>
                          </div>

                          <p className="text-xs text-gray-400 line-clamp-2 leading-relaxed mb-2">
                            {j.description}
                          </p>

                          {isSelected && (
                            <div className="mt-3 pt-3 border-t border-gray-800/80">
                              <button 
                                className="flex items-center gap-1.5 text-[10px] font-bold text-gray-500 uppercase mb-2 hover:text-gray-400"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setExpandedAgents(prev => {
                                    const next = new Set(prev);
                                    if (next.has(j.job_id)) next.delete(j.job_id);
                                    else next.add(j.job_id);
                                    return next;
                                  });
                                }}
                              >
                                {isAgentExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                                Agent ({startLog?.model || 'model'})
                              </button>
                              {isAgentExpanded && (
                                <div className="bg-gray-950 border border-gray-800/80 p-2.5 rounded text-[11px] leading-relaxed max-h-48 overflow-y-auto space-y-1.5 text-gray-300 font-mono">
                                  {toolLogs.length === 0 ? (
                                    <div className="text-gray-600 italic">Initializing...</div>
                                  ) : (
                                    toolLogs.map((log: any, lIdx: number) => {
                                      const args = log.name === 'execute_command' ? log.args?.command : 
                                                   log.name === 'read_file' || log.name === 'write_file' ? log.args?.path :
                                                   log.name === 'list_directory' ? (log.args?.path || '.') : '';
                                      let output = '';
                                      if (log.name === 'execute_command' && log.result) {
                                        const cmdResult = typeof log.result === 'string' ? JSON.parse(log.result) : log.result;
                                        output = cmdResult.stdout?.trim() || (cmdResult.exit_code === 0 ? '✓' : `exit ${cmdResult.exit_code}`);
                                      } else if (log.name === 'list_directory' && log.result?.files) {
                                        output = log.result.files.map((f: any) => f.type === 'directory' ? `${f.name}/` : f.name).join(', ');
                                      } else if (log.name === 'read_file' && log.result?.content) {
                                        output = log.result.content.length > 60 ? log.result.content.slice(0, 60) + '...' : log.result.content;
                                      } else if (log.name === 'write_file' && log.result?.message) {
                                        output = '✓';
                                      }
                                      return (
                                        <div key={lIdx} className="flex flex-col">
                                          <span className="text-indigo-400">▸ {log.name} <span className="text-gray-500">{args}</span></span>
                                          {output && <span className="text-gray-500 ml-4">{output}</span>}
                                        </div>
                                      );
                                    })
                                  )}
                                </div>
                              )}
                            </div>
                          )}

                          <div className="flex justify-end mt-1">
                            {isSelected ? <ChevronDown className="w-4 h-4 text-gray-500" /> : <ChevronRight className="w-4 h-4 text-gray-500" />}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Completed Jobs */}
                {completedJobs.length > 0 && (
                  <div>
                    <div className="text-[10px] font-bold text-gray-500 uppercase mb-2 px-1">Completed</div>
                    {completedJobs.map((j) => {
                      const isSelected = selectedJobId === j.job_id;
                      const isAgentExpanded = expandedAgents.has(j.job_id);
                      const toolLogs = j.logs.filter((l: any) => l.type === 'tool');
                      const summaryLog = j.logs.find((l: any) => l.type === 'summary');
                      const startLog = j.logs.find((l: any) => l.type === 'start');
                      return (
                        <div 
                          key={j.job_id}
                          className={`border rounded-lg p-4 cursor-pointer transition-all hover:bg-gray-900/40 shadow-sm mb-3 ${
                            isSelected ? 'bg-gray-900 border-green-500/50 shadow-md' : 'bg-gray-900/20 border-gray-800'
                          }`}
                          onClick={() => setSelectedJobId(isSelected ? null : j.job_id)}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-bold text-gray-300">{j.job_id}</span>
                            <span className="text-[10px] px-2.5 py-0.5 rounded-full font-bold uppercase border text-green-400 bg-green-950/30 border-green-800/40">
                              completed
                            </span>
                          </div>

                          <p className="text-xs text-gray-400 line-clamp-2 leading-relaxed mb-2">
                            {j.description}
                          </p>

                          {isSelected && (
                            <div className="mt-3 pt-3 border-t border-gray-800/80">
                              <button 
                                className="flex items-center gap-1.5 text-[10px] font-bold text-gray-500 uppercase mb-2 hover:text-gray-400"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setExpandedAgents(prev => {
                                    const next = new Set(prev);
                                    if (next.has(j.job_id)) next.delete(j.job_id);
                                    else next.add(j.job_id);
                                    return next;
                                  });
                                }}
                              >
                                {isAgentExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                                Agent ({startLog?.model || 'model'})
                              </button>
                              {isAgentExpanded && (
                                <div className="bg-gray-950 border border-gray-800/80 p-2.5 rounded text-[11px] leading-relaxed max-h-48 overflow-y-auto space-y-1.5 text-gray-300 font-mono">
                                  {toolLogs.length === 0 ? (
                                    <div className="text-gray-600 italic">No tool logs</div>
                                  ) : (
                                    toolLogs.map((log: any, lIdx: number) => {
                                      const args = log.name === 'execute_command' ? log.args?.command : 
                                                   log.name === 'read_file' || log.name === 'write_file' ? log.args?.path :
                                                   log.name === 'list_directory' ? (log.args?.path || '.') : '';
                                      let output = '';
                                      if (log.name === 'execute_command' && log.result) {
                                        const cmdResult = typeof log.result === 'string' ? JSON.parse(log.result) : log.result;
                                        output = cmdResult.stdout?.trim() || (cmdResult.exit_code === 0 ? '✓' : `exit ${cmdResult.exit_code}`);
                                      } else if (log.name === 'list_directory' && log.result?.files) {
                                        output = log.result.files.map((f: any) => f.type === 'directory' ? `${f.name}/` : f.name).join(', ');
                                      } else if (log.name === 'read_file' && log.result?.content) {
                                        output = log.result.content.length > 60 ? log.result.content.slice(0, 60) + '...' : log.result.content;
                                      } else if (log.name === 'write_file' && log.result?.message) {
                                        output = '✓';
                                      }
                                      return (
                                        <div key={lIdx} className="flex flex-col">
                                          <span className="text-indigo-400">▸ {log.name} <span className="text-gray-500">{args}</span></span>
                                          {output && <span className="text-gray-500 ml-4">{output}</span>}
                                        </div>
                                      );
                                    })
                                  )}
                                </div>
                              )}
                              {summaryLog && (
                                <div className="mt-2 flex items-start gap-2 text-xs">
                                  <span className="text-green-400 mt-0.5">✓</span>
                                  <span className="text-gray-400">{summaryLog.text}</span>
                                </div>
                              )}
                            </div>
                          )}

                          <div className="flex justify-end mt-1">
                            {isSelected ? <ChevronDown className="w-4 h-4 text-gray-500" /> : <ChevronRight className="w-4 h-4 text-gray-500" />}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Failed Jobs */}
                {failedJobs.length > 0 && (
                  <div>
                    <div className="text-[10px] font-bold text-gray-500 uppercase mb-2 px-1">Failed</div>
                    {failedJobs.map((j) => {
                      const isSelected = selectedJobId === j.job_id;
                      const errorLog = j.logs.find((l: any) => l.type === 'error');
                      return (
                        <div 
                          key={j.job_id}
                          className={`border rounded-lg p-4 cursor-pointer transition-all hover:bg-gray-900/40 shadow-sm mb-3 ${
                            isSelected ? 'bg-gray-900 border-red-500/50 shadow-md' : 'bg-gray-900/20 border-gray-800'
                          }`}
                          onClick={() => setSelectedJobId(isSelected ? null : j.job_id)}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-bold text-gray-300">{j.job_id}</span>
                            <span className="text-[10px] px-2.5 py-0.5 rounded-full font-bold uppercase border text-red-400 bg-red-950/30 border-red-800/40">
                              failed
                            </span>
                          </div>

                          <p className="text-xs text-gray-400 line-clamp-2 leading-relaxed mb-2">
                            {j.description}
                          </p>

                          {isSelected && errorLog && (
                            <div className="mt-3 pt-3 border-t border-gray-800/80">
                              <div className="flex items-start gap-2 text-xs">
                                <span className="text-red-400 mt-0.5">✕</span>
                                <span className="text-red-400">{errorLog.text}</span>
                              </div>
                            </div>
                          )}

                          <div className="flex justify-end mt-1">
                            {isSelected ? <ChevronDown className="w-4 h-4 text-gray-500" /> : <ChevronRight className="w-4 h-4 text-gray-500" />}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
