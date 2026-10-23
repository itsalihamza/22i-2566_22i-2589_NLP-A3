const chatEl = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const micBtn = document.getElementById("micBtn");
const voiceRepliesBtn = document.getElementById("voiceRepliesBtn");
const statusEl = document.getElementById("status");
const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const newSessionBtn = document.getElementById("newSessionBtn");

let ws = null;
let sessionId = null;
let streaming = false;
let assistantBubble = null;
let awaitingFirstToken = false;
let hadSession = false; // true once we've had a session, so a reconnect is announced instead of wiping the chat

const PULSE_SVG = `<svg class="pulse-indicator" viewBox="0 0 64 20" xmlns="http://www.w3.org/2000/svg">
  <polyline class="pulse-line sweep" points="0,10 14,10 19,2 24,18 29,6 33,14 37,10 64,10" />
</svg>`;

const GREETING =
  "Hi! I'm the IronPulse Fitness membership assistant. Ask me about plans, classes, trainers or freezing a membership — I can also book classes for you. Type, or tap the mic and talk.";

const SUGGESTIONS = [
  "Which plan is best for me?",
  "What classes are on today?",
  "Book me into Spin on Friday",
];

// ---------- small storage helpers (storage can be unavailable: private mode etc.) ----------

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch {
    return null;
  }
  return value;
}

// A random ID that stays in this browser, so the CRM can recognise a returning member.
function userId() {
  let id = store("ironpulse.userId");
  if (!id) {
    id = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2));
    store("ironpulse.userId", id);
  }
  return id;
}

function timeNow() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/chat?user_id=${encodeURIComponent(userId())}`;
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addRow(role, text) {
  const row = document.createElement("div");
  row.className = `row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === "assistant" ? "IP" : "U";

  const col = document.createElement("div");
  col.className = "bubble-col";

  const bubble = document.createElement("div");
  bubble.className = `msg ${role}`;
  bubble.textContent = text;

  const ts = document.createElement("span");
  ts.className = "timestamp";
  ts.textContent = timeNow();

  col.appendChild(bubble);
  col.appendChild(ts);
  row.appendChild(avatar);
  row.appendChild(col);
  chatEl.appendChild(row);
  scrollToBottom();
  return bubble;
}

function addSystemMessage(text, extraClass = "") {
  const wrap = document.createElement("div");
  wrap.className = "system-row";
  const div = document.createElement("div");
  div.className = `msg system ${extraClass}`.trim();
  div.textContent = text;
  wrap.appendChild(div);
  chatEl.appendChild(wrap);
  scrollToBottom();
}

function addSuggestions() {
  const wrap = document.createElement("div");
  wrap.className = "suggestions";
  SUGGESTIONS.forEach((label) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = label;
    chip.addEventListener("click", () => sendMessage(label));
    wrap.appendChild(chip);
  });
  chatEl.appendChild(wrap);
  scrollToBottom();
}

function setConnectionState(state) {
  statusDot.className = `dot ${state}`;
  statusLabel.textContent = state;
}

function setStatus(text) {
  statusEl.textContent = text;
}

function readyStatus() {
  setStatus(`Ready · session ${sessionId ? sessionId.slice(0, 8) : "?"}…`);
}

// Single place that switches the composer between "Send" and "Stop".
function setStreaming(on) {
  streaming = on;
  sendBtn.textContent = on ? "Stop" : "Send";
  sendBtn.classList.toggle("stop", on);
}

// Ends the in-progress assistant bubble, whatever the reason (done, stopped, error, disconnect).
function finishAssistantBubble({ remove = false, note = "" } = {}) {
  if (assistantBubble) {
    if (remove) {
      assistantBubble.closest(".row")?.remove();
    } else {
      if (awaitingFirstToken) assistantBubble.textContent = ""; // drop the pulse animation
      assistantBubble.classList.remove("typing");
      if (note) assistantBubble.textContent += note;
    }
  }
  assistantBubble = null;
  awaitingFirstToken = false;
  setStreaming(false);
}

function startAssistantBubble() {
  muted = false; // a new answer: play its audio
  assistantBubble = addRow("assistant", "");
  assistantBubble.innerHTML = PULSE_SVG;
  awaitingFirstToken = true;
  setStreaming(true);
  setStatus("Assistant is typing…");
  return assistantBubble;
}

function startChat() {
  chatEl.innerHTML = "";
  greetingBubble = addRow("assistant", GREETING);
  addSuggestions();
}

// ======================= voice: speaking to the assistant =======================

// AudioWorklet: runs on the audio thread, resamples the microphone to 16 kHz, converts to
// 16-bit PCM and posts 100 ms chunks (1600 samples), which we send over the WebSocket.
const CAPTURE_WORKLET = `
class Capture extends AudioWorkletProcessor {
  constructor() { super(); this.ratio = sampleRate / 16000; this.t = 0; this.prev = 0; this.out = new Int16Array(1600); this.n = 0; }
  process(inputs) {
    const x = inputs[0][0];
    if (!x) return true;
    for (let i = 0; i < x.length; i++) {
      const cur = x[i];
      while (this.t <= 1) {
        const v = Math.max(-1, Math.min(1, this.prev + (cur - this.prev) * this.t));
        this.out[this.n++] = v < 0 ? v * 0x8000 : v * 0x7fff;
        if (this.n === this.out.length) {
          this.port.postMessage(this.out.buffer, [this.out.buffer]);
          this.out = new Int16Array(1600);
          this.n = 0;
        }
        this.t += this.ratio;
      }
      this.t -= 1;
      this.prev = cur;
    }
    return true;
  }
}
registerProcessor("capture", Capture);`;

let audioCtx = null; // one AudioContext for capture and playback (created on a user click)
let workletReady = false;
let micStream = null;
let micSource = null;
let micNode = null;
let listening = false; // server is receiving our audio
let draftBubble = null; // the user's words while they are still speaking
let greetingBubble = null;

async function ensureAudio() {
  if (!audioCtx) audioCtx = new AudioContext();
  if (audioCtx.state === "suspended") await audioCtx.resume();
  if (!workletReady) {
    const url = URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: "application/javascript" }));
    await audioCtx.audioWorklet.addModule(url);
    workletReady = true;
  }
  return audioCtx;
}

function setMicState(state) {
  micBtn.classList.toggle("listening", state === "listening" || state === "hearing");
  micBtn.classList.toggle("hearing", state === "hearing");
  micBtn.setAttribute("aria-label", state === "idle" ? "Speak" : "Stop listening");
}

async function startListening() {
  if (!ws || ws.readyState !== WebSocket.OPEN || !sessionId) {
    addSystemMessage("Not connected yet — please wait a moment and try again.", "error");
    return;
  }
  stopPlayback(); // barge-in: talking over the assistant stops its voice (and the server stops the reply)
  try {
    const ctx = await ensureAudio();
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    micSource = ctx.createMediaStreamSource(micStream);
    micNode = new AudioWorkletNode(ctx, "capture");
    micNode.port.onmessage = (e) => {
      if (listening && ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
    };
    micSource.connect(micNode);
  } catch (err) {
    stopCapture();
    addSystemMessage(`Microphone unavailable: ${err.message || err}`, "error");
    return;
  }
  if (streaming) finishAssistantBubble({ note: " [stopped]" });
  ws.send(JSON.stringify({ type: "audio_start", sample_rate: 16000 }));
  setMicState("listening");
  setStatus("Listening… speak now (it stops when you pause)");
}

function stopCapture() {
  listening = false;
  try {
    micSource?.disconnect();
    micNode?.disconnect();
  } catch {}
  micStream?.getTracks().forEach((t) => t.stop());
  micStream = micSource = micNode = null;
  setMicState("idle");
}

function showDraft(text) {
  if (!draftBubble) {
    draftBubble = addRow("user", "");
    draftBubble.classList.add("draft");
  }
  draftBubble.textContent = text || "…";
  scrollToBottom();
}

function clearDraft() {
  draftBubble?.closest(".row")?.remove();
  draftBubble = null;
}

micBtn.addEventListener("click", () => {
  if (micStream) {
    // second tap: the member finished speaking
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "audio_end" }));
    stopCapture();
    setStatus("Transcribing…");
  } else {
    startListening();
  }
});

// ======================= voice: hearing the assistant =======================

let playHead = 0; // when the next audio chunk should start, so chunks play back-to-back in order
let playing = [];
let muted = false; // after Stop / barge-in, ignore audio chunks that were already on their way

function playChunk(b64, sampleRate) {
  if (!audioCtx || muted) return;
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const pcm = new Int16Array(bytes.buffer, 0, bytes.length >> 1);
  const buffer = audioCtx.createBuffer(1, pcm.length, sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) data[i] = pcm[i] / 32768;
  const src = audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(audioCtx.destination);
  const start = Math.max(audioCtx.currentTime + 0.03, playHead);
  src.start(start);
  playHead = start + buffer.duration;
  playing.push(src);
  src.onended = () => (playing = playing.filter((s) => s !== src));
}

function stopPlayback() {
  muted = true;
  playing.forEach((s) => {
    try {
      s.stop();
    } catch {}
  });
  playing = [];
  playHead = 0;
}

let voiceReplies = store("ironpulse.voiceReplies") === "1";

function sendVoiceReplies() {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "tts", enabled: voiceReplies }));
}

function renderVoiceRepliesButton() {
  voiceRepliesBtn.setAttribute("aria-pressed", String(voiceReplies));
}

voiceRepliesBtn.addEventListener("click", async () => {
  voiceReplies = !voiceReplies;
  store("ironpulse.voiceReplies", voiceReplies ? "1" : "0");
  renderVoiceRepliesButton();
  if (voiceReplies) await ensureAudio(); // audio playback needs a user click to start
  else stopPlayback();
  sendVoiceReplies();
});

function fmt(ms) {
  if (ms == null) return null;
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

// Latency report under the answer (sent by the server after every spoken turn).
function showMetrics(m) {
  const rows = chatEl.querySelectorAll(".row.assistant");
  const col = rows[rows.length - 1]?.querySelector(".bubble-col");
  if (!col) return;
  const parts = [];
  if (m.asr_ms != null) parts.push(`ASR ${fmt(m.asr_ms)}`);
  parts.push(`retrieval ${fmt(m.retrieval_ms)}`);
  if (m.tools && m.tools.length) parts.push(`tools ${fmt(m.tool_ms)} (${m.tools.map((t) => t.name + (t.ok ? "" : " ✗")).join(", ")})`);
  if (m.llm_first_token_ms != null) parts.push(`LLM first token ${fmt(m.llm_first_token_ms)}`);
  if (m.tts_first_audio_ms != null) parts.push(`TTS ${fmt(m.tts_first_audio_ms)}`);
  const after = m.first_audio_from === "end_of_speech" ? "you stopped speaking" : "sending";
  if (m.acknowledgement && m.first_audio_ms != null) {
    parts.push(`"${m.acknowledgement}" after ${fmt(m.first_audio_ms)}`);
    if (m.first_answer_audio_ms != null) parts.push(`answer audio ${fmt(m.first_answer_audio_ms)} after ${after}`);
  } else if (m.first_audio_ms != null) {
    parts.push(`first audio ${fmt(m.first_audio_ms)} after ${after}`);
  }
  const line = document.createElement("div");
  line.className = "metrics";
  line.textContent = parts.join(" · ");
  col.appendChild(line);
  scrollToBottom();
}

// ======================= connection =======================

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  setConnectionState("connecting");
  ws = new WebSocket(wsUrl());

  ws.onopen = () => {
    setStatus("Connected · starting session…");
    ws.send(JSON.stringify({ type: "new_session" }));
  };

  ws.onclose = () => {
    setConnectionState("offline");
    stopCapture();
    stopPlayback();
    clearDraft();
    if (streaming) finishAssistantBubble({ note: " [connection lost]" });
    sessionId = null;
    setStatus("Disconnected. Reconnecting in 2s…");
    setTimeout(connect, 2000);
  };

  ws.onerror = () => setStatus("WebSocket error");

  ws.onmessage = (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
      addSystemMessage("Invalid server message", "error");
      return;
    }

    switch (data.type) {
      case "session":
        sessionId = data.session_id;
        setConnectionState("online");
        if (hadSession && !chatEl.dataset.resetting) {
          // Reconnected after a drop: keep what the user can see, but be honest that the
          // server-side memory is gone.
          addSystemMessage("Reconnected with a new session — the assistant won't remember the messages above.");
        } else {
          startChat();
        }
        delete chatEl.dataset.resetting;
        hadSession = true;
        sendVoiceReplies();
        readyStatus();
        break;
      case "member": // returning member recognised by the CRM
        if (greetingBubble && data.name) {
          greetingBubble.textContent =
            `Welcome back, ${data.name}!` +
            (data.last_question ? ` Last time you asked: “${data.last_question}”.` : "") +
            " What can I help you with today?";
        }
        break;
      case "listening":
        listening = true;
        showDraft("");
        break;
      case "speech_start":
        setMicState("hearing");
        break;
      case "transcript":
        if (!data.final) {
          showDraft(data.text);
          break;
        }
        stopCapture();
        clearDraft();
        if (data.text) {
          addRow("user", data.text);
          startAssistantBubble();
        }
        break;
      case "speech_end":
        stopCapture();
        setStatus("Transcribing…");
        break;
      case "token":
        if (!assistantBubble) break;
        if (awaitingFirstToken) {
          assistantBubble.textContent = "";
          assistantBubble.classList.add("typing");
          awaitingFirstToken = false;
        }
        assistantBubble.textContent += data.content;
        scrollToBottom();
        break;
      case "audio":
        playChunk(data.data, data.sample_rate);
        break;
      case "audio_done":
        break;
      case "metrics":
        showMetrics(data);
        break;
      case "done":
        finishAssistantBubble({ note: data.stopped ? " [stopped]" : "" });
        readyStatus();
        break;
      case "error":
        if (streaming) finishAssistantBubble({ remove: true });
        if (["no_speech", "voice_unavailable", "asr_error"].includes(data.code)) {
          stopCapture();
          clearDraft();
        }
        addSystemMessage(data.message || "Error", "error");
        setStatus("Error — you can try again");
        break;
      default:
        break;
    }
  };
}

function newSession() {
  chatEl.dataset.resetting = "1";
  stopCapture();
  stopPlayback();
  clearDraft();
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    connect();
    return;
  }
  if (streaming) finishAssistantBubble({ note: " [stopped]" });
  ws.send(JSON.stringify({ type: "new_session" }));
}

function stopReply() {
  stopPlayback();
  if (streaming && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop" }));
    setStatus("Stopping…");
  }
}

// Returns true if the message was sent, so the caller only clears the input on success.
function sendMessage(text) {
  text = text.trim();
  if (!text || streaming) return false;
  if (!ws || ws.readyState !== WebSocket.OPEN || !sessionId) {
    addSystemMessage("Not connected yet — please wait a moment and try again.", "error");
    return false;
  }
  if (micStream) stopCapture(); // typing cancels listening
  clearDraft();
  stopPlayback();

  addRow("user", text);
  startAssistantBubble();
  ws.send(JSON.stringify({ type: "message", session_id: sessionId, content: text }));
  return true;
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (streaming) {
    stopReply();
    return;
  }
  if (sendMessage(input.value)) {
    input.value = "";
    input.style.height = "auto";
  }
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!streaming) form.requestSubmit(); // Enter never acts as "Stop"
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

newSessionBtn.addEventListener("click", newSession);

renderVoiceRepliesButton();
connect();
