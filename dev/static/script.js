// --- START OF FILE script.js ---

// --- Configuration ---
const pageProtocol = window.location.protocol;
const wsProtocol = pageProtocol === 'https:' ? 'wss:' : 'ws:';
const WEBSOCKET_URL = `${wsProtocol}//${window.location.host}/ws`;
const TARGET_SAMPLE_RATE = 16000; // Mic input rate
// const TTS_EXPECTED_SAMPLE_RATE = 22050; // REMOVED - Will be dynamic
const AUDIO_FRAME_LENGTH = 512;
const AUDIO_PROCESSOR_URL = '/static/audio-processor.js'; // Make sure this path is correct
const ACTIVATION_SOUND_CLIENT_URL = '/static/sounds/activation.wav'; // Path to client-side activation sound

// --- State ---
let websocket = null;
let audioContext = null;
let audioWorkletNode = null;
let microphoneStream = null;
let isRecording = false;
let currentTTSChunks = [];
let isAudioWorkletReady = false;
let currentTTSSampleRate = 22050; // Default, will be updated by server via tts_info
let activationSoundBuffer = null; // To store decoded activation sound

// --- DOM Elements ---
const statusDiv = document.getElementById('status');
const transcriptDiv = document.getElementById('transcript');
const errorDiv = document.getElementById('error');
// Assuming you have an audio element for client-side activation sound in your HTML:
// <audio id="activationSoundClient" src="/static/sounds/activation_client.wav" preload="auto" style="display:none;"></audio>
// Or we can load it via fetch and decodeAudioData for more control. Let's do the latter.

// --- Helper Function ---
function concatenateArrayBuffers(buffers) {
    let totalLength = 0;
    for (const buffer of buffers) {
        totalLength += buffer.byteLength;
    }
    const result = new Uint8Array(totalLength);
    let offset = 0;
    for (const buffer of buffers) {
        result.set(new Uint8Array(buffer), offset);
        offset += buffer.byteLength;
    }
    return result.buffer;
}

function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    try {
        return btoa(binary);
    } catch (e) {
        console.error("btoa failed:", e);
        return null;
    }
}

// --- Audio Initialization and Activation Sound Loading ---
async function loadActivationSound() {
    if (!audioContext) {
        console.warn("AudioContext not ready, cannot load activation sound yet.");
        return;
    }
    try {
        const response = await fetch(ACTIVATION_SOUND_CLIENT_URL);
        if (!response.ok) {
            throw new Error(`Failed to fetch activation sound: ${response.statusText}`);
        }
        const arrayBuffer = await response.arrayBuffer();
        activationSoundBuffer = await audioContext.decodeAudioData(arrayBuffer);
        console.log("Client-side activation sound loaded and decoded.");
    } catch (e) {
        console.error("Error loading client-side activation sound:", e);
        showError("Could not load activation sound.");
    }
}

async function playClientActivationSound() {
    if (audioContext && activationSoundBuffer) {
        try {
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            const source = audioContext.createBufferSource();
            source.buffer = activationSoundBuffer;
            source.connect(audioContext.destination);
            source.start();
            console.log("Playing client-side activation sound.");
        } catch (e) {
            console.error("Error playing client-side activation sound:", e);
        }
    } else {
        console.warn("Cannot play activation sound: AudioContext or sound buffer not ready.");
    }
}


// --- Audio Processing ---
async function createAudioContextAndWorklet() {
    if (audioContext && isAudioWorkletReady) {
        return true;
    }
    if (!audioContext) {
        try {
            window.AudioContext = window.AudioContext || window.webkitAudioContext;
            audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
            console.log(`AudioContext created with sample rate: ${audioContext.sampleRate}`);
            if (audioContext.sampleRate !== TARGET_SAMPLE_RATE) {
                console.warn(`Requested ${TARGET_SAMPLE_RATE}Hz but got ${audioContext.sampleRate}Hz. Input stream constraints are crucial.`);
            }
            await loadActivationSound(); // Load activation sound after context is ready
        } catch (e) {
            console.error("Error creating AudioContext:", e);
            showError("Could not initialize audio processing. Please use a modern browser.");
            return false;
        }
    }

    if (!isAudioWorkletReady) {
        try {
            console.log(`Loading AudioWorklet module from: ${AUDIO_PROCESSOR_URL}`);
            await audioContext.audioWorklet.addModule(AUDIO_PROCESSOR_URL);
            console.log("AudioWorklet module loaded successfully.");
            isAudioWorkletReady = true;
        } catch (e) {
            console.error("Error loading AudioWorklet module:", e);
            showError(`Could not load audio processor: ${e.message}. Check the console and file path.`);
            return false;
        }
    }
    return true;
}


async function startAudioProcessing() {
    if (isRecording) return;
    if (!await createAudioContextAndWorklet()) return;
    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
        console.warn("WebSocket not ready, cannot start audio processing.");
        return;
    }

    try {
        microphoneStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: TARGET_SAMPLE_RATE,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
        console.log("Microphone access granted.");

        const source = audioContext.createMediaStreamSource(microphoneStream);
        audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-recorder-processor');
        console.log("AudioWorkletNode created.");

        audioWorkletNode.port.onmessage = (event) => {
            if (!isRecording || !websocket || websocket.readyState !== WebSocket.OPEN) return;
            if (event.data.type === 'audio_data') {
                const pcm16Buffer = event.data.buffer;
                const base64String = arrayBufferToBase64(pcm16Buffer);
                if (base64String) {
                    try {
                        websocket.send(JSON.stringify({
                            type: "audio_chunk",
                            data: base64String
                        }));
                    } catch (err) {
                        console.error("Error sending audio chunk:", err);
                        if (websocket.readyState !== WebSocket.OPEN) {
                            console.warn("WebSocket closed while trying to send audio.");
                            stopAudioProcessing();
                        }
                    }
                } else {
                    console.error("Failed to encode audio buffer to Base64.");
                }
            } else {
                console.warn("Received unknown message type from AudioWorklet:", event.data.type);
            }
        };

        source.connect(audioWorkletNode);
        audioWorkletNode.connect(audioContext.destination);

        if (audioContext.state === 'suspended') {
            await audioContext.resume();
            console.log('AudioContext resumed ->', audioContext.state);
        }

        audioWorkletNode.port.postMessage({ command: 'start' });
        isRecording = true;
        console.log("Audio processing started using AudioWorklet.");

    } catch (err) {
        console.error("Error starting audio processing:", err);
        showError(`Could not access microphone or start processing: ${err.message}. Please grant permission.`);
        stopAudioProcessing();
    }
}


function stopAudioProcessing() {
    if (!isRecording && !audioWorkletNode) {
        console.log("Audio processing already stopped or not started.");
        return;
    }
    console.log("Stopping audio processing...");
    isRecording = false;

    if (audioWorkletNode) {
        audioWorkletNode.port.postMessage({ command: 'stop' });
        audioWorkletNode.port.onmessage = null;
        try {
             audioWorkletNode.disconnect();
             console.log("AudioWorkletNode disconnected.");
        } catch (e) {
            console.warn("Error disconnecting AudioWorkletNode:", e);
        }
        audioWorkletNode = null;
    }

    if (microphoneStream) {
        microphoneStream.getTracks().forEach(track => track.stop());
        console.log("Microphone stream stopped.");
        microphoneStream = null;
    }
    console.log("Audio processing stopped.");
}


// --- WebSocket Communication ---
function connectWebSocket() {
    clearError();
    updateStatus("Connecting...", "status-connecting");
    transcriptDiv.textContent = '';

    if (websocket && websocket.readyState !== WebSocket.CLOSED) {
        console.warn("Closing existing WebSocket connection before reconnecting.");
        websocket.onclose = null; // Prevent old onclose from firing
        websocket.close(1000, "Client initiated reconnect");
    }
    websocket = null;

    console.log(`Attempting to connect WebSocket to: ${WEBSOCKET_URL}`);
    websocket = new WebSocket(WEBSOCKET_URL);

    websocket.onopen = (event) => {
        console.log("WebSocket connected.");
        createAudioContextAndWorklet().then(ready => {
            if (ready) {
                startAudioProcessing();
            } else {
                showError("Failed to initialize audio system after WebSocket connection.");
            }
        });
    };

    websocket.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            // console.log("WebSocket message received:", message.type, message.code || '');

            switch (message.type) {
                case "status":
                    updateStatus(message.message, `status-${message.code.split('_')[0]}`);
                    if (message.code === "play_activation_sound_cue") {
                        playClientActivationSound(); // Play client-side sound
                    }
                    if ((message.code === "wakeword_listening" || message.code === "listening_started") && !isRecording) {
                       console.log("Server expects listening, ensuring audio processing is active.");
                       createAudioContextAndWorklet().then(ready => {
                           if (ready) { startAudioProcessing(); }
                           else { showError("Failed to initialize audio system when server requested listening."); }
                       });
                    }
                    break;
                case "transcript":
                    updateTranscript(message.text, message.is_final);
                    break;
                case "tts_info": // New: Handle TTS info from server
                    console.log("Received TTS info:", message);
                    if (message.sample_rate) {
                        currentTTSSampleRate = message.sample_rate;
                        console.log(`TTS sample rate set to: ${currentTTSSampleRate} Hz`);
                    }
                    // Could also handle channels, bit_depth if server sends them
                    break;
                case "tts_chunk":
                    handleTTSChunk(message.data);
                    break;
                case "tts_finished":
                    console.log("Received TTS finished signal.");
                    processBufferedTTS(currentTTSSampleRate); // Pass the current rate
                    // Optionally reset currentTTSSampleRate to a default if desired,
                    // or rely on next tts_info message.
                    // currentTTSSampleRate = 22050; // Example reset
                    break;
                case "error":
                    showError(`Server error: ${message.message}`);
                    break;
                default:
                    console.warn("Unknown message type:", message.type);
            }
        } catch (e) {
            console.error("Error parsing WebSocket message:", e);
            showError("Received invalid message from server.");
        }
    };

    websocket.onerror = (event) => {
        console.error("WebSocket error:", event);
        showError("WebSocket connection error. Check console. Trying to reconnect...");
        stopAudioProcessing(); // Stop audio processing on error
        if (websocket) {
            websocket.onclose = null; // Avoid onclose firing after manual close
            websocket.close();
        }
         // No automatic reconnect here, onclose will handle it if it was unclean
    };

    websocket.onclose = (event) => {
        console.log(`WebSocket closed: Code=${event.code}, Reason='${event.reason}', WasClean=${event.wasClean}`);
        stopAudioProcessing();
        websocket = null;
        if (event.reason === "Client initiated reconnect") {
             updateStatus("Reconnecting...", "status-connecting");
             // The connectWebSocket() call will handle the actual reconnection attempt
        } else if (!event.wasClean && event.code !== 1000 /* Normal Closure */ && event.code !== 1001 /* Going Away */) {
            showError(`WebSocket closed unexpectedly (Code: ${event.code}). Trying to reconnect...`);
            setTimeout(connectWebSocket, 5000); // Retry connection
        } else {
            updateStatus("Disconnected", "status-error");
        }
    };
}


// --- UI Updates ---
function updateStatus(text, cssClass = '') {
    statusDiv.textContent = text;
    statusDiv.className = `status ${cssClass}`;
}

function updateTranscript(text, isFinal) {
    if (isFinal) {
        transcriptDiv.textContent = `You: ${text}`;
    } else {
        transcriptDiv.textContent = `You: ${text}...`;
    }
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    console.error("UI Error:", message);
}

function clearError() {
    errorDiv.textContent = '';
    errorDiv.style.display = 'none';
}


// --- TTS Playback ---
function base64ToArrayBuffer(base64) {
    try {
        const binaryString = atob(base64);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    } catch (e) {
        console.error("Base64 decoding error:", e);
        showError("Error decoding audio data from server.");
        return null;
    }
}

function handleTTSChunk(base64Data) {
    const arrayBuffer = base64ToArrayBuffer(base64Data);
    if (arrayBuffer && arrayBuffer.byteLength > 0) {
        currentTTSChunks.push(arrayBuffer);
    } else if (arrayBuffer && arrayBuffer.byteLength === 0) {
         console.log("Received empty TTS chunk, ignoring.");
    } else {
         console.error("Failed to decode TTS chunk, skipping.");
    }
}

// Modified to accept the actual sample rate
function processBufferedTTS(actualSampleRate) {
    if (currentTTSChunks.length > 0) {
        const completeAudioData = concatenateArrayBuffers(currentTTSChunks);
        console.log(`Reassembled TTS audio data: ${completeAudioData.byteLength} bytes`);
        playCompleteTTSAudio(completeAudioData, actualSampleRate); // Pass the sample rate
    } else {
        console.warn("TTS finished signal received, but no valid chunks were buffered.");
    }
    currentTTSChunks = [];
}

// Modified to use the passed sourceSampleRate
async function playCompleteTTSAudio(completeAudioData, sourceSampleRate) {
    if (!await createAudioContextAndWorklet()) { // Ensures AudioContext is ready
        console.error("Cannot play TTS, AudioContext not available.");
        showError("Audio playback failed.");
        return;
    }
    if (!completeAudioData || completeAudioData.byteLength === 0) {
        console.warn("Attempted to play empty or invalid audio data.");
        return;
    }
    if (!sourceSampleRate) {
        console.error("Cannot play TTS: Sample rate not provided or invalid.");
        showError("Audio playback failed: missing sample rate information.");
        return;
    }

    try {
        console.log(`Processing complete raw TTS audio data (${completeAudioData.byteLength} bytes) at ${sourceSampleRate} Hz...`);
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        const numberOfChannels = 1; // Assuming mono output
        const numberOfSamples = completeAudioData.byteLength / 2; // 16-bit = 2 bytes/sample

        if (completeAudioData.byteLength % 2 !== 0) {
            console.warn("Raw audio data has odd byte length, potential data corruption.");
        }

        console.log(`Creating AudioBuffer: ${numberOfChannels}ch, ${numberOfSamples} samples, ${sourceSampleRate}Hz`);
        // The AudioBuffer is created with the *source* sample rate.
        // The browser handles resampling to audioContext.destination.sampleRate during playback.
        const audioBuffer = audioContext.createBuffer(numberOfChannels, numberOfSamples, sourceSampleRate);
        const channelData = audioBuffer.getChannelData(0);
        const pcmData = new Int16Array(completeAudioData); // CORRECTED TYPO HERE

        for (let i = 0; i < numberOfSamples; i++) {
            if (i < pcmData.length) {
                 channelData[i] = pcmData[i] / 32768.0;
            }
        }
        console.log(`Successfully created AudioBuffer from raw PCM. Duration: ${audioBuffer.duration.toFixed(2)}s`);

        const sourceNode = audioContext.createBufferSource(); // Renamed to sourceNode to avoid conflict
        sourceNode.buffer = audioBuffer;
        sourceNode.connect(audioContext.destination);
        sourceNode.onended = () => {
            console.log("TTS playback finished.");
        };
        sourceNode.start();
        console.log("Starting TTS playback...");

    } catch (e) {
        console.error("Error processing or playing raw TTS audio:", e);
        showError(`Error playing assistant response: ${e.message}`);
        const firstBytes = new Uint8Array(completeAudioData.slice(0, 12));
        console.error("First 12 bytes of failed audio data:", firstBytes);
    }
}


// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    clearError();
    let supported = true;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError("Your browser does not support microphone access (getUserMedia).");
        supported = false;
    }
     if (!window.WebSocket) {
        showError("Your browser does not support WebSockets.");
        supported = false;
    }
     if (!window.AudioContext && !window.webkitAudioContext) {
         showError("Your browser does not support the Web Audio API.");
         supported = false;
     }
     if (!window.AudioWorkletNode) {
        showError("Your browser does not support AudioWorkletNode.");
        supported = false;
     }

    if (!supported) {
        updateStatus("Browser Not Supported", "status-error");
        return;
    }

    // Create audio context early if possible, also loads activation sound
    createAudioContextAndWorklet().then(ready => {
        if (ready) {
            connectWebSocket(); // Start connection after audio context is ready
        } else {
            showError("Failed to initialize audio system. Cannot connect.");
            updateStatus("Audio Init Failed", "status-error");
        }
    });
});