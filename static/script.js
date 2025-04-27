// --- Configuration ---
const pageProtocol = window.location.protocol; // Check if page is https: or http:
const wsProtocol = pageProtocol === 'https:' ? 'wss:' : 'ws:'; // Use wss: if page is https:, otherwise ws:
const WEBSOCKET_URL = `${wsProtocol}//${window.location.host}/ws/user-${Date.now()}`; // Construct URL dynamically
const TARGET_SAMPLE_RATE = 16000; // Rate expected by backend (Vosk, Porcupine, Cobra)
const AUDIO_FRAME_LENGTH = 512; // Samples per chunk sent to backend (matches Porcupine/Cobra)
const BUFFER_SIZE = 4096; // Audio processing buffer size (adjust if needed)

// --- State ---
let websocket = null;
let audioContext = null;
let audioProcessor = null;
let microphoneStream = null;
let isRecording = false;
let currentTTSChunks = []; // Buffer for incoming TTS chunks
let isBufferingTTS = false; // Flag to indicate if we are collecting TTS chunks

// --- DOM Elements ---
const statusDiv = document.getElementById('status');
const transcriptDiv = document.getElementById('transcript');
const errorDiv = document.getElementById('error');


// --- Helper Function ---
// Function to concatenate multiple ArrayBuffers
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
    return result.buffer; // Return as ArrayBuffer
}


// --- Audio Processing ---
function createAudioContext() {
    if (!audioContext) {
        try {
            window.AudioContext = window.AudioContext || window.webkitAudioContext;
            audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
            console.log(`AudioContext created with sample rate: ${audioContext.sampleRate}`);
            if (audioContext.sampleRate !== TARGET_SAMPLE_RATE) {
                console.warn(`Requested ${TARGET_SAMPLE_RATE}Hz but got ${audioContext.sampleRate}Hz. Resampling might occur if browser doesn't support target rate.`);
                // We'll rely on the browser or ScriptProcessorNode to handle this mismatch for now.
                // For perfect control, AudioWorklet with explicit resampling would be needed.
            }
        } catch (e) {
            console.error("Error creating AudioContext:", e);
            showError("Could not initialize audio processing. Please use a modern browser.");
            return false;
        }
    }
    return true;
}


async function startAudioProcessing() {
    if (isRecording) return;
    if (!createAudioContext()) return;
    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
        console.warn("WebSocket not ready, cannot start audio processing.");
        return;
    }

    try {
        // Get microphone access
        microphoneStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: TARGET_SAMPLE_RATE, // Request desired rate
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });

        const source = audioContext.createMediaStreamSource(microphoneStream);

        // Use ScriptProcessorNode for simplicity (AudioWorklet is preferred for performance)
        // Adjust bufferSize to influence latency vs. processing load
        // Frame length needs conversion: bufferSize = frameLength * channels
        // We want to send chunks matching AUDIO_FRAME_LENGTH
        // ScriptProcessorNode buffer size must be a power of 2 (e.g., 256, 512, 1024, 2048, 4096, ...)
        // Choose a size that allows sending chunks of AUDIO_FRAME_LENGTH frequently.
        // If AUDIO_FRAME_LENGTH is 512, a BUFFER_SIZE of 4096 works well (8 chunks per call).
        audioProcessor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1); // bufferSize, inputChannels, outputChannels

        audioProcessor.onaudioprocess = (event) => {
            if (!isRecording || !websocket || websocket.readyState !== WebSocket.OPEN) return;

            const inputData = event.inputBuffer.getChannelData(0); // Float32Array [-1.0, 1.0]

            // Convert Float32 to Int16 PCM and process in desired frame lengths
            const samples = inputData.length;
            for (let i = 0; i < samples; i += AUDIO_FRAME_LENGTH) {
                const end = Math.min(i + AUDIO_FRAME_LENGTH, samples);
                const frame = inputData.slice(i, end);

                if (frame.length === AUDIO_FRAME_LENGTH) { // Only send full frames
                    const pcm16 = new Int16Array(AUDIO_FRAME_LENGTH);
                    for (let j = 0; j < AUDIO_FRAME_LENGTH; j++) {
                        pcm16[j] = Math.max(-32768, Math.min(32767, Math.floor(frame[j] * 32767)));
                    }

                    // Convert Int16Array buffer to base64
                    const bufferBytes = pcm16.buffer;
                    const base64String = btoa(String.fromCharCode(...new Uint8Array(bufferBytes)));

                    // Send audio chunk via WebSocket
                    try {
                        console.log('send frame', base64String.length);
                        websocket.send(JSON.stringify({
                            type: "audio_chunk",
                            data: base64String
                        }));
                    } catch (err) {
                        console.error("Error sending audio chunk:", err);
                        // Handle potential WebSocket closure during sending
                        if (websocket.readyState !== WebSocket.OPEN) {
                            stopAudioProcessing();
                        }
                    }
                } else {
                    // Handle partial frames if necessary (e.g., buffer them)
                    // console.log("Partial frame:", frame.length);
                }
            }
        };

        source.connect(audioProcessor);
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
            console.log('AudioContext resumed ->', audioContext.state);
        }
        audioProcessor.connect(audioContext.destination); // Connect to output to avoid issues, though we don't play mic audio

        isRecording = true;
        console.log("Audio processing started.");
        // Status might be updated by server message shortly

    } catch (err) {
        console.error("Error starting audio processing:", err);
        showError(`Could not access microphone: ${err.message}. Please grant permission.`);
        stopAudioProcessing(); // Clean up if failed
    }
}


function stopAudioProcessing() {
    if (!isRecording) return;
    isRecording = false;

    if (audioProcessor) {
        audioProcessor.disconnect();
        audioProcessor.onaudioprocess = null; // Remove handler
        audioProcessor = null;
    }
    if (microphoneStream) {
        microphoneStream.getTracks().forEach(track => track.stop());
        microphoneStream = null;
    }
    // Don't close AudioContext here, needed for TTS playback
    console.log("Audio processing stopped.");
}


// --- WebSocket Communication ---
function connectWebSocket() {
    clearError();
    updateStatus("Connecting...", "status-connecting");
    transcriptDiv.textContent = ''; // Clear transcript on reconnect

    websocket = new WebSocket(WEBSOCKET_URL);

    websocket.onopen = (event) => {
        console.log("WebSocket connected.");
        // Status will be updated by the first message from the server
        // Start listening immediately after connection
        startAudioProcessing();
    };

    websocket.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            // console.log("WebSocket message received:", message); // Debug

            switch (message.type) {
                case "status":
                    updateStatus(message.message, `status-${message.code.split('_')[0]}`); // e.g., status-wakeword, status-listening
                    // Ensure recording is active when expected
                    if (message.code === "wakeword_listening" || message.code === "listening_started") {
                        if (!isRecording) {
                           console.log("Server expects listening, starting audio processing.");
                           startAudioProcessing();
                        }
                    }
                    break;
                case "transcript":
                    updateTranscript(message.text, message.is_final);
                    break;
                case "tts_chunk":
                    // Start buffering if this is the first chunk (implicitly)
                    isBufferingTTS = true;
                    handleTTSChunk(message.data, message.is_final); // Pass the flag
                    break;
                case "error":
                    showError(`Server error: ${message.message}`);
                    // Optionally stop recording on critical errors
                    // stopAudioProcessing();
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
        showError("WebSocket connection error. Trying to reconnect...");
        // Clean up before attempting reconnect
        stopAudioProcessing();
        if (websocket) {
            websocket.close(); // Ensure it's closed
            websocket = null;
        }
        // Implement reconnection logic (e.g., exponential backoff)
        setTimeout(connectWebSocket, 5000); // Simple reconnect after 5 seconds
    };

    websocket.onclose = (event) => {
        console.log("WebSocket closed:", event.code, event.reason);
        stopAudioProcessing(); // Stop mic when disconnected
        if (!event.wasClean) {
            showError(`WebSocket closed unexpectedly (Code: ${event.code}). Check server logs. Attempting to reconnect...`);
            // Schedule reconnection only if not closed cleanly or intentionally
             setTimeout(connectWebSocket, 5000);
        } else {
             updateStatus("Disconnected", "status-error");
        }
        websocket = null; // Ensure websocket is nullified
    };
}


// --- UI Updates ---
function updateStatus(text, cssClass = '') {
    statusDiv.textContent = text;
    statusDiv.className = `status ${cssClass}`; // Reset classes and add the new one
}

function updateTranscript(text, isFinal) {
    if (isFinal) {
        // Maybe keep the final transcript visible for a bit longer
        transcriptDiv.textContent = `You: ${text}`;
    } else {
        // Show interim results dynamically
        transcriptDiv.textContent = `You: ${text}...`;
    }
}


function showError(message) {
    errorDiv.textContent = message;
    console.error(message); // Also log to console
    // Maybe add a class to highlight the error div
}

function clearError() {
    errorDiv.textContent = '';
}


// --- TTS Playback ---
// Decode Base64 string to ArrayBuffer
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
        return null; // Indicate failure
    }
}


function handleTTSChunk(base64Data, isFinal) {
    if (!isBufferingTTS) {
        console.warn("Received TTS chunk but not in buffering state. Ignoring.");
        return;
    }

    const arrayBuffer = base64ToArrayBuffer(base64Data);
    if (arrayBuffer && arrayBuffer.byteLength > 0) {
        currentTTSChunks.push(arrayBuffer); // Add chunk to buffer
    } else if (arrayBuffer && arrayBuffer.byteLength === 0) {
         console.log("Received empty TTS chunk, ignoring.");
         // Don't treat an empty chunk as final unless the flag says so
    } else {
         console.error("Failed to decode TTS chunk, skipping.");
         // Decide if this error should stop buffering? Maybe not.
    }

    // If this is the final chunk, process the complete audio
    if (isFinal) {
        console.log(`Received final TTS chunk. Total chunks: ${currentTTSChunks.length}`);
        isBufferingTTS = false; // Stop buffering state

        if (currentTTSChunks.length > 0) {
            const completeAudioData = concatenateArrayBuffers(currentTTSChunks);
            console.log(`Reassembled TTS audio data: ${completeAudioData.byteLength} bytes`);
            currentTTSChunks = []; // Clear the buffer
            playCompleteTTSAudio(completeAudioData); // Play the full audio
        } else {
            console.warn("Final TTS chunk indicated, but no valid chunks were buffered.");
            currentTTSChunks = []; // Ensure buffer is clear
        }
    }
}


async function playCompleteTTSAudio(completeAudioData) {
    if (!createAudioContext()) { // Ensure context is ready
        console.error("Cannot play TTS, AudioContext not available.");
        showError("Audio playback failed.");
        return;
    }
     if (completeAudioData.byteLength === 0) {
        console.warn("Attempted to play empty audio data.");
        return;
    }

    try {
        console.log("Decoding complete TTS audio data...");
        // Decode the *entire* WAV data
        const audioBuffer = await audioContext.decodeAudioData(completeAudioData);
        console.log(`Successfully decoded audio: Duration ${audioBuffer.duration.toFixed(2)}s`);

        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.onended = () => {
            console.log("TTS playback finished.");
            // No need to trigger next chunk, playback is complete
        };
        source.start();
        console.log("Starting TTS playback...");

    } catch (e) {
        console.error("Error decoding or playing complete TTS audio:", e);
        showError(`Error playing assistant response: ${e.message}`);
        // Log the first few bytes if decoding fails, might hint at format issues
        const firstBytes = new Uint8Array(completeAudioData.slice(0, 12));
        console.error("First 12 bytes of failed audio data:", firstBytes);
    }
}


// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError("Your browser does not support microphone access (getUserMedia). Please use a modern browser like Chrome or Firefox.");
        updateStatus("Error", "status-error");
        return;
    }
     if (!window.WebSocket) {
        showError("Your browser does not support WebSockets. Please use a modern browser.");
        updateStatus("Error", "status-error");
        return;
    }
     if (!window.AudioContext && !window.webkitAudioContext) {
         showError("Your browser does not support the Web Audio API. Please use a modern browser.");
         updateStatus("Error", "status-error");
         return;
     }

    connectWebSocket(); // Start connection on load
});