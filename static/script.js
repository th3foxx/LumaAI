// --- START OF FILE script.js ---

// --- Configuration ---
const pageProtocol = window.location.protocol; // Check if page is https: or http:
const wsProtocol = pageProtocol === 'https:' ? 'wss:' : 'ws:'; // Use wss: if page is https:, otherwise ws:
const WEBSOCKET_URL = `${wsProtocol}//${window.location.host}/ws`; // Fixed WebSocket endpoint URL
const TARGET_SAMPLE_RATE = 16000; // Rate expected by backend (Vosk, Porcupine, Cobra)
const TTS_EXPECTED_SAMPLE_RATE = 22050;
const AUDIO_FRAME_LENGTH = 512; // Samples per chunk sent to backend (matches Porcupine/Cobra)
const AUDIO_PROCESSOR_URL = '/static/audio-processor.js';

// --- State ---
let websocket = null;
let audioContext = null;
let audioWorkletNode = null;
let microphoneStream = null;
let isRecording = false;
let currentTTSChunks = []; // Buffer for incoming TTS chunks
let isAudioWorkletReady = false; // Flag to track worklet loading

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

// Helper to convert ArrayBuffer/TypedArray to Base64
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
        // Handle potential InvalidCharacterError if binary string contains chars outside Latin1 range
        console.error("btoa failed:", e);
        // Fallback or alternative encoding might be needed if this happens often
        // For PCM16 data, this should generally not be an issue.
        return null;
    }
}


// --- Audio Processing ---
async function createAudioContextAndWorklet() {
    if (audioContext && isAudioWorkletReady) {
        return true; // Already initialized
    }
    if (!audioContext) {
        try {
            window.AudioContext = window.AudioContext || window.webkitAudioContext;
            audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
            console.log(`AudioContext created with sample rate: ${audioContext.sampleRate}`);
            if (audioContext.sampleRate !== TARGET_SAMPLE_RATE) {
                console.warn(`Requested ${TARGET_SAMPLE_RATE}Hz but got ${audioContext.sampleRate}Hz. Input stream constraints are crucial.`);
            }
        } catch (e) {
            console.error("Error creating AudioContext:", e);
            showError("Could not initialize audio processing. Please use a modern browser.");
            return false;
        }
    }

    // Load the Audio Worklet processor
    if (!isAudioWorkletReady) {
        try {
            console.log(`Loading AudioWorklet module from: ${AUDIO_PROCESSOR_URL}`);
            await audioContext.audioWorklet.addModule(AUDIO_PROCESSOR_URL);
            console.log("AudioWorklet module loaded successfully.");
            isAudioWorkletReady = true;
        } catch (e) {
            console.error("Error loading AudioWorklet module:", e);
            showError(`Could not load audio processor: ${e.message}. Check the console and file path.`);
            // Clean up context if worklet fails? Maybe not, TTS still needs it.
            // audioContext.close();
            // audioContext = null;
            return false;
        }
    }
    return true;
}


async function startAudioProcessing() {
    if (isRecording) return;
    if (!await createAudioContextAndWorklet()) return; // Ensure context and worklet are ready
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
        console.log("Microphone access granted.");

        const source = audioContext.createMediaStreamSource(microphoneStream);

        // Create the AudioWorkletNode
        // The name must match the one registered in audio-processor.js
        audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-recorder-processor');
        console.log("AudioWorkletNode created.");

        // Handle messages received FROM the AudioWorkletProcessor
        audioWorkletNode.port.onmessage = (event) => {
            if (!isRecording || !websocket || websocket.readyState !== WebSocket.OPEN) return;

            if (event.data.type === 'audio_data') {
                // event.data.buffer is the ArrayBuffer containing Int16 PCM data
                const pcm16Buffer = event.data.buffer;

                // Convert ArrayBuffer to base64
                const base64String = arrayBufferToBase64(pcm16Buffer);

                if (base64String) {
                    // Send audio chunk via WebSocket
                    try {
                        // console.log(`Sending frame, base64 length: ${base64String.length}`); // Less verbose logging
                        websocket.send(JSON.stringify({
                            type: "audio_chunk",
                            data: base64String
                        }));
                    } catch (err) {
                        console.error("Error sending audio chunk:", err);
                        // Handle potential WebSocket closure during sending
                        if (websocket.readyState !== WebSocket.OPEN) {
                            console.warn("WebSocket closed while trying to send audio.");
                            stopAudioProcessing(); // Stop processing if WS is closed
                        }
                    }
                } else {
                    console.error("Failed to encode audio buffer to Base64.");
                }
            } else {
                console.warn("Received unknown message type from AudioWorklet:", event.data.type);
            }
        };

        // Connect the microphone source to the worklet node
        source.connect(audioWorkletNode);

        // Connect the worklet node to the destination to keep the audio graph running.
        // This is generally required even if you don't want to play the mic audio back.
        audioWorkletNode.connect(audioContext.destination);

        // Resume context if suspended (e.g., due to browser auto-suspend)
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
            console.log('AudioContext resumed ->', audioContext.state);
        }

        // Send 'start' command to the worklet
        audioWorkletNode.port.postMessage({ command: 'start' });

        isRecording = true;
        console.log("Audio processing started using AudioWorklet.");
        // Status might be updated by server message shortly

    } catch (err) {
        console.error("Error starting audio processing:", err);
        showError(`Could not access microphone or start processing: ${err.message}. Please grant permission.`);
        stopAudioProcessing(); // Clean up if failed
    }
}


function stopAudioProcessing() {
    if (!isRecording && !audioWorkletNode) { // Check both flags
        console.log("Audio processing already stopped or not started.");
        return;
    }
    console.log("Stopping audio processing...");
    isRecording = false; // Set flag immediately

    if (audioWorkletNode) {
        // Send 'stop' command to the worklet
        audioWorkletNode.port.postMessage({ command: 'stop' });

        // Clean up message handler to prevent potential memory leaks
        audioWorkletNode.port.onmessage = null;

        // Disconnect the node from the graph
        // Note: Disconnecting source first might be slightly cleaner
        // Assuming 'source' is not accessible here, disconnect the worklet node itself.
        try {
             audioWorkletNode.disconnect(); // Disconnects from all outputs (destination)
             console.log("AudioWorkletNode disconnected.");
        } catch (e) {
            console.warn("Error disconnecting AudioWorkletNode:", e);
        }
        // Consider explicitly closing the port? Usually not needed.
        // audioWorkletNode.port.close();
        audioWorkletNode = null; // Release reference
    }

    if (microphoneStream) {
        microphoneStream.getTracks().forEach(track => track.stop());
        console.log("Microphone stream stopped.");
        microphoneStream = null;
    }

    // Don't close AudioContext here, needed for TTS playback
    // Don't reset isAudioWorkletReady, the module is still loaded
    console.log("Audio processing stopped.");
    // Update status explicitly if needed, though server messages usually handle this
    // updateStatus("Idle", "status-idle");
}


// --- WebSocket Communication ---
function connectWebSocket() {
    clearError();
    updateStatus("Connecting...", "status-connecting");
    transcriptDiv.textContent = ''; // Clear transcript on reconnect

    // Ensure previous instance is closed before creating a new one
    if (websocket && websocket.readyState !== WebSocket.CLOSED) {
        console.warn("Closing existing WebSocket connection before reconnecting.");
        websocket.close(1000, "Client initiated reconnect"); // 1000 is normal closure
    }
    // Defensive nullification
    websocket = null;

    console.log(`Attempting to connect WebSocket to: ${WEBSOCKET_URL}`);
    websocket = new WebSocket(WEBSOCKET_URL);

    websocket.onopen = (event) => {
        console.log("WebSocket connected.");
        // Status will be updated by the first message from the server
        // Start listening immediately after connection
        // Ensure audio context/worklet is ready before starting processing
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
            // console.log("WebSocket message received:", message.type);

            switch (message.type) {
                case "status":
                    updateStatus(message.message, `status-${message.code.split('_')[0]}`);
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
                case "tts_chunk":
                    // Просто добавляем чанк в буфер. Флаг is_final больше не используется здесь.
                    handleTTSChunk(message.data);
                    break;
                case "tts_finished":
                    console.log("Received TTS finished signal.");
                    // Обрабатываем накопленные чанки
                    processBufferedTTS();
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
        stopAudioProcessing();
        if (websocket) { websocket.close(); }
    };

    websocket.onclose = (event) => {
        console.log(`WebSocket closed: Code=${event.code}, Reason='${event.reason}', WasClean=${event.wasClean}`);
        stopAudioProcessing();
        websocket = null; // Clear reference *after* checking state
        if (!event.wasClean) {
            showError(`WebSocket closed unexpectedly (Code: ${event.code}). Trying to reconnect...`);
             setTimeout(connectWebSocket, 5000);
        } else {
            if (event.code === 1000 && event.reason === "Client initiated reconnect") { updateStatus("Reconnecting...", "status-connecting"); }
            else { updateStatus("Disconnected", "status-error"); }
        }
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
    errorDiv.style.display = 'block'; // Ensure error div is visible
    console.error("UI Error:", message); // Also log to console
}

function clearError() {
    errorDiv.textContent = '';
    errorDiv.style.display = 'none'; // Hide error div
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


function handleTTSChunk(base64Data) {
    const arrayBuffer = base64ToArrayBuffer(base64Data);
    if (arrayBuffer && arrayBuffer.byteLength > 0) {
        currentTTSChunks.push(arrayBuffer); // Просто добавляем в буфер
    } else if (arrayBuffer && arrayBuffer.byteLength === 0) {
         console.log("Received empty TTS chunk, ignoring.");
    } else {
         console.error("Failed to decode TTS chunk, skipping.");
    }
}


function processBufferedTTS() {
    if (currentTTSChunks.length > 0) {
        const completeAudioData = concatenateArrayBuffers(currentTTSChunks);
        console.log(`Reassembled TTS audio data: ${completeAudioData.byteLength} bytes`);
        // Воспроизводим собранные данные
        playCompleteTTSAudio(completeAudioData);
    } else {
        console.warn("TTS finished signal received, but no valid chunks were buffered.");
    }
    // Очищаем буфер в любом случае
    currentTTSChunks = [];
}


async function playCompleteTTSAudio(completeAudioData) {
    // Use createAudioContextAndWorklet to ensure context exists
    if (!await createAudioContextAndWorklet()) {
        console.error("Cannot play TTS, AudioContext not available.");
        showError("Audio playback failed.");
        return;
    }
    if (!completeAudioData || completeAudioData.byteLength === 0) {
        console.warn("Attempted to play empty or invalid audio data.");
        return;
    }

    try {
        console.log(`Processing complete raw TTS audio data (${completeAudioData.byteLength} bytes) expected at ${TTS_EXPECTED_SAMPLE_RATE} Hz...`);
        // Resume context just before playing, in case it suspended
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        // --- Manual AudioBuffer Creation from Raw PCM ---
        // Assumptions: TTS_EXPECTED_SAMPLE_RATE (e.g., 22050 Hz), 16-bit Signed PCM, Mono

        // Use the DEFINED TTS rate here, NOT audioContext.sampleRate
        const sourceSampleRate = TTS_EXPECTED_SAMPLE_RATE;
        const numberOfChannels = 1; // Assuming mono output from Paroli

        // Calculate number of samples: byteLength / (bytes per sample)
        const numberOfSamples = completeAudioData.byteLength / 2; // 16-bit = 2 bytes/sample

        if (completeAudioData.byteLength % 2 !== 0) {
            console.warn("Raw audio data has odd byte length, potential data corruption or incorrect assumption.");
            // You might need to decide how to handle this - e.g., truncate
        }

        console.log(`Creating AudioBuffer: ${numberOfChannels}ch, ${numberOfSamples} samples, ${sourceSampleRate}Hz`);

        // Create an empty AudioBuffer specifying the *source* sample rate
        // The browser will resample this buffer to the audioContext's output rate during playback.
        const audioBuffer = audioContext.createBuffer(numberOfChannels, numberOfSamples, sourceSampleRate);

        // Get the channel data buffer (Float32Array) to fill
        const channelData = audioBuffer.getChannelData(0); // Channel 0 for mono

        // Create an Int16Array view onto the raw ArrayBuffer data
        const pcmData = new Int16Array(completeAudioData);

        // Convert Int16 PCM to Float32 (-1.0 to 1.0) and fill the AudioBuffer
        for (let i = 0; i < numberOfSamples; i++) {
            // Ensure index is within bounds if byte length was odd and not handled above
            if (i < pcmData.length) {
                 channelData[i] = pcmData[i] / 32768.0; // Normalize Int16 to Float32 range
            }
        }
        console.log(`Successfully created AudioBuffer from raw PCM. Duration: ${audioBuffer.duration.toFixed(2)}s`);
        // --- End Manual AudioBuffer Creation ---


        // Play the created AudioBuffer
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.onended = () => {
            console.log("TTS playback finished.");
            // Any cleanup or state change needed after playback finishes
        };
        source.start();
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
    // Hide error div initially
    clearError();

    // Check for necessary APIs
    let supported = true;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError("Your browser does not support microphone access (getUserMedia). Please use a modern browser like Chrome or Firefox.");
        supported = false;
    }
     if (!window.WebSocket) {
        showError("Your browser does not support WebSockets. Please use a modern browser.");
        supported = false;
    }
     if (!window.AudioContext && !window.webkitAudioContext) {
         showError("Your browser does not support the Web Audio API. Please use a modern browser.");
         supported = false;
     }
     // Specifically check for AudioWorklet support
     if (!window.AudioWorkletNode) {
        showError("Your browser does not support AudioWorkletNode. Please use a recent version of a modern browser.");
        supported = false;
     }

    if (!supported) {
        updateStatus("Browser Not Supported", "status-error");
        return; // Stop initialization if essential APIs are missing
    }

    // Start connection on load if supported
    connectWebSocket();
});