// static/script.js

// Configuration from server (main.py's AudioSettings)
const TARGET_SAMPLE_RATE = 16000;
const FRAME_LENGTH = 512; // Samples per chunk to send

// DOM Elements
const connectButton = document.getElementById('connectButton');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const statusDiv = document.getElementById('status');
const transcriptDiv = document.getElementById('transcript');
const partialTranscriptDiv = document.getElementById('partial-transcript');
const errorLogDiv = document.getElementById('error-log');
const activationSound = document.getElementById('activationSound');
const voiceVisualizer = document.getElementById('voice-visualizer'); // Новый элемент

let websocket = null;
let audioContext = null;
let microphoneStream = null;
let audioProcessorNode = null;
let gainNode = null; 

let ttsAudioBufferQueue = []; 
let ttsPlaybackSampleRate = 24000; 
let ttsPlaybackChannels = 1;       

let isClientListening = false;
let isPlayingTTS = false; // Отслеживаем, проигрывается ли TTS

// --- Управление состоянием визуализатора ---
function setVisualizerState(state) { // state: 'idle', 'listening', 'speaking'
    voiceVisualizer.classList.remove('idle', 'listening', 'speaking');
    if (state) {
        voiceVisualizer.classList.add(state);
    } else { // Если состояние не указано, по умолчанию 'idle'
        voiceVisualizer.classList.add('idle');
    }
}


// --- WebSocket Handling ---
function connectWebSocket() {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        logError("Already connected.");
        return;
    }

    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    
    websocket = new WebSocket(wsUrl);
    updateStatus("Connecting...");
    setVisualizerState('idle'); // Устанавливаем состояние покоя при попытке подключения

    websocket.onopen = () => {
        updateStatus("Connected. Server is ready.");
        logError("");
        connectButton.disabled = true;
        startButton.disabled = false;
        stopButton.disabled = true;
        // Текст кнопок уже изменен в HTML, здесь можно не дублировать
        setVisualizerState('idle'); 
    };

    websocket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleServerMessage(message);
    };

    websocket.onerror = (error) => {
        logError("WebSocket Error. Check console.");
        console.error("WebSocket Error:", error);
        updateStatus("Connection Error");
        resetUIForDisconnect();
    };

    websocket.onclose = (event) => {
        updateStatus(`Disconnected: ${event.reason || 'No reason given'} (Code: ${event.code})`);
        console.log("WebSocket closed:", event);
        stopAudioCapture(); // Это также обновит визуализатор
        resetUIForDisconnect();
    };
}

function resetUIForDisconnect() {
    connectButton.disabled = false;
    startButton.disabled = true;
    stopButton.disabled = true;
    isClientListening = false;
    isPlayingTTS = false; // Сбрасываем флаг TTS
    setVisualizerState('idle'); // Сброс визуализатора в состояние покоя
}

function handleServerMessage(message) {
    switch (message.type) {
        case "status":
            updateStatus(message.message); 
            if (message.code === "play_activation_sound_cue") {
                playActivationSound();
            }
            if (message.code === "listening_started") {
                partialTranscriptDiv.textContent = "";
                transcriptDiv.textContent = "";
                 // Если сервер говорит, что начал слушать, а клиент еще нет (например, после wake word)
                if (!isClientListening && !isPlayingTTS) {
                    // setVisualizerState('listening'); // Можно рассмотреть, если сервер инициирует слушание
                }
            }
            break;
        case "transcript":
            if (message.is_final) {
                transcriptDiv.textContent = message.text;
                partialTranscriptDiv.textContent = "";
            } else {
                partialTranscriptDiv.textContent = message.text;
            }
            break;
        case "tts_info":
            ttsPlaybackSampleRate = message.sample_rate;
            ttsPlaybackChannels = message.channels;
            console.log(`TTS Info: SR=${ttsPlaybackSampleRate}Hz, Channels=${ttsPlaybackChannels}`);
            ttsAudioBufferQueue = []; 
            break;
        case "tts_chunk":
            const rawString = atob(message.data);
            const chunkBytes = new Uint8Array(rawString.length);
            for (let i = 0; i < rawString.length; i++) {
                chunkBytes[i] = rawString.charCodeAt(i);
            }
            ttsAudioBufferQueue.push(chunkBytes);
            break;
        case "tts_finished":
            playBufferedTTS();
            break;
        case "error":
            logError(message.message);
            break;
        default:
            console.warn("Unknown message type from server:", message.type);
    }
}

// --- Audio Processing & Streaming ---
async function startAudioCapture() {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
        logError("Not connected to server.");
        return;
    }
    if (isClientListening) {
        logError("Already sending audio.");
        return;
    }

    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: TARGET_SAMPLE_RATE
        });
        
        if (audioContext.sampleRate !== TARGET_SAMPLE_RATE) {
            console.warn(`AudioContext running at ${audioContext.sampleRate}Hz, not requested ${TARGET_SAMPLE_RATE}Hz.`);
            logError(`Audio system running at ${audioContext.sampleRate}Hz. Quality might be affected.`);
        }

        microphoneStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: TARGET_SAMPLE_RATE,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            },
            video: false
        });

        const source = audioContext.createMediaStreamSource(microphoneStream);
        gainNode = audioContext.createGain();
        gainNode.gain.value = 0; 

        if (typeof AudioWorkletNode !== 'undefined') {
            await audioContext.audioWorklet.addModule('/static/audio-processor.js');
            audioProcessorNode = new AudioWorkletNode(audioContext, 'audio-stream-processor', {
                processorOptions: {
                    targetSampleRate: TARGET_SAMPLE_RATE,
                    frameLength: FRAME_LENGTH 
                }
            });

            audioProcessorNode.port.onmessage = (event) => {
                if (event.data.type === 'audioData' && websocket && websocket.readyState === WebSocket.OPEN && isClientListening) {
                    sendAudioChunk(event.data.buffer);
                } else if (event.data.type === 'debug') {
                    console.log('AudioWorklet:', event.data.message);
                }
            };
            source.connect(audioProcessorNode).connect(gainNode).connect(audioContext.destination);
        } else {
            console.warn("AudioWorklet not supported, using ScriptProcessorNode (deprecated and less reliable).");
            logError("AudioWorklet not supported. Using fallback (less optimal).");
            setupScriptProcessorNode(source); 
        }
        
        isClientListening = true;
        startButton.disabled = true;
        stopButton.disabled = false;
        console.log("Client: Started sending audio."); 
        logError("");
        if (!isPlayingTTS) { // Не переопределять состояние 'speaking', если TTS активен
            setVisualizerState('listening');
        }

    } catch (err) {
        logError(`Error getting microphone: ${err.message}`);
        console.error("Error initializing audio capture:", err);
        stopAudioCapture(); // Обработает состояние визуализатора
    }
}

function setupScriptProcessorNode(source) { 
    const bufferSize = FRAME_LENGTH; 
    audioProcessorNode = audioContext.createScriptProcessor(bufferSize, 1, 1);

    audioProcessorNode.onaudioprocess = (audioProcessingEvent) => {
        if (!isClientListening || !websocket || websocket.readyState !== WebSocket.OPEN) return;
        const pcmData = audioProcessingEvent.inputBuffer.getChannelData(0);
        const int16Pcm = new Int16Array(pcmData.length);
        for (let i = 0; i < pcmData.length; i++) {
            let s = Math.max(-1, Math.min(1, pcmData[i]));
            int16Pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        sendAudioChunk(int16Pcm.buffer);
    };
    source.connect(audioProcessorNode);
    audioProcessorNode.connect(gainNode); 
    gainNode.connect(audioContext.destination);
}

function stopAudioCapture() {
    if (microphoneStream) {
        microphoneStream.getTracks().forEach(track => track.stop());
        microphoneStream = null;
    }
    if (audioProcessorNode) {
        audioProcessorNode.disconnect();
        if (audioProcessorNode.port) audioProcessorNode.port.close();
        audioProcessorNode = null;
    }
    if (gainNode) {
        gainNode.disconnect();
        gainNode = null;
    }

    isClientListening = false;
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        startButton.disabled = false;
    } else {
        startButton.disabled = true;
    }
    stopButton.disabled = true;
    console.log("Client: Stopped sending audio.");
    if (!isPlayingTTS) { // Изменять на 'idle' только если TTS не активен
        setVisualizerState('idle');
    }
}

function sendAudioChunk(arrayBuffer) {
    if (websocket && websocket.readyState === WebSocket.OPEN && isClientListening) {
        const base64String = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
        websocket.send(JSON.stringify({ type: "audio_chunk", data: base64String }));
    }
}

// --- TTS Playback ---
async function playBufferedTTS() {
    if (ttsAudioBufferQueue.length === 0) {
        console.log("No TTS audio to play.");
        // Если TTS не играет и клиент не слушает, переходим в idle
        if (!isClientListening) {
            setVisualizerState('idle');
        }
        isPlayingTTS = false; // Убедимся, что флаг сброшен
        return;
    }
    
    isPlayingTTS = true;
    setVisualizerState('speaking'); // TTS сейчас начнется

    if (!audioContext || audioContext.state === 'closed') {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    let totalLength = 0;
    ttsAudioBufferQueue.forEach(chunk => totalLength += chunk.byteLength);
    const concatenatedBytes = new Uint8Array(totalLength);
    let offset = 0;
    ttsAudioBufferQueue.forEach(chunk => {
        concatenatedBytes.set(chunk, offset);
        offset += chunk.byteLength;
    });
    ttsAudioBufferQueue = [];

    const pcmInt16 = new Int16Array(concatenatedBytes.buffer);
    const pcmFloat32 = new Float32Array(pcmInt16.length);
    for (let i = 0; i < pcmInt16.length; i++) {
        pcmFloat32[i] = pcmInt16[i] / 32768.0;
    }

    try {
        const audioBuffer = audioContext.createBuffer(
            ttsPlaybackChannels,
            pcmFloat32.length / ttsPlaybackChannels,
            ttsPlaybackSampleRate
        );

        if (ttsPlaybackChannels === 1) {
            audioBuffer.copyToChannel(pcmFloat32, 0);
        } else {
            // (код для многоканального аудио остается прежним)
            for (let c = 0; c < ttsPlaybackChannels; c++) {
                const channelData = new Float32Array(audioBuffer.length);
                for (let i = 0; i < audioBuffer.length; i++) {
                    channelData[i] = pcmFloat32[i * ttsPlaybackChannels + c];
                }
                audioBuffer.copyToChannel(channelData, c);
            }
        }

        const sourceNode = audioContext.createBufferSource();
        sourceNode.buffer = audioBuffer;
        sourceNode.connect(audioContext.destination);
        sourceNode.start();
        sourceNode.onended = () => {
            console.log("TTS playback finished.");
            isPlayingTTS = false;
            // После окончания TTS, решаем, в какое состояние перейти
            if (isClientListening) {
                setVisualizerState('listening'); // Возвращаемся к 'listening', если микрофон все еще активен
            } else {
                setVisualizerState('idle'); // Иначе, в 'idle'
            }
        };
    } catch (e) {
        logError("Error playing TTS: " + e.message);
        console.error("Error decoding/playing TTS audio:", e);
        isPlayingTTS = false; // Сброс флага при ошибке
        if (isClientListening) {
            setVisualizerState('listening');
        } else {
            setVisualizerState('idle');
        }
    }
}

// --- UI Update Functions ---
function updateStatus(text) {
    statusDiv.textContent = `Status: ${text}`;
}

function logError(text) {
    errorLogDiv.textContent = text;
    if (text) console.error(`Client UI: ${text}`);
}

function playActivationSound() {
    activationSound.play().catch(e => console.warn("Failed to play activation sound:", e));
}

// --- Event Listeners ---
connectButton.addEventListener('click', connectWebSocket);
startButton.addEventListener('click', startAudioCapture);
stopButton.addEventListener('click', stopAudioCapture);

window.addEventListener('beforeunload', () => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.close(1000, "Client navigating away");
    }
    stopAudioCapture();
});

// Initial state
resetUIForDisconnect(); // Это установит visualizer в 'idle'
updateStatus("Disconnected. Click 'Connect'.");
// setVisualizerState('idle'); // Уже вызывается в resetUIForDisconnect