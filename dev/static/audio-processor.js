const AUDIO_FRAME_LENGTH = 512; // Must match the main script's setting
const TARGET_SAMPLE_RATE = 16000; // Must match the main script's setting

class AudioRecorderProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        // We expect sampleRate to be passed from the main thread if needed,
        // but typically the context's sampleRate is used implicitly.
        // Check if the actual sample rate matches the target.
        // Note: `sampleRate` is a global variable in AudioWorkletGlobalScope
        if (sampleRate !== TARGET_SAMPLE_RATE) {
            console.warn(`[AudioWorklet] Warning: AudioContext running at ${sampleRate}Hz, target is ${TARGET_SAMPLE_RATE}Hz. Ensure resampling is handled if necessary.`);
            // Ideally, resampling would happen here if needed, but it adds complexity.
            // For now, we assume the input stream is already at TARGET_SAMPLE_RATE
            // due to the constraints passed to getUserMedia and the AudioContext constructor.
        }

        this._buffer = new Float32Array(TARGET_SAMPLE_RATE); // Buffer up to 1 second of audio
        this._bufferPos = 0;
        this._isRecording = false; // Controlled by messages from the main thread

        this.port.onmessage = (event) => {
            if (event.data.command === 'start') {
                console.log('[AudioWorklet] Recording started.');
                this._isRecording = true;
                this._bufferPos = 0; // Reset buffer on start
            } else if (event.data.command === 'stop') {
                console.log('[AudioWorklet] Recording stopped.');
                this._isRecording = false;
                // Optionally process any remaining buffered data on stop?
                // this.processRemainingBuffer();
            }
        };
    }

    // process() is called whenever a new block of audio data is available.
    // The block size is typically 128 frames, but can vary.
    process(inputs, outputs, parameters) {
        // inputs[0] refers to the first input source.
        // inputs[0][0] refers to the first channel of the first input source.
        // It's a Float32Array containing the audio samples for this block.
        const inputChannelData = inputs[0][0];

        // If not recording, or if there's no input data (silence detection?), exit early.
        // Keep the worklet running by returning true.
        if (!this._isRecording || !inputChannelData) {
            return true; // Keep processor alive
        }

        // Append new data to the internal buffer
        const availableSpace = this._buffer.length - this._bufferPos;
        const dataToCopy = Math.min(inputChannelData.length, availableSpace);
        if (dataToCopy < inputChannelData.length) {
            console.warn('[AudioWorklet] Buffer overflow, dropping audio data!');
            // Handle overflow: either drop old data or new data. Dropping new is simpler here.
            // A larger buffer or faster processing might be needed.
        }
        if (dataToCopy > 0) {
             this._buffer.set(inputChannelData.subarray(0, dataToCopy), this._bufferPos);
             this._bufferPos += dataToCopy;
        }


        // Process the buffer in chunks of AUDIO_FRAME_LENGTH
        while (this._bufferPos >= AUDIO_FRAME_LENGTH) {
            // Extract a frame
            const frame = this._buffer.slice(0, AUDIO_FRAME_LENGTH);

            // Convert Float32 to Int16 PCM
            const pcm16 = new Int16Array(AUDIO_FRAME_LENGTH);
            for (let j = 0; j < AUDIO_FRAME_LENGTH; j++) {
                // Clamp and scale
                pcm16[j] = Math.max(-32768, Math.min(32767, Math.floor(frame[j] * 32767)));
            }

            // Send the Int16Array's underlying ArrayBuffer back to the main thread.
            // Marking it as transferable ([pcm16.buffer]) improves performance
            // by transferring ownership instead of copying.
            this.port.postMessage({ type: 'audio_data', buffer: pcm16.buffer }, [pcm16.buffer]);

            // Remove the processed frame from the buffer by shifting the remaining data
            this._buffer.copyWithin(0, AUDIO_FRAME_LENGTH, this._bufferPos);
            this._bufferPos -= AUDIO_FRAME_LENGTH;
        }

        // Return true to keep the processor node alive and processing continuously.
        // Return false would terminate the node.
        return true;
    }

    // Optional: Process any remaining data in the buffer when stopping
    // processRemainingBuffer() {
    //     if (this._bufferPos > 0) {
    //         console.log(`[AudioWorklet] Processing remaining ${this._bufferPos} samples.`);
    //         // Simplified: Convert remaining data directly (might not be full frame)
    //         const remainingFrame = this._buffer.slice(0, this._bufferPos);
    //         const pcm16 = new Int16Array(this._bufferPos);
    //          for (let j = 0; j < this._bufferPos; j++) {
    //             pcm16[j] = Math.max(-32768, Math.min(32767, Math.floor(remainingFrame[j] * 32767)));
    //         }
    //         this.port.postMessage({ type: 'audio_data', buffer: pcm16.buffer }, [pcm16.buffer]);
    //         this._bufferPos = 0;
    //     }
    // }
}

// Register the processor with a unique name.
registerProcessor('audio-recorder-processor', AudioRecorderProcessor);
