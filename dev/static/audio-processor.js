// static/audio-processor.js

class AudioStreamProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        // sampleRate is a global variable in AudioWorkletGlobalScope (the AudioContext's sampleRate)
        this.inputSampleRate = sampleRate; 
        this.targetSampleRate = options.processorOptions.targetSampleRate || 16000;
        this.frameLength = options.processorOptions.frameLength || 512;
        
        this._buffer = new Float32Array(this.frameLength * 2); // Internal buffer
        this._bufferPos = 0;

        // Basic resampling: if input SR is different from target SR, we need to handle it.
        // This worklet assumes the AudioContext is already running at targetSampleRate,
        // or that minor differences are handled by this very simple resampler.
        // A robust solution would involve a proper resampling library if significant SR differences occur.
        this.resampleStep = this.inputSampleRate / this.targetSampleRate;
        this.resampleRemainder = 0; // For fractional step resampling

        if (this.inputSampleRate !== this.targetSampleRate) {
            this.port.postMessage({ type: 'debug', message: `AudioWorklet: Input SR ${this.inputSampleRate}, Target SR ${this.targetSampleRate}. Resample step: ${this.resampleStep.toFixed(2)}` });
        } else {
            this.port.postMessage({ type: 'debug', message: `AudioWorklet: Input SR ${this.inputSampleRate} matches Target SR ${this.targetSampleRate}.` });
        }
    }

    process(inputs, outputs, parameters) {
        const inputChannelData = inputs[0]?.[0]; // First channel of the first input

        if (!inputChannelData || inputChannelData.length === 0) {
            return true; // Keep processor alive
        }
        
        let currentInputSamples = inputChannelData;

        // Simplified resampling: Process input samples to fit into our internal buffer
        // This is a very basic approach (nearest neighbor or skipping/duplicating based on ratio)
        // and not ideal for audio quality if sample rates differ significantly.
        // The primary expectation is that AudioContext runs at targetSampleRate.
        if (this.inputSampleRate !== this.targetSampleRate) {
            const resampled = [];
            this.resampleRemainder += currentInputSamples.length;
            while (this.resampleRemainder >= this.resampleStep) {
                // Calculate the approximate original index
                // This is a simplification; proper resampling is complex.
                const originalIndex = Math.floor((currentInputSamples.length - this.resampleRemainder) / this.resampleStep * (currentInputSamples.length / this.resampleStep));
                resampled.push(currentInputSamples[Math.min(Math.max(0, originalIndex), currentInputSamples.length - 1)] || 0);
                this.resampleRemainder -= this.resampleStep;
            }
            if (resampled.length > 0) {
                 currentInputSamples = new Float32Array(resampled);
            } else {
                // Not enough samples to form a resampled output yet, or resampleStep is very large
                // This path needs careful handling in a real resampler.
                // For now, if no resampled output, we skip this block.
            }
        }


        for (let i = 0; i < currentInputSamples.length; i++) {
            if (this._bufferPos >= this._buffer.length) {
                // Buffer is full, should not happen if flushed correctly
                // For safety, drop oldest and continue, or just reset.
                // This indicates an issue if frameLength is small relative to process block.
                console.warn('AudioWorklet: Internal buffer overflow, resetting.');
                this._bufferPos = 0; 
            }
            this._buffer[this._bufferPos++] = currentInputSamples[i];

            if (this._bufferPos >= this.frameLength) {
                const frameData = this._buffer.slice(0, this.frameLength);
                
                const int16Pcm = new Int16Array(this.frameLength);
                for (let j = 0; j < this.frameLength; j++) {
                    let s = Math.max(-1, Math.min(1, frameData[j]));
                    int16Pcm[j] = s < 0 ? s * 0x8000 : s * 0x7FFF; // Scale to 16-bit
                }

                this.port.postMessage({ type: 'audioData', buffer: int16Pcm.buffer }, [int16Pcm.buffer]);
                
                // Shift remaining data in buffer
                const remaining = this._buffer.slice(this.frameLength, this._bufferPos);
                this._buffer.fill(0); 
                this._buffer.set(remaining);
                this._bufferPos = remaining.length;
            }
        }
        return true; 
    }
}

registerProcessor('audio-stream-processor', AudioStreamProcessor);