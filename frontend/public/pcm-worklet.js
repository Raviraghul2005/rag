// AudioWorkletProcessor that resamples the browser's native mic rate (typically
// 48kHz) down to 16kHz mono linear16 PCM — the only format Sarvam's realtime STT
// WebSocket accepts (see backend/app/stt/sarvam_client.py). Runs on the audio
// rendering thread, not the main thread, so it can't block on network I/O; it only
// posts finished Int16 buffers back to the main thread, which owns the WebSocket.
//
// Resampling is linear interpolation, not a proper windowed-sinc resampler — audible
// quality is more than sufficient for speech transcription and it's cheap enough to
// run in real time on the audio thread. A carry buffer across process() calls keeps
// the resampling phase continuous at block boundaries instead of restarting from
// scratch on every ~128-sample render quantum.
class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const targetSampleRate = options.processorOptions?.targetSampleRate ?? 16000;
    this.resampleRatio = sampleRate / targetSampleRate;
    this.fractionalIndex = 0;
    this.carry = new Float32Array(0);
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel || channel.length === 0) return true;

    const samples = new Float32Array(this.carry.length + channel.length);
    samples.set(this.carry, 0);
    samples.set(channel, this.carry.length);

    const outLength = Math.max(
      0,
      Math.floor((samples.length - 1 - this.fractionalIndex) / this.resampleRatio) + 1
    );
    const out = new Int16Array(outLength);
    let idx = this.fractionalIndex;
    for (let i = 0; i < outLength; i++) {
      const lower = Math.floor(idx);
      const upper = Math.min(lower + 1, samples.length - 1);
      const frac = idx - lower;
      const sample = samples[lower] * (1 - frac) + samples[upper] * frac;
      const clamped = Math.max(-1, Math.min(1, sample));
      out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      idx += this.resampleRatio;
    }

    const consumed = Math.floor(idx);
    this.carry = samples.slice(consumed);
    this.fractionalIndex = idx - consumed;

    if (out.length > 0) {
      this.port.postMessage(out.buffer, [out.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-worklet-processor", PCMWorkletProcessor);
