export const AUDIO_SAMPLE_RATE = 16000

/** Decode any browser-supported audio blob to 16 kHz mono PCM, as expected by the Gemma 4 feature extractor. */
export async function decodeToMono16k(blob: Blob): Promise<Float32Array> {
  const ctx = new AudioContext({ sampleRate: AUDIO_SAMPLE_RATE })
  try {
    const buffer = await ctx.decodeAudioData(await blob.arrayBuffer())
    if (buffer.numberOfChannels === 1) return buffer.getChannelData(0).slice()
    const out = new Float32Array(buffer.length)
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      const data = buffer.getChannelData(c)
      for (let i = 0; i < data.length; i++) out[i] += data[i] / buffer.numberOfChannels
    }
    return out
  } finally {
    void ctx.close()
  }
}

export type Recorder = { stop: () => Promise<Blob>; cancel: () => void }

export async function startRecording(): Promise<Recorder> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  const recorder = new MediaRecorder(stream)
  const chunks: Blob[] = []
  recorder.ondataavailable = (e) => chunks.push(e.data)
  recorder.start()

  const release = () => stream.getTracks().forEach((t) => t.stop())
  return {
    stop: () =>
      new Promise((resolve) => {
        recorder.onstop = () => {
          release()
          resolve(new Blob(chunks, { type: recorder.mimeType }))
        }
        recorder.stop()
      }),
    cancel: () => {
      if (recorder.state === 'inactive') return release()
      recorder.onstop = release
      recorder.stop()
    },
  }
}
