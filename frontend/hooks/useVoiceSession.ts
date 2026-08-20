"use client";

import { useCallback, useRef, useState } from "react";
import { wsUrl } from "@/lib/config";
import type { PipelineResponse, ServerMessage } from "@/lib/types";

export type VoiceStatus =
  | "idle"
  | "requesting_mic"
  | "connecting"
  | "listening"
  | "processing"
  | "error";

export interface VoiceSessionState {
  status: VoiceStatus;
  partialTranscript: string;
  finalTranscript: string | null;
  response: PipelineResponse | null;
  error: string | null;
}

const INITIAL_STATE: VoiceSessionState = {
  status: "idle",
  partialTranscript: "",
  finalTranscript: null,
  response: null,
  error: null,
};

export function useVoiceSession() {
  const [state, setState] = useState<VoiceSessionState>(INITIAL_STATE);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);

  const teardown = useCallback(() => {
    workletRef.current?.disconnect();
    workletRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      void audioContextRef.current.close();
    }
    audioContextRef.current = null;
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  const start = useCallback(
    async (sarvamLanguageCode: string, strategy: string | null) => {
      teardown();
      setState({ ...INITIAL_STATE, status: "requesting_mic" });

      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
      } catch {
        setState((prev) => ({ ...prev, status: "error", error: "Microphone access denied" }));
        return;
      }
      streamRef.current = stream;

      setState((prev) => ({ ...prev, status: "connecting" }));

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      await audioContext.audioWorklet.addModule("/pcm-worklet.js");

      const params = new URLSearchParams({ language: sarvamLanguageCode });
      if (strategy) params.set("strategy", strategy);
      const ws = new WebSocket(wsUrl(`/ws/transcribe?${params.toString()}`));
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        setState((prev) => ({ ...prev, status: "listening" }));
      };

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data) as ServerMessage;
        switch (message.type) {
          case "partial_transcript":
            setState((prev) => ({ ...prev, partialTranscript: message.text }));
            break;
          case "final_transcript":
            setState((prev) => ({
              ...prev,
              status: "processing",
              finalTranscript: message.text,
              partialTranscript: "",
            }));
            break;
          case "answer":
            setState((prev) => ({ ...prev, status: "listening", response: message.payload }));
            break;
          case "stt_error":
            setState((prev) => ({ ...prev, error: message.message }));
            break;
        }
      };

      ws.onerror = () => {
        setState((prev) => ({ ...prev, status: "error", error: "Connection to backend failed" }));
      };

      ws.onclose = () => {
        setState((prev) => (prev.status === "error" ? prev : { ...prev, status: "idle" }));
      };

      const source = audioContext.createMediaStreamSource(stream);
      const worklet = new AudioWorkletNode(audioContext, "pcm-worklet-processor", {
        processorOptions: { targetSampleRate: 16000 },
      });
      worklet.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(event.data);
        }
      };
      source.connect(worklet);
      workletRef.current = worklet;
    },
    [teardown]
  );

  const stop = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send("__end__");
    }
    teardown();
    setState((prev) => ({ ...prev, status: "idle" }));
  }, [teardown]);

  return { ...state, start, stop };
}
