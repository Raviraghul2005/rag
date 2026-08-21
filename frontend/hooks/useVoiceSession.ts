"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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

// Generation can take a while when providers are struggling (each retried with
// backoff before falling through) — this is a safety net so the UI never waits
// forever if the backend somehow never responds, not the expected happy-path time.
const ANSWER_TIMEOUT_MS = 25000;

export function useVoiceSession() {
  const [state, setState] = useState<VoiceSessionState>(INITIAL_STATE);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const answerTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearAnswerTimeout = useCallback(() => {
    if (answerTimeoutRef.current !== null) {
      clearTimeout(answerTimeoutRef.current);
      answerTimeoutRef.current = null;
    }
  }, []);

  // Stops audio capture only — the WebSocket stays open so the eventual "answer"
  // message (which needs the backend to finish retrieval + generation first) can
  // still arrive. Closing the socket here was the original bug: on "stop", it raced
  // ahead of the backend's response and discarded it every single time, which is
  // why nothing ever appeared to happen after recording — the transcript would show
  // (sent before the socket died) but the answer never could.
  const teardownAudio = useCallback(() => {
    workletRef.current?.disconnect();
    workletRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      void audioContextRef.current.close();
    }
    audioContextRef.current = null;
  }, []);

  const teardownAll = useCallback(() => {
    teardownAudio();
    clearAnswerTimeout();
    wsRef.current?.close();
    wsRef.current = null;
  }, [teardownAudio, clearAnswerTimeout]);

  const start = useCallback(
    async (sarvamLanguageCode: string, strategy: string | null) => {
      teardownAll();
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
            // One question in, one answer out (push-to-talk) — the exchange is
            // complete once this arrives, so close out rather than re-arm for
            // another utterance the user never asked to continue.
            clearAnswerTimeout();
            setState((prev) => ({ ...prev, status: "idle", response: message.payload }));
            wsRef.current?.close();
            wsRef.current = null;
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
        clearAnswerTimeout();
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
    [teardownAll, clearAnswerTimeout]
  );

  const stop = useCallback(() => {
    teardownAudio();
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send("__end__");
      setState((prev) => ({ ...prev, status: "processing" }));
      clearAnswerTimeout();
      answerTimeoutRef.current = setTimeout(() => {
        setState((prev) => ({ ...prev, status: "error", error: "No response from backend" }));
        wsRef.current?.close();
        wsRef.current = null;
      }, ANSWER_TIMEOUT_MS);
    } else {
      setState((prev) => ({ ...prev, status: "idle" }));
    }
  }, [teardownAudio, clearAnswerTimeout]);

  useEffect(() => teardownAll, [teardownAll]);

  return { ...state, start, stop };
}
