import type { Device, LiveMetadata, LiveSource } from "../types";

export type LiveStreamState = {
  readonly liveRunning: boolean;
  readonly liveMeta: LiveMetadata | null;
  readonly liveError: string;
};

export function useLiveStream() {
  let liveRunning = $state(false);
  let liveMeta = $state<LiveMetadata | null>(null);
  let liveError = $state("");

  let ws: WebSocket | null = null;
  let wsReadyPromise: Promise<WebSocket> | null = null;
  let outputCanvas = $state<HTMLCanvasElement | null>(null);
  let outputCtx = $state<CanvasRenderingContext2D | null>(null);
  let objectUrl: string | null = null;

  function wsUrl(): string {
    const cfg = (window as any).__FOOTBALLAI_WS__ as string | undefined;
    if (cfg) return cfg;
    const loc = window.location;
    const protocol = loc.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${loc.host}/ws`;
  }

  function ensureOutputCtx(): CanvasRenderingContext2D | null {
    if (!outputCtx && outputCanvas) {
      outputCtx = outputCanvas.getContext("2d");
    }
    return outputCtx;
  }

  function bindCanvas(canvas: HTMLCanvasElement | null): void {
    outputCanvas = canvas;
    outputCtx = canvas?.getContext("2d") ?? null;
  }

  function handleWsMessage(event: MessageEvent<string | Blob>): void {
    if (typeof event.data === "string") {
      try {
        const payload = JSON.parse(event.data) as { type: string } & Record<string, unknown>;
        if (payload.type === "metadata") {
          liveMeta = payload as unknown as LiveMetadata;
        } else if (payload.type === "error") {
          liveError = (payload.message as string) ?? "Server error";
        }
      } catch {
        // ignore non-JSON text
      }
    } else if (event.data instanceof Blob) {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(event.data);
      const ctx = ensureOutputCtx();
      const canvas = outputCanvas;
      if (ctx && canvas) {
        const img = new Image();
        img.onload = () => {
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          ctx.drawImage(img, 0, 0);
        };
        img.src = objectUrl;
      }
    }
  }

  function openWsIfNeeded(): Promise<WebSocket> {
    if (ws?.readyState === WebSocket.OPEN) return Promise.resolve(ws);
    if (wsReadyPromise) return wsReadyPromise;

    wsReadyPromise = new Promise<WebSocket>((resolve, reject) => {
      if (ws && ws.readyState !== WebSocket.CLOSED && ws.readyState !== WebSocket.CLOSING) {
        try {
          ws.close();
        } catch {
          // ignore
        }
      }

      const socket = new WebSocket(wsUrl());
      ws = socket;
      socket.binaryType = "blob";
      socket.onmessage = handleWsMessage;
      socket.onopen = () => resolve(socket);
      socket.onerror = () => reject(new Error("WebSocket error. Is the Python server running?"));
      socket.onclose = () => {
        if (ws === socket) {
          ws = null;
          liveRunning = false;
        }
      };
    }).finally(() => {
      wsReadyPromise = null;
    });

    return wsReadyPromise;
  }

  async function start(source: LiveSource, device: Device): Promise<void> {
    liveError = "";
    liveMeta = null;
    liveRunning = true;

    try {
      const socket = await openWsIfNeeded();
      socket.send(JSON.stringify({ action: "configure", options: { device } }));
      socket.send(JSON.stringify({ action: "live_start", source, options: { device } }));
    } catch (err) {
      liveRunning = false;
      liveError = err instanceof Error ? err.message : "Failed to start live stream";
    }
  }

  function stop(): void {
    liveRunning = false;
    try {
      ws?.send(JSON.stringify({ action: "live_stop" }));
    } catch {
      // ignore
    }
    ws?.close();
    ws = null;
    wsReadyPromise = null;
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
  }

  function setError(message: string): void {
    liveError = message;
  }

  return {
    get liveRunning() {
      return liveRunning;
    },
    get liveMeta() {
      return liveMeta;
    },
    get liveError() {
      return liveError;
    },
    start,
    stop,
    setError,
    bindCanvas,
  };
}
