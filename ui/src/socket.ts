import type { ClientMessage, ServerMessage } from "./protocol";

export interface SocketOptions {
  url: string;
  onMessage: (msg: ServerMessage) => void;
  onConnection: (connected: boolean) => void;
  webSocketFactory?: (url: string) => WebSocket; // injectable for tests
  backoffMs?: number[]; // default [250, 500, 1000, 2000, 5000]
}

const DEFAULT_BACKOFF_MS = [250, 500, 1000, 2000, 5000];
const OPEN_STATE = 1; // WebSocket.OPEN

export class ReconnectingSocket {
  private readonly opts: SocketOptions;
  private readonly backoff: number[];
  private ws: WebSocket | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private attempt = 0;
  private stopped = true;

  constructor(opts: SocketOptions) {
    this.opts = opts;
    this.backoff = opts.backoffMs ?? DEFAULT_BACKOFF_MS;
  }

  start(): void {
    this.stopped = false;
    if (this.ws === null && this.timer === null) {
      this.open();
    }
  }

  stop(): void {
    this.stopped = true;
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    const sock = this.ws;
    this.ws = null;
    if (sock !== null) {
      sock.close();
      this.opts.onConnection(false);
    }
  }

  send(msg: ClientMessage): boolean {
    const sock = this.ws;
    if (sock === null || sock.readyState !== OPEN_STATE) return false;
    sock.send(JSON.stringify(msg));
    return true;
  }

  private open(): void {
    const factory = this.opts.webSocketFactory ?? ((url: string) => new WebSocket(url));
    let sock: WebSocket;
    try {
      sock = factory(this.opts.url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = sock;

    sock.onopen = () => {
      if (this.ws !== sock) return;
      this.attempt = 0;
      this.opts.onConnection(true);
      sock.send(JSON.stringify({ type: "sync" }));
    };

    sock.onmessage = (ev: MessageEvent) => {
      if (this.ws !== sock) return;
      let parsed: unknown;
      try {
        parsed = JSON.parse(String(ev.data));
      } catch {
        return;
      }
      if (typeof parsed !== "object" || parsed === null || typeof (parsed as { type?: unknown }).type !== "string") {
        return;
      }
      this.opts.onMessage(parsed as ServerMessage);
    };

    const down = () => {
      if (this.ws !== sock) return;
      this.ws = null;
      this.opts.onConnection(false);
      this.scheduleReconnect();
    };
    sock.onclose = down;
    sock.onerror = down;
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.timer !== null) return;
    const index = Math.min(this.attempt, this.backoff.length - 1);
    const delay = this.backoff.length > 0 ? this.backoff[index] : 0;
    this.attempt += 1;
    this.timer = setTimeout(() => {
      this.timer = null;
      if (!this.stopped) this.open();
    }, delay);
  }
}

export function defaultSocketUrl(): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}
