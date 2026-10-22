import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ReconnectingSocket, type SocketOptions } from "./socket";
import { useStore } from "./store";

// A scripted WebSocket stand-in. The test drives open, message and drop events.
class FakeSocket {
  readyState = 0;
  sent: string[] = [];
  onopen: ((ev: unknown) => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: unknown) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  closed = false;

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.closed = true;
    this.readyState = 3;
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.({});
  }

  receive(data: unknown): void {
    this.onmessage?.({ data });
  }

  drop(): void {
    this.readyState = 3;
    this.onclose?.({});
  }

  fail(): void {
    this.readyState = 3;
    this.onerror?.({});
    this.onclose?.({});
  }
}

function harness(overrides: Partial<SocketOptions> = {}) {
  const sockets: FakeSocket[] = [];
  const onMessage = vi.fn();
  const onConnection = vi.fn();
  const opts: SocketOptions = {
    url: "ws://test/ws",
    onMessage,
    onConnection,
    webSocketFactory: () => {
      const s = new FakeSocket();
      sockets.push(s);
      return s as unknown as WebSocket;
    },
    ...overrides,
  };
  return { sockets, onMessage, onConnection, opts };
}

describe("ReconnectingSocket", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useStore.getState().reset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("reports connected and sends sync on open", () => {
    const { sockets, onConnection, opts } = harness();
    const client = new ReconnectingSocket(opts);
    client.start();
    sockets[0].open();
    expect(onConnection).toHaveBeenLastCalledWith(true);
    expect(sockets[0].sent).toEqual([JSON.stringify({ type: "sync" })]);
    client.stop();
  });

  it("parses frames, drops malformed ones", () => {
    const { sockets, onMessage, opts } = harness();
    const client = new ReconnectingSocket(opts);
    client.start();
    sockets[0].open();
    sockets[0].receive("not json{");
    sockets[0].receive(JSON.stringify({ type: "status", backend_ok: true }));
    expect(onMessage).toHaveBeenCalledTimes(1);
    expect(onMessage.mock.calls[0][0]).toMatchObject({ type: "status" });
    client.stop();
  });

  it("send returns false when not open and true when open", () => {
    const { sockets, opts } = harness();
    const client = new ReconnectingSocket(opts);
    expect(client.send({ type: "send", request_id: 1 })).toBe(false);
    client.start();
    expect(client.send({ type: "send", request_id: 1 })).toBe(false);
    sockets[0].open();
    expect(client.send({ type: "send", request_id: 1 })).toBe(true);
    expect(sockets[0].sent.at(-1)).toBe(JSON.stringify({ type: "send", request_id: 1 }));
    client.stop();
  });

  it("test_reconnect_sync: reconnects with backoff, resends sync and the replay restores the view", () => {
    const { sockets, onConnection, opts } = harness();
    const client = new ReconnectingSocket({
      ...opts,
      onMessage: (m) => useStore.getState().applyMessage(m),
      onConnection: (c) => {
        onConnection(c);
        useStore.getState().setConnected(c);
      },
    });
    client.start();
    sockets[0].open();
    sockets[0].receive(
      JSON.stringify({ type: "request_new", request_id: 7, messages: [], params: {}, tools: null, age_s: 0 }),
    );
    sockets[0].receive(
      JSON.stringify({ type: "request_active", request_id: 7, prompt_tail: "p", max_tokens: null, knowledge: [] }),
    );
    expect(useStore.getState().connected).toBe(true);

    sockets[0].drop();
    expect(useStore.getState().connected).toBe(false);
    expect(sockets).toHaveLength(1);

    vi.advanceTimersByTime(249);
    expect(sockets).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(2);

    sockets[1].open();
    expect(sockets[1].sent).toEqual([JSON.stringify({ type: "sync" })]);
    expect(useStore.getState().connected).toBe(true);

    sockets[1].receive(
      JSON.stringify({ type: "request_new", request_id: 7, messages: [], params: {}, tools: null, age_s: 0 }),
    );
    sockets[1].receive(
      JSON.stringify({ type: "request_active", request_id: 7, prompt_tail: "p", max_tokens: null, knowledge: [] }),
    );
    const s = useStore.getState();
    expect(s.requests.map((r) => r.id)).toEqual([7]);
    expect(s.activeId).toBe(7);
    client.stop();
  });

  it("uses the last backoff delay repeatedly and resets after a successful open", () => {
    const { sockets, opts } = harness({ backoffMs: [100, 200] });
    const client = new ReconnectingSocket(opts);
    client.start();
    sockets[0].fail(); // never opened: delay 100
    vi.advanceTimersByTime(100);
    expect(sockets).toHaveLength(2);
    sockets[1].fail(); // delay 200
    vi.advanceTimersByTime(199);
    expect(sockets).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(3);
    sockets[2].fail(); // still last delay 200
    vi.advanceTimersByTime(200);
    expect(sockets).toHaveLength(4);
    sockets[3].open(); // success resets the sequence
    sockets[3].drop();
    vi.advanceTimersByTime(100);
    expect(sockets).toHaveLength(5);
    client.stop();
  });

  it("stop closes the socket and cancels any pending reconnect", () => {
    const { sockets, onConnection, opts } = harness();
    const client = new ReconnectingSocket(opts);
    client.start();
    sockets[0].open();
    sockets[0].drop();
    client.stop();
    vi.advanceTimersByTime(60_000);
    expect(sockets).toHaveLength(1);
    expect(onConnection).toHaveBeenLastCalledWith(false);
  });

  it("stop closes an open socket without reconnecting", () => {
    const { sockets, opts } = harness();
    const client = new ReconnectingSocket(opts);
    client.start();
    sockets[0].open();
    client.stop();
    expect(sockets[0].closed).toBe(true);
    vi.advanceTimersByTime(60_000);
    expect(sockets).toHaveLength(1);
    expect(client.send({ type: "sync" })).toBe(false);
  });
});
