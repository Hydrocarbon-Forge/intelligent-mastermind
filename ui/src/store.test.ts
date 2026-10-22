import { beforeEach, describe, expect, it } from "vitest";
import { useStore } from "./store";
import type { Ask, AutoToken, ServerMessage } from "./protocol";

const token = (idx: number, display = `t${idx}`): AutoToken => ({
  idx,
  token_id: idx,
  display,
  confidence: 0.9,
});

const makeAsk = (overrides: Partial<Ask> = {}): Ask => ({
  type: "ask",
  request_id: 1,
  pos_id: 1,
  idx: 2,
  committed_text: "",
  provisional: [],
  space_default: true,
  candidates: [],
  other_mass: 0,
  explore: false,
  tokens_remaining: null,
  ...overrides,
});

const requestNew = (id: number): ServerMessage => ({
  type: "request_new",
  request_id: id,
  messages: [{ role: "user", content: "hi" }],
  params: { model: "x" },
  tools: null,
  age_s: 0,
});

const requestActive = (id: number): ServerMessage => ({
  type: "request_active",
  request_id: id,
  prompt_tail: "tail",
  max_tokens: 64,
  knowledge: ["note"],
});

const apply = (...msgs: ServerMessage[]) => {
  for (const m of msgs) useStore.getState().applyMessage(m);
};

describe("useStore", () => {
  beforeEach(() => {
    useStore.getState().reset();
  });

  it("appends request_new and replaces duplicates in place", () => {
    apply(requestNew(1), requestNew(2));
    apply({ ...requestNew(1), age_s: 5 } as ServerMessage);
    const reqs = useStore.getState().requests;
    expect(reqs.map((r) => r.id)).toEqual([1, 2]);
    expect(reqs[0].ageS).toBe(5);
    expect(typeof reqs[0].receivedAt).toBe("number");
  });

  it("request_active sets activeId and fills the request view", () => {
    apply(requestNew(1), requestActive(1));
    const s = useStore.getState();
    expect(s.activeId).toBe(1);
    expect(s.requests[0].promptTail).toBe("tail");
    expect(s.requests[0].maxTokens).toBe(64);
    expect(s.requests[0].knowledge).toEqual(["note"]);
  });

  it("ask before request_active is ignored", () => {
    apply(requestNew(1), makeAsk({ request_id: 1, pos_id: 1 }));
    expect(useStore.getState().ask).toBeNull();
  });

  it("test_stale_ask_ignored", () => {
    apply(requestNew(1), requestActive(1));
    apply(makeAsk({ pos_id: 5, committed_text: "new" }));
    apply(makeAsk({ pos_id: 3, committed_text: "stale" }));
    const s = useStore.getState();
    expect(s.ask?.pos_id).toBe(5);
    expect(s.committedText).toBe("new");
  });

  it("equal or greater pos_id replaces the held ask", () => {
    apply(requestNew(1), requestActive(1));
    apply(makeAsk({ pos_id: 2 }));
    apply(makeAsk({ pos_id: 2, committed_text: "same" }));
    expect(useStore.getState().committedText).toBe("same");
    apply(makeAsk({ pos_id: 3 }));
    expect(useStore.getState().ask?.pos_id).toBe(3);
  });

  it("test_provisional_display: auto tokens accumulate, rewind truncates, new ask replaces", () => {
    apply(requestNew(1), requestActive(1));
    apply({ type: "auto_run", request_id: 1, tokens: [token(0), token(1)] });
    apply({ type: "auto_run", request_id: 1, tokens: [token(1, "again"), token(2)] });
    let s = useStore.getState();
    expect(s.provisional.map((t) => [t.idx, t.display])).toEqual([
      [0, "t0"],
      [1, "again"],
      [2, "t2"],
    ]);

    apply({ type: "rewound", request_id: 1, to_idx: 1 });
    s = useStore.getState();
    expect(s.provisional.map((t) => t.idx)).toEqual([0]);
    expect(s.ask).toBeNull();

    apply(makeAsk({ pos_id: 4, provisional: [token(0)] }));
    s = useStore.getState();
    expect(s.provisional).toEqual([token(0)]);
    expect(s.ask?.pos_id).toBe(4);
  });

  it("auto_run for a non-active request is ignored", () => {
    apply(requestNew(1), requestActive(1));
    apply({ type: "auto_run", request_id: 9, tokens: [token(0)] });
    expect(useStore.getState().provisional).toEqual([]);
  });

  it("request_done removes the request and clears the active state", () => {
    apply(requestNew(1), requestActive(1), makeAsk({ committed_text: "abc", provisional: [token(0)] }));
    apply({ type: "request_done", request_id: 1, finish_reason: "stop" });
    const s = useStore.getState();
    expect(s.requests).toEqual([]);
    expect(s.activeId).toBeNull();
    expect(s.ask).toBeNull();
    expect(s.provisional).toEqual([]);
    expect(s.committedText).toBe("");
    expect(s.lastDone).toEqual({ requestId: 1, finishReason: "stop" });
  });

  it("status replaces status; setConnected and reset work", () => {
    const status: ServerMessage = {
      type: "status",
      request_id: null,
      backend_ok: true,
      capabilities: {},
      scorer_version: 1,
      autonomy: 0,
      metrics: {},
      knowledge_count: 0,
      notes: [],
    };
    apply(status);
    expect(useStore.getState().status?.backend_ok).toBe(true);
    useStore.getState().setConnected(true);
    expect(useStore.getState().connected).toBe(true);
    apply(requestNew(1), requestActive(1));
    useStore.getState().reset();
    const s = useStore.getState();
    expect(s.connected).toBe(false);
    expect(s.status).toBeNull();
    expect(s.requests).toEqual([]);
    expect(s.activeId).toBeNull();
  });

  it("reconnect replay rebuilds the same view", () => {
    apply(requestNew(1), requestActive(1), makeAsk({ pos_id: 2, committed_text: "ab" }));
    const before = useStore.getState();
    const snapshot = { activeId: before.activeId, committed: before.committedText, ask: before.ask?.pos_id };
    apply(requestNew(1), requestActive(1), makeAsk({ pos_id: 2, committed_text: "ab" }));
    const after = useStore.getState();
    expect(after.requests).toHaveLength(1);
    expect({ activeId: after.activeId, committed: after.committedText, ask: after.ask?.pos_id }).toEqual(snapshot);
  });
});
