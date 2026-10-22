import { create, type StoreApi, type UseBoundStore } from "zustand";
import type { Ask, AutoToken, ServerMessage, Status } from "./protocol";

export interface RequestView {
  id: number;
  messages: Array<Record<string, unknown>>;
  params: Record<string, unknown>;
  tools: Array<Record<string, unknown>> | null;
  ageS: number;
  receivedAt: number; // Date.now() when request_new arrived
  promptTail?: string;
  maxTokens?: number | null;
  knowledge?: string[];
}

export interface UiState {
  connected: boolean;
  requests: RequestView[]; // queued and active, in arrival order
  activeId: number | null;
  ask: Ask | null; // the current ask for the active request
  provisional: AutoToken[]; // dimmed, not yet flushed
  committedText: string;
  status: Status | null;
  lastDone: { requestId: number; finishReason: string } | null;
  setConnected(connected: boolean): void;
  applyMessage(msg: ServerMessage): void;
  reset(): void;
}

export type BoundUiStore = UseBoundStore<StoreApi<UiState>>;

const initialState = {
  connected: false,
  requests: [] as RequestView[],
  activeId: null as number | null,
  ask: null as Ask | null,
  provisional: [] as AutoToken[],
  committedText: "",
  status: null as Status | null,
  lastDone: null as { requestId: number; finishReason: string } | null,
};

export const useStore: BoundUiStore = create<UiState>()((set) => ({
  ...initialState,

  setConnected(connected: boolean): void {
    set({ connected });
  },

  applyMessage(msg: ServerMessage): void {
    switch (msg.type) {
      case "request_new": {
        const view: RequestView = {
          id: msg.request_id,
          messages: msg.messages,
          params: msg.params,
          tools: msg.tools,
          ageS: msg.age_s,
          receivedAt: Date.now(),
        };
        set((s) => {
          const index = s.requests.findIndex((r) => r.id === msg.request_id);
          if (index >= 0) {
            const requests = s.requests.slice();
            requests[index] = view;
            return { requests };
          }
          return { requests: [...s.requests, view] };
        });
        return;
      }

      case "request_active": {
        set((s) => ({
          activeId: msg.request_id,
          requests: s.requests.map((r) =>
            r.id === msg.request_id
              ? { ...r, promptTail: msg.prompt_tail, maxTokens: msg.max_tokens, knowledge: msg.knowledge }
              : r,
          ),
          ask: null,
          provisional: [],
          committedText: "",
        }));
        return;
      }

      case "ask": {
        set((s) => {
          if (msg.request_id !== s.activeId) return {};
          if (
            s.ask !== null &&
            s.ask.request_id === msg.request_id &&
            msg.pos_id < s.ask.pos_id
          ) {
            return {};
          }
          return {
            ask: msg,
            provisional: msg.provisional,
            committedText: msg.committed_text,
          };
        });
        return;
      }

      case "auto_run": {
        set((s) => {
          if (msg.request_id !== s.activeId) return {};
          const byIdx = new Map<number, AutoToken>();
          for (const t of s.provisional) byIdx.set(t.idx, t);
          for (const t of msg.tokens) byIdx.set(t.idx, t);
          const provisional = [...byIdx.values()].sort((a, b) => a.idx - b.idx);
          return { provisional };
        });
        return;
      }

      case "rewound": {
        set((s) => {
          if (msg.request_id !== s.activeId) return {};
          return {
            provisional: s.provisional.filter((t) => t.idx < msg.to_idx),
            ask: null,
          };
        });
        return;
      }

      case "request_done": {
        set((s) => {
          const update: Partial<UiState> = {
            requests: s.requests.filter((r) => r.id !== msg.request_id),
            lastDone: { requestId: msg.request_id, finishReason: msg.finish_reason },
          };
          if (msg.request_id === s.activeId) {
            update.activeId = null;
            update.ask = null;
            update.provisional = [];
            update.committedText = "";
          }
          return update;
        });
        return;
      }

      case "status": {
        set({ status: msg });
        return;
      }
    }
  },

  reset(): void {
    set({ ...initialState });
  },
}));
