// WebSocket protocol between the UI and the Python program. Mirrors humanapi/api/schemas.py one-to-one.
// Every message is a JSON object with a `type` field; request-scoped messages carry `request_id`.

export interface AskCandidate {
  id: number;
  display: string; // token text with visible whitespace markers
  p_you: number;
  p_base: number | null;
  cls: string; // word_start | continuation | punct | layout | digit | end | fragment
}

export interface AutoToken {
  idx: number;
  token_id: number;
  display: string;
  confidence: number;
}

export interface NoteInfo {
  id: number;
  source: string;
  preview: string;
}

// ---- server -> UI

export interface RequestNew {
  type: "request_new";
  request_id: number;
  messages: Array<Record<string, unknown>>;
  params: Record<string, unknown>;
  tools: Array<Record<string, unknown>> | null;
  age_s: number;
}

export interface RequestActive {
  type: "request_active";
  request_id: number;
  prompt_tail: string;
  max_tokens: number | null;
  knowledge: string[];
}

export interface Ask {
  type: "ask";
  request_id: number;
  pos_id: number;
  idx: number;
  committed_text: string;
  provisional: AutoToken[];
  space_default: boolean;
  candidates: AskCandidate[];
  other_mass: number;
  explore: boolean;
  tokens_remaining: number | null;
}

export interface AutoRun {
  type: "auto_run";
  request_id: number;
  tokens: AutoToken[];
}

export interface Rewound {
  type: "rewound";
  request_id: number;
  to_idx: number;
}

export interface RequestDone {
  type: "request_done";
  request_id: number;
  finish_reason: string; // stop | length | cancelled | declined | error
}

export interface Status {
  type: "status";
  request_id: number | null;
  backend_ok: boolean;
  capabilities: Record<string, unknown>;
  scorer_version: number;
  autonomy: number; // slider delta in logit space, about -2 to +2
  metrics: Record<string, number>;
  knowledge_count: number;
  notes: NoteInfo[];
}

export type ServerMessage = RequestNew | RequestActive | Ask | AutoRun | Rewound | RequestDone | Status;

// ---- UI -> server

export interface Sync {
  type: "sync";
  request_id?: number | null;
}

export interface Pick {
  type: "pick";
  request_id: number;
  pos_id: number;
  token_id: number;
}

export interface Typed {
  type: "typed";
  request_id: number;
  pos_id: number;
  text: string;
  leading_space: boolean;
  final: boolean; // false: preview while typing (ignored by the server in v1)
}

export interface Rewind {
  type: "rewind";
  request_id: number;
  to_idx: number;
}

export interface Send {
  type: "send";
  request_id: number;
}

export interface Decline {
  type: "decline";
  request_id: number;
  reason: string;
}

export interface SetAutonomy {
  type: "set_autonomy";
  request_id?: number | null;
  delta: number; // absolute slider setting in logit space
}

export interface Teach {
  type: "teach";
  request_id?: number | null;
  text: string;
}

export interface Forget {
  type: "forget";
  request_id?: number | null;
  chunk_id: number;
}

export type ClientMessage = Sync | Pick | Typed | Rewind | Send | Decline | SetAutonomy | Teach | Forget;
