import type {
  Ask,
  Decline,
  Forget,
  Pick,
  Rewind,
  Send,
  SetAutonomy,
  Sync,
  Teach,
  Typed,
} from "./protocol";

export const pickMsg = (ask: Ask, tokenId: number): Pick => ({
  type: "pick",
  request_id: ask.request_id,
  pos_id: ask.pos_id,
  token_id: tokenId,
});

export const typedMsg = (ask: Ask, text: string, leadingSpace: boolean, final: boolean): Typed => ({
  type: "typed",
  request_id: ask.request_id,
  pos_id: ask.pos_id,
  text,
  leading_space: leadingSpace,
  final,
});

export const rewindMsg = (requestId: number, toIdx: number): Rewind => ({
  type: "rewind",
  request_id: requestId,
  to_idx: toIdx,
});

export const sendMsg = (requestId: number): Send => ({
  type: "send",
  request_id: requestId,
});

export const declineMsg = (requestId: number, reason: string): Decline => ({
  type: "decline",
  request_id: requestId,
  reason,
});

export const autonomyMsg = (delta: number): SetAutonomy => ({
  type: "set_autonomy",
  delta,
});

export const teachMsg = (text: string): Teach => ({
  type: "teach",
  text,
});

export const forgetMsg = (chunkId: number): Forget => ({
  type: "forget",
  chunk_id: chunkId,
});

export const syncMsg = (): Sync => ({
  type: "sync",
});
