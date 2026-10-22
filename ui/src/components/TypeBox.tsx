import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { Ask, AskCandidate, ClientMessage } from "../protocol";
import { pickMsg, rewindMsg, sendMsg, typedMsg } from "../messages";
import { flushedIdx, startsWithPunctuation } from "../typing";
import styles from "./components.module.css";

export interface TypeBoxProps {
  ask: Ask | null; // null: disabled
  candidates: AskCandidate[]; // the rows currently shown (after prefix filtering)
  sendKey: string; // e.g. "Ctrl+Enter"
  onSend: (msg: ClientMessage) => void;
  onTypedChange: (typed: string, leadingSpace: boolean) => void;
  onHighlight: (index: number) => void;
  highlight: number;
}

export interface KeyLike {
  key: string;
  ctrlKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
  metaKey: boolean;
}

// True when the event matches a combo such as "Ctrl+Enter" (modifiers must match exactly).
export function matchesKey(e: KeyLike, combo: string): boolean {
  const parts = combo
    .split("+")
    .map((p) => p.trim().toLowerCase())
    .filter((p) => p.length > 0);
  const key = parts.pop() ?? "";
  const mods = new Set(parts);
  return (
    e.key.toLowerCase() === key &&
    e.ctrlKey === mods.has("ctrl") &&
    e.altKey === mods.has("alt") &&
    e.shiftKey === mods.has("shift") &&
    e.metaKey === (mods.has("meta") || mods.has("cmd"))
  );
}

export default function TypeBox({
  ask,
  candidates,
  sendKey,
  onSend,
  onTypedChange,
  onHighlight,
  highlight,
}: TypeBoxProps) {
  const [typed, setTyped] = useState("");
  // marker: the leading-space flag of the word being typed (before the punctuation rule)
  const [marker, setMarker] = useState<boolean>(ask?.space_default ?? false);
  // pending: a space was typed since the last committed word, so the next word starts with one
  const [pending, setPending] = useState(false);

  const effective = marker && !startsWithPunctuation(typed);

  useEffect(() => {
    onTypedChange(typed, effective);
  }, [typed, effective]);

  // A new ask recomputes the marker only while the box is empty.
  useEffect(() => {
    if (ask && typed === "") {
      setMarker(pending || ask.space_default);
    }
  }, [ask]);

  const defaultMarker = ask?.space_default ?? false;

  function sendFinal(text: string, leading: boolean): void {
    if (ask) onSend(typedMsg(ask, text, leading, true));
  }

  function resetWord(nextPending: boolean, nextMarker: boolean): void {
    setTyped("");
    setPending(nextPending);
    setMarker(nextMarker);
  }

  function onKeyDown(e: KeyboardEvent<HTMLInputElement>): void {
    if (!ask) {
      e.preventDefault();
      return;
    }
    const n = candidates.length;

    if (matchesKey(e, sendKey)) {
      e.preventDefault();
      if (typed !== "") sendFinal(typed, effective);
      onSend(sendMsg(ask.request_id));
      resetWord(false, defaultMarker);
      return;
    }

    if (e.altKey && /^[1-8]$/.test(e.key)) {
      e.preventDefault();
      const cand = candidates[Number(e.key) - 1];
      if (cand) onSend(pickMsg(ask, cand.id));
      return;
    }

    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (n === 0) return;
      const cur = Math.min(highlight, n - 1);
      const next = e.key === "ArrowDown" ? (cur + 1) % n : (cur - 1 + n) % n;
      onHighlight(next);
      return;
    }

    if (e.key === "Tab" && !e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey) {
      if (n === 0) return;
      e.preventDefault();
      const cand = candidates[Math.min(highlight, n - 1)];
      if (cand) onSend(pickMsg(ask, cand.id));
      return;
    }

    if (e.ctrlKey || e.metaKey) return; // leave browser shortcuts alone

    if (e.key === "Enter") {
      e.preventDefault();
      if (e.altKey || e.shiftKey) return;
      const text = `${typed}\n`;
      const leading = typed !== "" && effective;
      sendFinal(text, leading);
      resetWord(false, defaultMarker);
      return;
    }

    if (e.key === "Backspace") {
      e.preventDefault();
      if (typed !== "") {
        setTyped(typed.slice(0, -1));
      } else if (marker) {
        setMarker(false);
      } else if (ask.provisional.length > 0) {
        onSend(rewindMsg(ask.request_id, ask.idx - 1));
      }
      return;
    }

    if (e.key === "Escape") {
      e.preventDefault();
      if (ask.provisional.length > 0) {
        onSend(rewindMsg(ask.request_id, flushedIdx(ask)));
      }
      return;
    }

    if (e.key === " ") {
      e.preventDefault();
      if (typed === "") {
        setPending(true);
        setMarker(true);
      } else {
        sendFinal(typed, effective);
        resetWord(true, true);
      }
      return;
    }

    if (e.key.length === 1 && !e.altKey) {
      e.preventDefault();
      if (startsWithPunctuation(e.key)) {
        const text = typed + e.key;
        sendFinal(text, marker && !startsWithPunctuation(text));
        resetWord(false, defaultMarker);
      } else {
        setTyped(typed + e.key);
      }
    }
  }

  return (
    <div className={styles.typeWrap}>
      <div className={styles.typeBox}>
        {effective ? (
          <span data-testid="space-marker" className={styles.marker}>
            {"␣"}
          </span>
        ) : null}
        <input
          data-testid="type-input"
          aria-label="Type your response"
          className={styles.input}
          value={typed}
          disabled={!ask}
          autoFocus
          onKeyDown={onKeyDown}
          onChange={(e) => setTyped(e.target.value)}
        />
      </div>
      {ask ? null : <p className={styles.hint}>Waiting for a request.</p>}
      <p className={styles.keys}>
        Space or punctuation commits a word. Enter commits and starts a new line. {sendKey} sends. Alt+1 to Alt+8 picks a
        row. Esc or Backspace on an empty box rewinds.
      </p>
    </div>
  );
}
