import type { Ask, AskCandidate } from "./protocol";

export const SPACE_MARK = "␣"; // ␣

const CLOSING_PUNCTUATION = new Set([",", ".", "!", "?", ":", ";", ")", "]", "}", "'"]);

// Candidates whose display starts with the typed prefix, in order, at most maxRows.
// The prefix carries the space marker when the word has a leading space.
export function filterCandidates(
  cands: AskCandidate[],
  typed: string,
  leadingSpace: boolean,
  maxRows: number,
): AskCandidate[] {
  if (typed === "") {
    return cands.slice(0, maxRows);
  }
  const prefix = (leadingSpace ? SPACE_MARK : "") + typed;
  return cands.filter((c) => c.display.startsWith(prefix)).slice(0, maxRows);
}

export function isClosingPunctuation(ch: string): boolean {
  return CLOSING_PUNCTUATION.has(ch);
}

// True when the first character is an ASCII punctuation character.
export function startsWithPunctuation(text: string): boolean {
  return /^[!-/:-@[-`{-~]/.test(text);
}

export function flushedIdx(ask: Ask): number {
  return ask.idx - ask.provisional.length;
}
