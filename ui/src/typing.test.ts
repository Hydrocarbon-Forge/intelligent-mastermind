import { describe, expect, it } from "vitest";
import {
  SPACE_MARK,
  filterCandidates,
  flushedIdx,
  isClosingPunctuation,
  startsWithPunctuation,
} from "./typing";
import type { Ask, AskCandidate } from "./protocol";

const cand = (id: number, display: string): AskCandidate => ({
  id,
  display,
  p_you: 0.5,
  p_base: null,
  cls: "word_start",
});

const cands: AskCandidate[] = [
  cand(1, `${SPACE_MARK}the`),
  cand(2, `${SPACE_MARK}there`),
  cand(3, "the"),
  cand(4, `${SPACE_MARK}cat`),
  cand(5, `${SPACE_MARK}The`),
  cand(6, `${SPACE_MARK}then`),
];

describe("filterCandidates", () => {
  it("empty typed returns the first maxRows candidates unchanged", () => {
    expect(filterCandidates(cands, "", true, 3)).toEqual(cands.slice(0, 3));
    expect(filterCandidates(cands, "", false, 10)).toEqual(cands);
  });

  it("filters by prefix with the space marker when leadingSpace is true", () => {
    const out = filterCandidates(cands, "the", true, 10);
    expect(out.map((c) => c.id)).toEqual([1, 2, 6]);
  });

  it("without leading space matches only unmarked displays", () => {
    expect(filterCandidates(cands, "the", false, 10).map((c) => c.id)).toEqual([3]);
  });

  it("is case-sensitive", () => {
    expect(filterCandidates(cands, "Th", true, 10).map((c) => c.id)).toEqual([5]);
  });

  it("slices to maxRows after filtering", () => {
    expect(filterCandidates(cands, "the", true, 2).map((c) => c.id)).toEqual([1, 2]);
  });
});

describe("punctuation", () => {
  it("isClosingPunctuation matches the closing set and apostrophe", () => {
    for (const ch of [",", ".", "!", "?", ":", ";", ")", "]", "}", "'"]) {
      expect(isClosingPunctuation(ch)).toBe(true);
    }
    for (const ch of ["(", "a", " ", "-", "\""]) {
      expect(isClosingPunctuation(ch)).toBe(false);
    }
  });

  it("startsWithPunctuation checks the first ASCII punctuation character", () => {
    expect(startsWithPunctuation(".x")).toBe(true);
    expect(startsWithPunctuation("(")).toBe(true);
    expect(startsWithPunctuation("[a]")).toBe(true);
    expect(startsWithPunctuation("~")).toBe(true);
    expect(startsWithPunctuation("abc.")).toBe(false);
    expect(startsWithPunctuation(" .")).toBe(false);
    expect(startsWithPunctuation("")).toBe(false);
  });
});

describe("flushedIdx", () => {
  it("is idx minus provisional length", () => {
    const ask = {
      idx: 10,
      provisional: [{ idx: 7, token_id: 1, display: "a", confidence: 1 }, { idx: 8, token_id: 2, display: "b", confidence: 1 }, { idx: 9, token_id: 3, display: "c", confidence: 1 }],
    } as unknown as Ask;
    expect(flushedIdx(ask)).toBe(7);
  });
});
