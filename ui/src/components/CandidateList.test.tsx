import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import CandidateList from "./CandidateList";
import type { Ask, AskCandidate } from "../protocol";

const SPACE = "␣";

const cands: AskCandidate[] = [
  { id: 1, display: `${SPACE}the`, p_you: 0.4, p_base: null, cls: "word_start" },
  { id: 2, display: `${SPACE}there`, p_you: 0.2, p_base: null, cls: "word_start" },
  { id: 3, display: "the", p_you: 0.1, p_base: null, cls: "continuation" },
  { id: 4, display: `${SPACE}cat`, p_you: 0.05, p_base: null, cls: "word_start" },
  { id: 5, display: `${SPACE}dog`, p_you: 0.05, p_base: null, cls: "word_start" },
  { id: 6, display: `${SPACE}then`, p_you: 0.03, p_base: null, cls: "word_start" },
  { id: 7, display: `${SPACE}a`, p_you: 0.02, p_base: null, cls: "word_start" },
  { id: 8, display: `${SPACE}an`, p_you: 0.02, p_base: null, cls: "word_start" },
  { id: 9, display: `${SPACE}it`, p_you: 0.01, p_base: null, cls: "word_start" },
];

const ask: Ask = {
  type: "ask",
  request_id: 1,
  pos_id: 1,
  idx: 0,
  committed_text: "",
  provisional: [],
  space_default: true,
  candidates: cands,
  other_mass: 0.12,
  explore: false,
  tokens_remaining: null,
};

describe("CandidateList", () => {
  it("test_candidate_list_rows", () => {
    render(
      <CandidateList ask={ask} typed="" leadingSpace={true} maxRows={8} highlight={0} onPick={() => {}} />,
    );
    const rows = screen.getAllByTestId("candidate-row");
    expect(rows).toHaveLength(8);
    expect(rows[0]).toHaveTextContent(SPACE + "the");
    expect(within(rows[0]).getByTestId("p-you-bar")).toHaveStyle({ width: "40%" });
    expect(within(rows[0]).getByText("40%")).toBeInTheDocument();
    expect(screen.getByTestId("other-mass")).toHaveTextContent("12%");
  });

  it("test_type_filters_candidates", () => {
    const { rerender } = render(
      <CandidateList ask={ask} typed="" leadingSpace={true} maxRows={8} highlight={0} onPick={() => {}} />,
    );
    rerender(
      <CandidateList ask={ask} typed="the" leadingSpace={true} maxRows={8} highlight={0} onPick={() => {}} />,
    );
    const rows = screen.getAllByTestId("candidate-row");
    expect(rows.map((r) => r.textContent)).toEqual([expect.stringContaining("the"), expect.stringContaining("there"), expect.stringContaining("then")]);
    expect(rows).toHaveLength(3);
  });

  it("clicking a row picks it", () => {
    const onPick = vi.fn();
    render(<CandidateList ask={ask} typed="" leadingSpace={true} maxRows={8} highlight={0} onPick={onPick} />);
    fireEvent.click(screen.getAllByTestId("candidate-row")[1]);
    expect(onPick).toHaveBeenCalledWith(2);
  });
});
