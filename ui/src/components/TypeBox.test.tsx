import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TypeBox, { type TypeBoxProps } from "./TypeBox";
import type { Ask, AskCandidate, ClientMessage } from "../protocol";

const cands: AskCandidate[] = [
  { id: 11, display: "␣the", p_you: 0.4, p_base: null, cls: "word_start" },
  { id: 12, display: "␣a", p_you: 0.3, p_base: null, cls: "word_start" },
  { id: 13, display: "␣cat", p_you: 0.1, p_base: null, cls: "word_start" },
];

function makeAsk(over: Partial<Ask> = {}): Ask {
  return {
    type: "ask",
    request_id: 7,
    pos_id: 3,
    idx: 5,
    committed_text: "",
    provisional: [],
    space_default: true,
    candidates: cands,
    other_mass: 0.05,
    explore: false,
    tokens_remaining: null,
    ...over,
  };
}

function setup(over: Partial<TypeBoxProps> = {}, ask: Ask | null = makeAsk()) {
  const props: TypeBoxProps = {
    ask,
    candidates: cands,
    sendKey: "Ctrl+Enter",
    onSend: vi.fn(),
    onTypedChange: vi.fn(),
    onHighlight: vi.fn(),
    highlight: 0,
    ...over,
  };
  const user = userEvent.setup();
  render(<TypeBox {...props} />);
  return { user, props, input: screen.getByTestId("type-input") as HTMLInputElement };
}

const sent = (fn: ReturnType<typeof vi.fn>): ClientMessage[] => fn.mock.calls.map((c) => c[0] as ClientMessage);

describe("TypeBox", () => {
  it("test_leading_space_toggle: marker follows space_default and backspace over it sends nothing", async () => {
    const { user, props } = setup();
    expect(screen.getByTestId("space-marker")).toBeInTheDocument();
    await user.click(screen.getByTestId("type-input"));
    await user.keyboard("{Backspace}");
    expect(screen.queryByTestId("space-marker")).toBeNull();
    expect(props.onSend).not.toHaveBeenCalled();
    await user.keyboard("hi{Enter}");
    expect(sent(props.onSend as ReturnType<typeof vi.fn>)).toEqual([
      { type: "typed", request_id: 7, pos_id: 3, text: "hi\n", leading_space: false, final: true },
    ]);
  });

  it("test_space_commits_word: space sends the word without the space and the next word leads with one", async () => {
    const { user, props } = setup({}, makeAsk({ space_default: false }));
    await user.click(screen.getByTestId("type-input"));
    await user.keyboard(" ");
    expect(props.onSend).not.toHaveBeenCalled();
    await user.keyboard("sounds ");
    expect(sent(props.onSend as ReturnType<typeof vi.fn>)).toEqual([
      { type: "typed", request_id: 7, pos_id: 3, text: "sounds", leading_space: true, final: true },
    ]);
    expect((screen.getByTestId("type-input") as HTMLInputElement).value).toBe("");
    await user.keyboard("hi{Enter}");
    expect(sent(props.onSend as ReturnType<typeof vi.fn>)[1]).toEqual({
      type: "typed",
      request_id: 7,
      pos_id: 3,
      text: "hi\n",
      leading_space: true,
      final: true,
    });
  });

  it("punctuation is included in the committed text", async () => {
    const { user, props } = setup();
    await user.click(screen.getByTestId("type-input"));
    await user.keyboard("sounds!");
    expect(sent(props.onSend as ReturnType<typeof vi.fn>)[0]).toMatchObject({ text: "sounds!", leading_space: true, final: true });
  });

  it("test_enter_commits_newline", async () => {
    const { user, props, input } = setup();
    await user.click(input);
    await user.keyboard("{Enter}");
    expect(sent(props.onSend as ReturnType<typeof vi.fn>)).toEqual([
      { type: "typed", request_id: 7, pos_id: 3, text: "\n", leading_space: false, final: true },
    ]);
    await user.keyboard("ok{Enter}");
    expect(sent(props.onSend as ReturnType<typeof vi.fn>)[1]).toMatchObject({ text: "ok\n", final: true });
    expect(props.onSend).toHaveBeenCalledTimes(2);
  });

  it("test_alt_digit_picks: Alt+2 picks row 2 and a plain digit goes into the box", async () => {
    const { user, props, input } = setup();
    await user.click(input);
    await user.keyboard("{Alt>}2{/Alt}");
    expect(sent(props.onSend as ReturnType<typeof vi.fn>)).toEqual([{ type: "pick", request_id: 7, pos_id: 3, token_id: 12 }]);
    expect(input.value).toBe("");
    await user.keyboard("2");
    expect(input.value).toBe("2");
    expect(props.onSend).toHaveBeenCalledTimes(1);
  });

  it("alt digit with no candidates does nothing", async () => {
    const { user, props, input } = setup({ candidates: [] });
    await user.click(input);
    await user.keyboard("{Alt>}1{/Alt}");
    expect(props.onSend).not.toHaveBeenCalled();
  });

  it("test_arrow_tab_picks", async () => {
    const { user, props, input } = setup({ highlight: 0 });
    await user.click(input);
    await user.keyboard("{ArrowDown}");
    expect(props.onHighlight).toHaveBeenLastCalledWith(1);
    await user.keyboard("{ArrowUp}");
    expect(props.onHighlight).toHaveBeenLastCalledWith(2);
    await user.keyboard("{Tab}");
    expect(sent(props.onSend as ReturnType<typeof vi.fn>)).toEqual([{ type: "pick", request_id: 7, pos_id: 3, token_id: 11 }]);
  });

  it("test_arrow_tab_picks: Tab picks the highlighted row", async () => {
    const { user, props, input } = setup({ highlight: 1 });
    await user.click(input);
    await user.keyboard("{Tab}");
    expect(sent(props.onSend as ReturnType<typeof vi.fn>)).toEqual([{ type: "pick", request_id: 7, pos_id: 3, token_id: 12 }]);
  });

  it("test_send_key: the configured key sends, Enter alone does not", async () => {
    const { user, props, input } = setup();
    await user.click(input);
    await user.keyboard("hi{Enter}");
    expect(sent(props.onSend as ReturnType<typeof vi.fn>).map((m) => m.type)).toEqual(["typed"]);
    await user.keyboard("yo{Control>}{Enter}{/Control}");
    expect(sent(props.onSend as ReturnType<typeof vi.fn>).map((m) => m.type)).toEqual(["typed", "typed", "send"]);
    expect(sent(props.onSend as ReturnType<typeof vi.fn>)[2]).toEqual({ type: "send", request_id: 7 });
  });

  it("test_backspace_rewinds_last: first backspace hides the marker, the next rewinds to idx - 1", async () => {
    const { user, props, input } = setup({}, makeAsk({ idx: 5, provisional: [{ idx: 4, token_id: 9, display: "x", confidence: 0.9 }] }));
    await user.click(input);
    await user.keyboard("{Backspace}");
    expect(props.onSend).not.toHaveBeenCalled();
    await user.keyboard("{Backspace}");
    expect(sent(props.onSend as ReturnType<typeof vi.fn>)).toEqual([{ type: "rewind", request_id: 7, to_idx: 4 }]);
  });

  it("backspace with no provisional tokens sends nothing", async () => {
    const { user, props, input } = setup({}, makeAsk({ space_default: false }));
    await user.click(input);
    await user.keyboard("{Backspace}");
    expect(props.onSend).not.toHaveBeenCalled();
  });

  it("test_esc_rewinds_all: Esc rewinds to the flushed index", async () => {
    const prov = [0, 1, 2].map((i) => ({ idx: 2 + i, token_id: i, display: "x", confidence: 0.9 }));
    const { user, props, input } = setup({}, makeAsk({ idx: 5, provisional: prov }));
    await user.click(input);
    await user.keyboard("{Escape}");
    expect(sent(props.onSend as ReturnType<typeof vi.fn>)).toEqual([{ type: "rewind", request_id: 7, to_idx: 2 }]);
  });

  it("disabled without an ask", () => {
    setup({}, null);
    expect(screen.getByTestId("type-input")).toBeDisabled();
    expect(screen.getByText(/Waiting for a request/)).toBeInTheDocument();
  });
});
