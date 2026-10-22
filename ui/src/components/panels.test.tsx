import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ResponseView from "./ResponseView";
import TeachPanel from "./TeachPanel";
import StatusBar from "./StatusBar";
import RequestList from "./RequestList";
import ConversationView from "./ConversationView";
import type { Status } from "../protocol";
import type { RequestView } from "../store";

const status: Status = {
  type: "status",
  request_id: null,
  backend_ok: true,
  capabilities: {},
  scorer_version: 3,
  autonomy: 0,
  metrics: { auto_select_rate: 42, asks_per_100: 58, rewind_rate: 1.5 },
  knowledge_count: 2,
  notes: [{ id: 9, source: "teach", preview: "the sky is blue" }],
};

describe("panels", () => {
  it("test_provisional_display: provisional tokens render dimmed and separate from committed text", () => {
    render(
      <ResponseView
        committedText="Hello"
        provisional={[
          { idx: 1, token_id: 1, display: "␣there", confidence: 0.9 },
          { idx: 2, token_id: 2, display: "!", confidence: 0.8 },
        ]}
      />,
    );
    expect(screen.getByTestId("committed-text")).toHaveTextContent("Hello");
    const prov = screen.getAllByTestId("provisional-token");
    expect(prov).toHaveLength(2);
    expect(prov[0]).toHaveTextContent("␣there");
    expect(prov[0].className).not.toBe("");
    expect(screen.getByTestId("committed-text").className).toBe("");
  });

  it("test_teach_and_forget", async () => {
    const onTeach = vi.fn();
    const onForget = vi.fn();
    const user = userEvent.setup();
    render(<TeachPanel notes={status.notes} onTeach={onTeach} onForget={onForget} />);
    await user.type(screen.getByTestId("teach-input"), "remember this");
    await user.click(screen.getByTestId("teach-button"));
    expect(onTeach).toHaveBeenCalledWith("remember this");
    expect(screen.getByTestId("teach-input")).toHaveValue("");
    await user.click(screen.getByTestId("forget-button"));
    expect(onForget).toHaveBeenCalledWith(9);
  });

  it("test_slider_sends_autonomy", () => {
    const onAutonomy = vi.fn();
    render(<StatusBar status={status} connected={true} onAutonomy={onAutonomy} />);
    const slider = screen.getByTestId("autonomy-slider");
    expect(slider).toHaveAttribute("min", "-2");
    expect(slider).toHaveAttribute("max", "2");
    fireEvent.change(slider, { target: { value: "1.5" } });
    expect(onAutonomy).toHaveBeenCalledWith(1.5);
    expect(screen.getByTestId("connection")).toHaveTextContent("connected");
  });

  it("request list shows age and declines", async () => {
    const onDecline = vi.fn();
    const requests: RequestView[] = [
      { id: 4, messages: [], params: {}, tools: null, ageS: 0, receivedAt: 1000 },
    ];
    const user = userEvent.setup();
    render(<RequestList requests={requests} activeId={4} now={1000 + 75_000} onDecline={onDecline} />);
    const item = screen.getByTestId("request-item");
    expect(item).toHaveTextContent("1m 15s");
    await user.click(within(item).getByTestId("decline-button"));
    expect(onDecline).toHaveBeenCalledWith(4);
  });

  it("conversation shows text parts, [image] for others and truncates long text", () => {
    const long = "x".repeat(500);
    const request: RequestView = {
      id: 1,
      messages: [
        { role: "user", content: [{ type: "text", text: "look" }, { type: "image_url", image_url: {} }] },
        { role: "system", content: long },
      ],
      params: {},
      tools: [{ name: "search", description: "find things" }],
      ageS: 0,
      receivedAt: 0,
    };
    render(<ConversationView request={request} />);
    expect(screen.getByText(/look \[image\]/)).toBeInTheDocument();
    expect(screen.getByText(/x{300}\.\.\./)).toBeInTheDocument();
    expect(screen.getByText("search")).toBeInTheDocument();
  });
});
