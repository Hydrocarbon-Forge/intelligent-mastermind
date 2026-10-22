import { useCallback, useEffect, useRef, useState } from "react";
import type { ClientMessage } from "./protocol";
import { ReconnectingSocket, defaultSocketUrl } from "./socket";
import { useStore } from "./store";
import { autonomyMsg, declineMsg, forgetMsg, teachMsg } from "./messages";
import { filterCandidates } from "./typing";
import CandidateList from "./components/CandidateList";
import ConversationView from "./components/ConversationView";
import RequestList from "./components/RequestList";
import ResponseView from "./components/ResponseView";
import StatusBar from "./components/StatusBar";
import TeachPanel from "./components/TeachPanel";
import TypeBox from "./components/TypeBox";
import styles from "./App.module.css";

const MAX_ROWS = 8;

export default function App() {
  const socketRef = useRef<ReconnectingSocket | null>(null);
  const requests = useStore((s) => s.requests);
  const activeId = useStore((s) => s.activeId);
  const ask = useStore((s) => s.ask);
  const provisional = useStore((s) => s.provisional);
  const committedText = useStore((s) => s.committedText);
  const status = useStore((s) => s.status);
  const connected = useStore((s) => s.connected);

  const [typed, setTyped] = useState("");
  const [leadingSpace, setLeadingSpace] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const sock = new ReconnectingSocket({
      url: defaultSocketUrl(),
      onMessage: (m) => useStore.getState().applyMessage(m),
      onConnection: (c) => useStore.getState().setConnected(c),
    });
    socketRef.current = sock;
    sock.start();
    return () => {
      sock.stop();
      socketRef.current = null;
    };
  }, []);

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  const onSend = useCallback((msg: ClientMessage) => {
    socketRef.current?.send(msg);
  }, []);

  const onTypedChange = useCallback((text: string, lead: boolean) => {
    setTyped(text);
    setLeadingSpace(lead);
    setHighlight(0);
  }, []);

  const request = requests.find((r) => r.id === activeId) ?? null;
  const candidates = ask ? filterCandidates(ask.candidates, typed, leadingSpace, MAX_ROWS) : [];
  const shownHighlight = Math.min(highlight, Math.max(0, candidates.length - 1));

  return (
    <div className={styles.app}>
      <StatusBar
        status={status}
        connected={connected}
        onAutonomy={(delta) => onSend(autonomyMsg(delta))}
      />
      <div className={styles.columns}>
        <aside className={styles.side}>
          <h2>Requests</h2>
          <RequestList
            requests={requests}
            activeId={activeId}
            now={now}
            onDecline={(id) => onSend(declineMsg(id, "declined by user"))}
          />
        </aside>
        <main className={styles.main}>
          <ConversationView request={request} />
          <ResponseView committedText={committedText} provisional={provisional} />
          {ask ? (
            <CandidateList
              ask={ask}
              typed={typed}
              leadingSpace={leadingSpace}
              maxRows={MAX_ROWS}
              highlight={shownHighlight}
              onPick={(tokenId) => onSend({ type: "pick", request_id: ask.request_id, pos_id: ask.pos_id, token_id: tokenId })}
            />
          ) : null}
          <TypeBox
            ask={ask}
            candidates={candidates}
            sendKey="Ctrl+Enter"
            onSend={onSend}
            onTypedChange={onTypedChange}
            onHighlight={setHighlight}
            highlight={shownHighlight}
          />
        </main>
        <aside className={styles.side}>
          <TeachPanel
            notes={status?.notes ?? []}
            onTeach={(text) => onSend(teachMsg(text))}
            onForget={(id) => onSend(forgetMsg(id))}
          />
        </aside>
      </div>
    </div>
  );
}
