import { useState } from "react";
import type { NoteInfo } from "../protocol";
import styles from "./components.module.css";

export interface TeachPanelProps {
  notes: NoteInfo[];
  onTeach: (text: string) => void;
  onForget: (chunkId: number) => void;
}

export default function TeachPanel({ notes, onTeach, onForget }: TeachPanelProps) {
  const [text, setText] = useState("");

  function teach(): void {
    const t = text.trim();
    if (t === "") return;
    onTeach(t);
    setText("");
  }

  return (
    <section className={styles.teach}>
      <h2>Teach it this</h2>
      <textarea
        data-testid="teach-input"
        aria-label="Teach it this"
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={3}
      />
      <button type="button" data-testid="teach-button" onClick={teach}>
        Teach
      </button>
      <ul className={styles.notes}>
        {notes.map((n) => (
          <li key={n.id} data-testid="note-item">
            <span>{n.preview}</span>
            <button type="button" data-testid="forget-button" onClick={() => onForget(n.id)}>
              Forget
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
