import type { Ask } from "../protocol";
import { filterCandidates } from "../typing";
import styles from "./components.module.css";

export interface CandidateListProps {
  ask: Ask;
  typed: string;
  leadingSpace: boolean;
  maxRows: number;
  highlight: number;
  onPick: (tokenId: number) => void;
}

export function pct(p: number): string {
  return `${Math.round(p * 100)}%`;
}

export default function CandidateList({ ask, typed, leadingSpace, maxRows, highlight, onPick }: CandidateListProps) {
  const rows = filterCandidates(ask.candidates, typed, leadingSpace, maxRows);
  return (
    <div className={styles.candidates}>
      {rows.length === 0 ? <p className={styles.empty}>No candidates</p> : null}
      <ol className={styles.rows}>
        {rows.map((c, i) => {
          const width = `${Math.min(Math.max(c.p_you, 0), 1) * 100}%`;
          return (
            <li
              key={c.id}
              data-testid="candidate-row"
              className={i === highlight ? `${styles.row} ${styles.rowActive}` : styles.row}
              onClick={() => onPick(c.id)}
            >
              <span className={styles.index}>{i + 1}</span>
              <span className={styles.display}>{c.display}</span>
              <span className={styles.barTrack}>
                <span data-testid="p-you-bar" className={styles.bar} style={{ width }} />
              </span>
              <span className={styles.pct}>{pct(c.p_you)}</span>
            </li>
          );
        })}
      </ol>
      <div data-testid="other-mass" className={styles.other}>
        other {pct(ask.other_mass)}
      </div>
    </div>
  );
}
