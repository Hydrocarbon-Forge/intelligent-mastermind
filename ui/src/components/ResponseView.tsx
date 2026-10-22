import type { AutoToken } from "../protocol";
import styles from "./components.module.css";

export interface ResponseViewProps {
  committedText: string;
  provisional: AutoToken[];
}

export default function ResponseView({ committedText, provisional }: ResponseViewProps) {
  return (
    <div className={styles.response}>
      <span data-testid="committed-text">{committedText}</span>
      {provisional.map((t) => (
        <span key={t.idx} data-testid="provisional-token" className={styles.provisional} title={`${t.confidence.toFixed(2)}`}>
          {t.display}
        </span>
      ))}
    </div>
  );
}
