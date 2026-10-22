import type { RequestView } from "../store";
import styles from "./components.module.css";

export interface RequestListProps {
  requests: RequestView[];
  activeId: number | null;
  onDecline: (requestId: number) => void;
  now: number;
}

export function formatAge(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

export default function RequestList({ requests, activeId, onDecline, now }: RequestListProps) {
  if (requests.length === 0) return <p className={styles.empty}>No requests queued.</p>;
  return (
    <ul className={styles.requests}>
      {requests.map((r) => (
        <li
          key={r.id}
          data-testid="request-item"
          className={r.id === activeId ? `${styles.requestItem} ${styles.requestActive}` : styles.requestItem}
        >
          <span>Request {r.id}</span>
          <span className={styles.age}>{formatAge((now - r.receivedAt) / 1000)}</span>
          <button type="button" data-testid="decline-button" onClick={() => onDecline(r.id)}>
            Decline
          </button>
        </li>
      ))}
    </ul>
  );
}
