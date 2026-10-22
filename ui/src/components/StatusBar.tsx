import type { Status } from "../protocol";
import styles from "./components.module.css";

export interface StatusBarProps {
  status: Status | null;
  connected: boolean;
  onAutonomy: (delta: number) => void;
}

function metric(status: Status | null, name: string, suffix = ""): string {
  const v = status?.metrics[name];
  return typeof v === "number" ? `${v.toFixed(1)}${suffix}` : "-";
}

export default function StatusBar({ status, connected, onAutonomy }: StatusBarProps) {
  const autonomy = status?.autonomy ?? 0;
  return (
    <header className={styles.statusBar}>
      <span className={connected ? styles.ok : styles.bad} data-testid="connection">
        {connected ? "connected" : "disconnected"}
      </span>
      <span>backend {status === null ? "unknown" : status.backend_ok ? "ok" : "down"}</span>
      <span>scorer v{status?.scorer_version ?? "-"}</span>
      <span>auto {metric(status, "auto_select_rate", "%")}</span>
      <span>asks/100 {metric(status, "asks_per_100")}</span>
      <span>rewinds {metric(status, "rewind_rate", "%")}</span>
      <label className={styles.slider}>
        autonomy {autonomy.toFixed(1)}
        <input
          type="range"
          data-testid="autonomy-slider"
          min={-2}
          max={2}
          step={0.1}
          value={autonomy}
          onChange={(e) => onAutonomy(Number(e.target.value))}
        />
      </label>
    </header>
  );
}
