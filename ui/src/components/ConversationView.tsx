import type { RequestView } from "../store";
import styles from "./components.module.css";

export interface ConversationViewProps {
  request: RequestView | null;
}

const MAX_CHARS = 300;

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

// Text parts show their text; every other part shows as [image].
function partText(part: unknown): string {
  if (typeof part === "string") return part;
  if (isRecord(part) && part.type === "text" && typeof part.text === "string") return part.text;
  return "[image]";
}

export function contentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map(partText).join(" ");
  if (content === null || content === undefined) return "";
  return "[image]";
}

function toolName(tool: Record<string, unknown>): string {
  if (typeof tool.name === "string") return tool.name;
  if (isRecord(tool.function) && typeof tool.function.name === "string") return tool.function.name;
  return "(unnamed)";
}

function toolDescription(tool: Record<string, unknown>): string {
  if (typeof tool.description === "string") return tool.description;
  if (isRecord(tool.function) && typeof tool.function.description === "string") return tool.function.description;
  return "";
}

export default function ConversationView({ request }: ConversationViewProps) {
  if (!request) return <p className={styles.empty}>No active request.</p>;
  return (
    <div className={styles.conversation}>
      {request.messages.map((m, i) => {
        const full = contentText(m.content);
        const shown = full.length > MAX_CHARS ? `${full.slice(0, MAX_CHARS)}...` : full;
        return (
          <div key={i} className={styles.msg}>
            <div className={styles.role}>{typeof m.role === "string" ? m.role : "message"}</div>
            <div className={styles.msgText} title={full}>
              {shown}
            </div>
          </div>
        );
      })}
      {request.tools && request.tools.length > 0 ? (
        <div className={styles.tools}>
          <div className={styles.role}>Tools</div>
          <ul>
            {request.tools.map((t, i) => (
              <li key={i}>
                <strong>{toolName(t)}</strong> {toolDescription(t)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
