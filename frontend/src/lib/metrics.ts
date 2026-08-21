export interface SlotInfo {
  id: number;
  is_processing: boolean;
  n_ctx: number;
  task_id: number;
  /** Prompt length this slot is holding. */
  n_prompt_tokens: number;
  /** How much of it this request had to actually run through the model… */
  n_prompt_tokens_processed: number;
  /** …and how much came straight back from the KV cache. */
  n_prompt_tokens_cache: number;
  n_decoded: number;
  n_remain: number;
  n_predict: number;
  has_next_token: boolean;
  temperature: number | null;
  top_k: number | null;
  top_p: number | null;
  max_tokens: number | null;
  speculative: boolean;
}

export interface JobRecord {
  slot_id: number;
  task_id: number;
  started_at: number;
  ended_at: number | null;
  last_seen_at: number;
  duration_s: number;
  tokens_decoded: number;
  n_predict: number;
  avg_decode_tps: number;
  active: boolean;
}

export interface MetricsFrame {
  ts: number;
  preset: string;
  port: number;
  slots: SlotInfo[];
  prom: Record<string, number>;
  instant_prompt_tps: number | null;
  instant_decode_tps: number | null;
  lifetime_prompt_tps: number | null;
  lifetime_decode_tps: number | null;
  requests_processing: number;
  requests_deferred: number;
  kv_cache_used_tokens: number;
  kv_cache_max_tokens: number;
  busy_slots: number;
  total_slots: number;
  error: string | null;
  loaded_model_id: string | null; // router-mode: id of the model currently loaded
  active_jobs: JobRecord[];
  recent_jobs: JobRecord[];
}

export interface MetricsSnapshot {
  presets: Record<string, { latest: MetricsFrame | null; history: MetricsFrame[] }>;
}

export async function fetchSnapshot(historyN = 120): Promise<MetricsSnapshot> {
  const r = await fetch(`/api/metrics/snapshot?history_n=${historyN}`);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export function openMetricsStream(preset: string, onFrame: (f: MetricsFrame) => void): EventSource {
  const es = new EventSource(`/api/metrics/stream/${encodeURIComponent(preset)}`);
  es.onmessage = (ev) => {
    try {
      onFrame(JSON.parse(ev.data));
    } catch { /* ignore */ }
  };
  return es;
}

export function sparklinePath(values: number[], width: number, height: number, max?: number): string {
  if (values.length === 0) return '';
  const m = max ?? Math.max(1, ...values);
  const step = values.length > 1 ? width / (values.length - 1) : 0;
  let d = '';
  for (let i = 0; i < values.length; i++) {
    const x = i * step;
    const y = height - (values[i] / m) * height;
    d += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1) + ' ';
  }
  return d.trim();
}
