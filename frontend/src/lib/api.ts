import { t } from './i18n.svelte';
export interface Health {
  status: string;
  version: string;
}

export interface Settings {
  controller_bind_host: string;
  controller_bind_port: number;
  mcp_bind_host: string;
  mcp_bind_port: number;
  lan_token: string | null;
  llama_repo: string;
  llama_bin: string;
  llama_server_url: string;
  /** Extra llama.cpp backends reachable over RPC. Empty on most machines. */
  rpc_servers?: RpcServerConfig[];
  scan_roots: string[];
  hf_models_root: string;
  hf_token: string | null;
  anthropic_api_key: string | null;
  llm_provider?: 'claude' | 'openai';
  llm_base_url?: string;
  llm_api_key?: string | null;
  llm_model?: string;
  ui_language?: 'en' | 'tr';
  gateway: { enabled: boolean };
}

export interface SummaryAuthStatus {
  provider: 'claude' | 'openai';
  mode: 'api_key' | 'env' | 'claude_cli' | 'profile' | 'openai' | 'none';
  model: string | null;
  base_url: string | null;
  detail: string | null;
}

export interface LlmEndpoint {
  base_url: string;
  reachable: boolean;
  models: string[];
  n_ctx: number | null;
  native: boolean;
  resolved: string | null;
  detail: string | null;
}

export interface PathProbe { exists: boolean; kind: 'file' | 'dir' | null; ok: boolean }
export interface SettingsCheck {
  llama_bin: PathProbe;
  llama_repo: PathProbe;
  hf_models_root: PathProbe;
  scan_roots: PathProbe[];
}

// Error carrying the HTTP status and the parsed FastAPI `detail`. The detail
// may be a string or a structured object (e.g. the fit-check block from
// /api/server/start), so callers can branch on it instead of scraping .message.
export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

export interface FitBlockDetail {
  code: 'fit_block';
  level: string;
  headline: string;
  messages: { severity: string; text: string }[];
}

/** Narrow an unknown error to the structured fit-check block, if it is one. */
export function fitBlock(e: unknown): FitBlockDetail | null {
  if (e instanceof ApiError && e.detail && typeof e.detail === 'object'
      && (e.detail as { code?: string }).code === 'fit_block') {
    return e.detail as FitBlockDetail;
  }
  return null;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) } });
  } catch {
    // fetch() rejects with an opaque "Failed to fetch" on network-level
    // failures (backend down / restarting). Give users a real sentence.
    throw new Error(t('Cannot reach the backend — it may be restarting or stopped.'));
  }
  if (!resp.ok) {
    // Surface the {detail} from the FastAPI error body — otherwise the user
    // only sees "400 Bad Request". detail may be a string or a structured
    // object; keep the object on the thrown ApiError for callers to inspect.
    let detailRaw: unknown;
    let msg = '';
    try {
      const body = await resp.json();
      detailRaw = body?.detail;
      if (typeof detailRaw === 'string') msg = `: ${detailRaw}`;
      else if (detailRaw && typeof detailRaw === 'object') {
        const h = (detailRaw as { headline?: string }).headline;
        if (h) msg = `: ${h}`;
      }
    } catch { /* if the body isn't JSON the status code is enough */ }
    throw new ApiError(`${resp.status} ${resp.statusText}${msg}`, resp.status, detailRaw);
  }
  // The SPA fallback can return 200 + HTML for unknown /api paths (old backend
  // + new UI window). Give a clear message instead of a raw parse error.
  const text = await resp.text();
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error(
      t('The backend does not know this feature yet — restart the service with the updated code.')
    );
  }
}

export interface LlamaConfig {
  name: string;
  model_path: string | null;
  hf_repo: string | null;
  hf_file: string | null;
  mmproj_path: string | null;
  host: string;
  port: number;
  api_key: string | null;
  ctx_size: number;
  n_gpu_layers: number;
  parallel: number;
  batch_size: number;
  ubatch_size: number;
  threads: number;
  flash_attn: string;
  cache_type_k: string;
  cache_type_v: string;
  cont_batching: boolean;
  /** Prompt-processing / KV-cache reuse knobs. null = llama.cpp's default
   *  (emit nothing); setting them turns prompt-prefix reuse on/off. */
  cache_reuse?: number | null;
  cache_idle_slots?: boolean | null;
  context_shift?: boolean | null;
  kv_offload?: boolean | null;
  temperature: number;
  top_k: number;
  top_p: number;
  min_p: number;
  repeat_penalty: number;
  /** Opt-in: null means "do not emit --presence-penalty", so the model's own
   *  default applies (Qwen3.6 family = 1.5). A vision/OCR preset pins 0. */
  presence_penalty?: number | null;
  jinja: boolean;
  metrics: boolean;
  slots: boolean;
  spec_type?: 'none' | 'draft-mtp' | 'draft-simple' | 'ngram-simple' | 'draft-model';
  model_path_draft?: string | null;
  n_gpu_layers_draft?: number;
  draft_max?: number | null;
  draft_min?: number | null;
  /** llama.cpp device ids this preset loads onto ("CUDA0", "Vulkan1"),
   *  emitted as `-dev`. Empty = let llama.cpp choose. Ids are backend-specific
   *  and come from /api/server/devices, never from the OS-level GPU probe. */
  devices?: string[];
  /** `-ts` proportions across the selected devices. */
  tensor_split?: string | null;
  /** Environment for the llama-server process, e.g.
   *  `{GGML_CUDA_DISABLE_GRAPHS: "1"}`. Some llama.cpp behaviour has no flag
   *  and can only be reached this way. Merged over the backend's own
   *  environment at spawn, per preset. */
  env?: Record<string, string>;
  extra_flags: string[];
  /** `--reasoning on|off`. "auto" (the default) emits nothing and lets the
   *  chat template decide — which is what every preset did before this field.
   *  Note that thinking and instruct modes want DIFFERENT sampling; the model
   *  recipes on the Model tab apply both together. */
  reasoning?: 'auto' | 'on' | 'off';
  /** A raw command line that REPLACES everything rendered from the fields
   *  above. When set, this is exactly what runs — see the Command tab. The
   *  fields are re-read from it on save, so the rest of the app still knows
   *  where the process listens and what it loaded. */
  argv_override?: string | null;
  notes: string;
  estimated_vram_mb: number | null;
  ui_hidden?: boolean;
  mode?: 'single' | 'router';
  models_dir?: string | null;
  models_max?: number;
  models_autoload?: boolean;
  models_preset_path?: string | null;
  sleep_idle_seconds?: number | null;
}

/** What `POST /api/presets/command` returns: the exact command a draft
 *  preset would execute, plus the field-rendered one for comparison. */
export interface CommandPreview {
  binary: string;
  source: 'fields' | 'override';
  command: string;
  fields_command: string;
  argv: string[];
  unknown_flags: UnknownFlag[];
  missing_values: MissingFlagValue[];
  shadowed: ShadowedFlag[];
  conflicts: FlagConflict[];
}

export interface UnknownFlag {
  flag: string;
  suggestions: string[];
}

/** A flag that needs a value and was given none. llama-server refuses the
 *  whole command line for this, so the preset never starts. */
export interface MissingFlagValue {
  flag: string;
  placeholder: string;
  help: string;
}

/** A flag the command passes twice with different values. llama.cpp keeps the
 *  last one, so the earlier value — usually a form field — has no effect. */
export interface ShadowedFlag {
  flag: string;
  spellings: string[];
  wins: string | null;
  shadowed: (string | null)[];
}

/** Two different flags whose meanings cancel out — llama-server accepts both
 *  and silently honours one. */
export interface FlagConflict {
  id: string;
  flags: string[];
  message: string;
}

export interface CommandParse {
  config: LlamaConfig;
  diff: { field: string; from: unknown; to: unknown }[];
  warnings: string[];
  unknown_flags: UnknownFlag[];
  missing_values: MissingFlagValue[];
  shadowed: ShadowedFlag[];
  conflicts: FlagConflict[];
}

/** One click in the editor: fields to set, flags to add/drop. */
export interface ModelRecipe {
  id: string;
  label: string;
  why: string;
  set: Record<string, unknown>;
  add_flags: string[];
  remove_flags: string[];
  source: 'builtin' | 'user';
}

export interface ModelProfile {
  model_id: string;
  architecture: string | null;
  family: string | null;
  context_length: number | null;
  capabilities: { thinking: boolean; tools: boolean; vision: boolean };
  detected_by: Record<string, string>;
  sampling: { thinking: Record<string, number>; non_thinking: Record<string, number> };
  sampling_source: string;
  recipes: ModelRecipe[];
  notes: string[];
  overlay_path: string;
  overlay_loaded: boolean;
}

export interface FlagSpec {
  names: string[];
  canonical: string;
  placeholder: string | null;
  takes_value: boolean;
  value_required: boolean;
  choices: string[];
  help: string;
  env: string | null;
  section: string;
}

export interface RouterModel {
  id: string;
  in_cache?: boolean;
  path?: string;
  status?: { value: 'loaded' | 'unloaded' | 'loading' | 'sleeping'; args?: string[]; failed?: boolean; exit_code?: number };
}

/** One key where the INI on disk and the router's parsed table disagree.
 *  `key` is "(model)" when a whole section is missing from the router. */
export interface RouterIniDrift {
  model: string;
  key: string;
  ini: string;
  live: string;
}

export interface RouterModels {
  data: RouterModel[];
  running?: boolean;
  router_preset?: string | null;
  /** The INI the *running* router was launched with. */
  ini_path?: string;
  /** Empty when the router's table matches the file. */
  ini_drift?: RouterIniDrift[];
  /** Set when the drift check itself could not run — never fatal. */
  ini_error?: string;
}

/** An extra_flag that llama-server would refuse for want of a value. In the
 *  INI it becomes `flag = true`, which fails only when a model is loaded. */
export interface RouterFlagWarning {
  preset: string;
  flag: string;
  placeholder: string;
}

export interface RouterIni {
  path: string;
  ini: string;
  flag_warnings?: RouterFlagWarning[];
}

export interface RouterActive {
  running: boolean;
  preset: string | null;
  status: PresetStatus | null;
}

export interface VramEstimate {
  total_mb: number;
  model_mb: number;
  kv_cache_mb: number;
  compute_mb: number;
  source: 'computed' | 'approx' | 'unavailable';
  details: Record<string, number | string>;
  gpu_mb?: number;
  ram_mb?: number;
}

/** One offload target of the configured llama-server binary. */
export interface LlamaDevice {
  id: string;
  /** What the picker shows in the id column. Equals `id` for every real
   *  device; the CPU row sends "CPU" because its id is llama.cpp's "none". */
  label?: string;
  name: string;
  total_mb: number;
  free_mb: number;
  backend: string;
  /** An APU iGPU — listed so the UI can explain it, never a target. */
  integrated: boolean;
  /** A software rasterizer (llvmpipe & co). */
  software: boolean;
  /** Set when this row is the same physical card as another backend's row;
   *  holds the id we prefer (CUDA over Vulkan for the same GPU). */
  duplicate_of: string | null;
  /** RPC rows only: the host:port behind the device. */
  rpc_endpoint?: string | null;
  /** RPC rows only, and a suspicion rather than a fact — a local device of
   *  exactly this size exists, so the two may be one card reached two ways.
   *  The RPC protocol reports no hardware identity, so this warns rather than
   *  excluding: the endpoint might be another machine with an identical card. */
  may_alias?: string | null;
  selectable: boolean;
}

/** Persisted configuration of one RPC offload server. */
export interface RpcServerConfig {
  name: string;
  /** Blank = resolve at start time, so a later rebuild is picked up. */
  binary: string;
  host: string;
  port: number;
  /** Device ids in the RPC server's own binary (e.g. "ROCm0"). Empty exports
   *  everything it can see, including its CPU. */
  devices: string[];
  autostart: boolean;
}

/** Live status of one ggml-rpc-server LlamaDeck manages. */
export interface RpcServer {
  name: string;
  endpoint: string;
  binary: string;
  devices: string[];
  autostart: boolean;
  running: boolean;
  /** Whether LlamaDeck spawned it (vs. finding it already on the port). */
  owned: boolean;
  last_error: string | null;
  log_tail: string[];
}

export interface DeviceReport {
  binary: string;
  devices: LlamaDevice[];
  selectable_ids: string[];
}

export interface PresetStatus {
  name: string;
  running: boolean;
  adopted: boolean;
  pid: number | null;
  port: number;
  started_at: number | null;
  uptime_seconds: number | null;
  rss_mb: number | null;
  cpu_percent: number | null;
  config: LlamaConfig;
  log_file: string | null;
  returncode: number | null;
  last_error: string | null;
  vram_estimate: VramEstimate | null;
}

export interface ScanEntry {
  pid: number;
  cmdline: string[];
  started_at: number;
  config: LlamaConfig;
  suggested_preset: string | null;
}

export interface GpuInfo {
  index: number;
  name: string;
  total_mb: number;
  used_mb: number;
  free_mb: number;
  /** "nvidia" | "amd" | "apple" */
  vendor?: string;
  /** GPU memory is carved out of system RAM (Apple Silicon, AMD APUs). */
  unified?: boolean;
  /** An APU's iGPU. Reports a small carve-out plus the whole GTT aperture,
   *  so its total is mostly system RAM — never a VRAM budget. */
  integrated?: boolean;
  gtt_total_mb?: number;
  gtt_used_mb?: number;
  /** Live sensors. Independently null when the card cannot answer — an iGPU
   *  has no fan, Metal has no sensors at all. Render "—", never 0. */
  util_percent?: number | null;
  temp_c?: number | null;
  /** "junction" — the probe that actually throttles the card. */
  hotspot_c?: number | null;
  mem_temp_c?: number | null;
  fan_percent?: number | null;
  fan_rpm?: number | null;
  power_w?: number | null;
  clock_mhz?: number | null;
}

/** The GPUs worth showing a VRAM budget for — mirrors `offload_gpus()` in
 *  backend/lld/vram.py. Summing every reported GPU would add a desktop
 *  Ryzen's iGPU aperture (~46 GB of system RAM) to the VRAM total. An
 *  all-integrated list is returned unchanged: on an APU box that iGPU is the
 *  only accelerator there is. */
export function offloadGpus(gpus: GpuInfo[]): GpuInfo[] {
  const discrete = gpus.filter(g => !g.integrated);
  return discrete.length > 0 ? discrete : gpus;
}

export interface PlatformInfo {
  os: string;
  arch: string;
  is_apple_silicon: boolean;
  unified_memory: boolean;
  cpu_name: string;
  detail: string;
}

export interface VramProcess {
  pid: number;
  process_name: string;
  used_mb: number;
  preset: string | null;
  model: string | null;
  adopted: boolean;
}

export interface PowerReport {
  gpu_w: number | null;
  cpu_w: number | null;
  // Why cpu_w is null: 'ok' | 'warming' | 'denied' | 'unsupported'.
  cpu_status?: string;
  total_w: number | null;
  energy_wh?: number;
  energy_j?: number;
  busy_seconds?: number;
}

export interface RamReport {
  total_mb: number;
  used_mb: number;
  free_mb: number;
  percent: number;
}

export interface VramReport {
  gpus: GpuInfo[];
  total_mb: number;
  used_mb: number;
  free_mb: number;
  active_estimate_mb: number;
  unified_memory?: boolean;
  platform?: PlatformInfo;
  processes: VramProcess[];
  power?: PowerReport;
  // Why a card the machine has is missing from `gpus` — a driver that is
  // installed but not answering. Null when nothing is wrong.
  probe_warning?: string | null;
  cpu_percent?: number | null;
  cpu_temp_c?: number | null;
  ram?: RamReport;
}

export const api = {
  health: () => req<Health>('/health'),
  getSettings: () => req<Settings>('/api/settings'),
  putSettings: (s: Settings) => req<Settings>('/api/settings', { method: 'PUT', body: JSON.stringify(s) }),
  settingsCheck: (s: Partial<Settings>) =>
    req<SettingsCheck>('/api/settings/check', { method: 'POST', body: JSON.stringify(s) }),

  listPresets: () => req<LlamaConfig[]>('/api/presets'),
  getPreset: (name: string) => req<LlamaConfig>(`/api/presets/${encodeURIComponent(name)}`),
  putPreset: (name: string, cfg: LlamaConfig) => req<LlamaConfig>(`/api/presets/${encodeURIComponent(name)}`, { method: 'PUT', body: JSON.stringify(cfg) }),
  createPreset: (cfg: LlamaConfig) => req<LlamaConfig>('/api/presets', { method: 'POST', body: JSON.stringify(cfg) }),
  deletePreset: (name: string) => req<{ deleted: string }>(`/api/presets/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  /** Render the exact command line this draft preset would run. */
  presetCommand: (cfg: Partial<LlamaConfig>, signal?: AbortSignal) =>
    req<CommandPreview>('/api/presets/command', { method: 'POST', body: JSON.stringify(cfg), signal }),
  /** Read a hand-typed command back into preset fields. */
  presetCommandParse: (command: string, base: Partial<LlamaConfig>) =>
    req<CommandParse>('/api/presets/command/parse', { method: 'POST', body: JSON.stringify({ command, base }) }),

  serverStatuses: () => req<{ presets: Record<string, PresetStatus> }>('/api/server/statuses'),
  serverScan: () => req<{ found: ScanEntry[] }>('/api/server/scan'),
  serverVram: () => req<VramReport>('/api/server/vram'),
  serverDevices: () => req<DeviceReport>('/api/server/devices'),
  /** Flags the configured binary accepts, read from its own --help. */
  serverFlags: () => req<{ binary: string; available: boolean; flags: FlagSpec[] }>('/api/server/flags'),
  rpcServers: () => req<{ servers: RpcServer[] }>('/api/server/rpc'),
  rpcStart: (name: string) => req<RpcServer>(`/api/server/rpc/${encodeURIComponent(name)}/start`, { method: 'POST' }),
  rpcStop: (name: string) => req<RpcServer>(`/api/server/rpc/${encodeURIComponent(name)}/stop`, { method: 'POST' }),
  systemRestart: () => req<{ restarting: boolean; pid: number }>('/api/system/restart', { method: 'POST' }),
  // `force` skips the backend fit-check preflight (see /api/server/start).
  serverStart: (preset: string, force = false) => req<PresetStatus>(`/api/server/start/${encodeURIComponent(preset)}${force ? '?force=true' : ''}`, { method: 'POST' }),
  serverStop: (preset: string) => req<PresetStatus>(`/api/server/stop/${encodeURIComponent(preset)}`, { method: 'POST' }),
  serverRestart: (preset: string, force = false) => req<PresetStatus>(`/api/server/restart/${encodeURIComponent(preset)}${force ? '?force=true' : ''}`, { method: 'POST' }),
  serverSwitch: (to_preset: string, from_preset: string | null = null, force = false) => req<{ switched: Record<string, PresetStatus> }>('/api/server/switch', { method: 'POST', body: JSON.stringify({ from_preset, to_preset, force }) }),
  serverAdopt: (pid: number, preset: string | null = null) => req<PresetStatus>('/api/server/adopt', { method: 'POST', body: JSON.stringify({ pid, preset }) }),
  serverRelease: (preset: string) => req<PresetStatus>(`/api/server/release/${encodeURIComponent(preset)}`, { method: 'POST' }),
  logsTail: (preset: string, n = 500) => req<{ lines: string[] }>(`/api/server/logs/tail/${encodeURIComponent(preset)}?n=${n}`),

  listModels: (family: string | null = null) => req<ModelEntry[]>('/api/models' + (family ? `?family=${encodeURIComponent(family)}` : '')),
  modelFamilies: () => req<string[]>('/api/models/families'),
  startVerify: (path: string) =>
    req<VerifyJob>('/api/models/verify?path=' + encodeURIComponent(path), { method: 'POST' }),
  verifyStatus: (path: string) =>
    req<VerifyJob>('/api/models/verify?path=' + encodeURIComponent(path)),

  scanModels: () => req<{ added: number; updated: number; removed: number; total: number }>('/api/models/scan', { method: 'POST' }),
  modelDefaults: (path: string, preset: string | null = null) => {
    const qs = new URLSearchParams({ path });
    if (preset) qs.set('preset', preset);
    return req<ModelDefaults>(`/api/models/defaults?${qs.toString()}`);
  },
  modelProfile: (path: string, preset: string | null = null, mmproj: string | null = null) => {
    const qs = new URLSearchParams({ path });
    if (preset) qs.set('preset', preset);
    if (mmproj) qs.set('mmproj', mmproj);
    return req<ModelProfile>(`/api/models/profile?${qs.toString()}`);
  },
  modelInfo: (path: string, preset: string | null = null) => {
    const qs = new URLSearchParams({ path });
    if (preset) qs.set('preset', preset);
    return req<ModelInfoBundle>(`/api/models/info?${qs.toString()}`);
  },
  fitCheck: (cfg: Partial<LlamaConfig>, signal?: AbortSignal) =>
    req<FitCheck>('/api/models/fit-check', { method: 'POST', body: JSON.stringify(cfg), signal }),

  hfClassify: (repo_id: string, filename?: string) => {
    const qs = new URLSearchParams({ repo_id });
    if (filename) qs.set('filename', filename);
    return req<{ brand: string; series: string; base_model: string }>(`/api/hf/classify?${qs.toString()}`);
  },
  hfSearch: (q: string, limit = 20) => req<{ results: HFSearchResult[] }>(`/api/hf/search?q=${encodeURIComponent(q)}&limit=${limit}`),
  hfFiles: (repo_id: string) => req<{ repo_id: string; brand: string; series: string; files: HFFile[] }>(`/api/hf/files?repo_id=${encodeURIComponent(repo_id)}`),
  hfDownload: (repo_id: string, filename: string, brand?: string, series?: string, base_model?: string, revision = 'main') =>
    req<DownloadJob>('/api/hf/download', { method: 'POST', body: JSON.stringify({ repo_id, filename, brand, series, base_model, revision }) }),
  hfJobs: () => req<{ jobs: DownloadJob[] }>('/api/hf/jobs'),
  hfJob: (job_id: string) => req<DownloadJob>(`/api/hf/jobs/${encodeURIComponent(job_id)}`),
  hfPause: (job_id: string) => req<DownloadJob>(`/api/hf/jobs/${encodeURIComponent(job_id)}/pause`, { method: 'POST' }),
  hfResume: (job_id: string) => req<DownloadJob>(`/api/hf/jobs/${encodeURIComponent(job_id)}/resume`, { method: 'POST' }),
  hfJobDelete: (job_id: string) => req<{ ok: boolean }>(`/api/hf/jobs/${encodeURIComponent(job_id)}`, { method: 'DELETE' }),

  setupState: () => req<SetupState>('/api/setup/state'),
  setupUseBinary: (path: string) =>
    req<{ llama_bin: string; llama_repo: string; version: string }>(
      '/api/setup/use-binary', { method: 'POST', body: JSON.stringify({ path }) }),
  setupBuild: (repo_path: string, backend: string, jobs: number | null = null) =>
    req<{ job: BuildJob; repo_path: string; cloning: boolean }>(
      '/api/setup/build', { method: 'POST', body: JSON.stringify({ repo_path, backend, jobs }) }),
  setupModelsRoot: (path: string, create = false) =>
    req<{ hf_models_root: string; scan_roots: string[] }>(
      '/api/setup/models-root', { method: 'POST', body: JSON.stringify({ path, create }) }),
  setupRescan: () =>
    req<{ count: number; roots: string[] }>('/api/setup/rescan', { method: 'POST' }),

  buildVersion: () => req<LlamaVersion>('/api/build/version'),
  buildCheck: () => req<BuildCheck>('/api/build/check'),
  buildActive: () => req<BuildJob | { status: 'idle' }>('/api/build/active'),
  buildHistory: (limit = 20) => req<BuildRecord[]>(`/api/build/history?limit=${limit}`),
  buildBackends: (refresh = false) =>
    req<BuildBackends>(`/api/build/backends${refresh ? '?refresh=true' : ''}`),
  buildRebuild: (backend = 'auto', jobs: number | null = null) =>
    req<BuildJob>('/api/build/rebuild', { method: 'POST', body: JSON.stringify({ backend, jobs }) }),

  benchActive: () => req<BenchJob | { status: 'idle' }>('/api/bench/active'),
  benchHistory: (limit = 50, model_path: string | null = null) => {
    const qs = new URLSearchParams({ limit: String(limit) });
    if (model_path) qs.set('model_path', model_path);
    return req<BenchRecord[]>(`/api/bench/history?${qs.toString()}`);
  },
  benchRun: (body: BenchRunBody) =>
    req<BenchJob>('/api/bench/run', { method: 'POST', body: JSON.stringify(body) }),
  benchCancel: () =>
    req<BenchJob>('/api/bench/cancel', { method: 'POST' }),

  routerActive: () => req<RouterActive>('/api/router/active'),
  routerModels: () => req<RouterModels>('/api/router/models'),
  /** Make the router re-read its INI. Evicts any running model whose preset
   *  changed — it reloads on the next request with the new settings. */
  routerReload: () => req<RouterModels>('/api/router/reload', { method: 'POST' }),
  routerLoad: (model: string, autoload: boolean | null = null) =>
    req<{ success: boolean }>('/api/router/load', { method: 'POST', body: JSON.stringify({ model, autoload }) }),
  routerUnload: (model: string) =>
    req<{ success: boolean }>('/api/router/unload', { method: 'POST', body: JSON.stringify({ model }) }),
  routerIniPreview: (models_dir: string | null = null) =>
    req<RouterIni>('/api/router/ini/preview' + (models_dir ? `?models_dir=${encodeURIComponent(models_dir)}` : '')),
  routerIniWrite: (models_dir: string | null = null) =>
    req<RouterIni & { bytes: number }>('/api/router/ini/write', { method: 'POST', body: JSON.stringify({ models_dir }) }),

  featuresList: (opts: { unseen_only?: boolean; arch?: string; scan_to?: string; limit?: number } = {}) => {
    const qs = new URLSearchParams();
    if (opts.unseen_only) qs.set('unseen_only', 'true');
    if (opts.arch) qs.set('arch', opts.arch);
    if (opts.scan_to) qs.set('scan_to', opts.scan_to);
    if (opts.limit) qs.set('limit', String(opts.limit));
    const s = qs.toString();
    return req<FeatureCard[]>('/api/features' + (s ? `?${s}` : ''));
  },
  /** Cards worth showing next to a draft preset — filtered on the backend
   *  against the command it would run and the binary's own flag list. */
  featureHints: (config: Partial<LlamaConfig>, architecture: string | null, limit = 3) =>
    req<{ hints: FeatureHint[] }>('/api/features/hints', {
      method: 'POST',
      body: JSON.stringify({ config, architecture, limit }),
    }),
  featuresUnseenCount: () => req<{ count: number }>('/api/features/unseen-count'),
  featuresAuthStatus: () => req<SummaryAuthStatus>('/api/features/auth-status'),
  llmEndpointProbe: (base_url?: string, api_key?: string | null) =>
    req<LlmEndpoint>('/api/features/llm-endpoint', {
      method: 'POST',
      body: JSON.stringify({ base_url: base_url ?? null, api_key: api_key ?? null })
    }),
  featureScans: (limit = 20) => req<FeatureScan[]>(`/api/features/scans?limit=${limit}`),
  featuresScanNow: () => req<FeatureScan>('/api/features/scan', { method: 'POST' }),
  featureScanRetry: (scanId: number) => req<FeatureScan>(`/api/features/scans/${scanId}/retry`, { method: 'POST' }),
  featureScanDelete: (scanId: number) => req<{ ok: boolean }>(`/api/features/scans/${scanId}`, { method: 'DELETE' }),
  featuresScansDelete: (status?: string) =>
    req<{ deleted: number }>(`/api/features/scans${status ? `?status=${encodeURIComponent(status)}` : ''}`, { method: 'DELETE' }),
  featureSeen: (id: number) => req<{ ok: boolean }>(`/api/features/${id}/seen`, { method: 'POST' }),
  featuresSeenAll: () => req<{ ok: boolean }>('/api/features/seen-all', { method: 'POST' }),
  featureTry: (id: number, preset_name: string, start: boolean) =>
    req<{ preset: string; added_flags: string[]; started: boolean }>(`/api/features/${id}/try`, { method: 'POST', body: JSON.stringify({ preset_name, start }) }),
  featureAbRun: (id: number, model_path: string, opts: { n_prompts?: number; n_gens?: number; repetitions?: number } = {}) =>
    req<{ id: number; status: string }>(`/api/features/${id}/ab`, { method: 'POST', body: JSON.stringify({ model_path, ...opts }) }),
  featureAbRuns: (feature_id: number | null = null, limit = 20) => {
    const qs = new URLSearchParams({ limit: String(limit) });
    if (feature_id != null) qs.set('feature_id', String(feature_id));
    return req<FeatureAbRun[]>(`/api/features/ab-runs?${qs.toString()}`);
  },
  featuresGuide: () => req<Guide>('/api/features/guide'),
  featuresGuideStart: () => req<{ id: number; status: string }>('/api/features/guide', { method: 'POST' }),
};

export interface ModelEntry {
  path: string;
  family: string | null;
  quant: string | null;
  size_bytes: number;
  size_gb: number;
  mtime: number;
  has_mmproj: boolean;
  mmproj_path: string | null;
  last_used: number | null;
}

export interface ModelDefaults {
  source: 'gguf' | 'props' | 'family' | 'none';
  architecture: string | null;
  name: string | null;
  base_model: string | null;
  quantized_by: string | null;
  context_length: number | null;
  chat_template_preview: string | null;
  sampling: { [k: string]: number };
  fallback_family: string | null;
}

export interface RecommendedSampling {
  model_id: string;
  source: 'props' | 'gguf' | 'family-variants' | 'family' | 'none';
  architecture: string | null;
  fallback_family: string | null;
  thinking: { [k: string]: number };
  non_thinking: { [k: string]: number };
  notes: string | null;
}

export interface RecommendedDrafter {
  label: string;
  family_pattern: string;
  name_pattern: string;
  rationale: string;
  draft_max: number | null;
  draft_min: number | null;
}

export interface ModelInfoNarrative {
  family: string;
  summary: string;
  prompt_format: string;
  behavior: string[];
  deployment: string[];
  caveats: string[];
  references: { title: string; url: string }[];
  recommended_drafter: RecommendedDrafter | null;
}

export interface ModelInfoBundle {
  defaults: ModelDefaults;
  recommended: RecommendedSampling;
  info: ModelInfoNarrative | null;
}

// ---- Fit-check ----

export interface FitSuggestion {
  id: string;
  label: string;
  explanation: string;
  add_flags: string[];
  remove_flags: string[];
  set: Record<string, number | string>;
}

export interface VerifyShard {
  name: string;
  path: string;
  size_bytes: number;
  expected_sha256: string | null;
  actual_sha256: string | null;
  /** pending | ok | corrupt | unverifiable | missing */
  status: string;
}

export interface VerifyJob {
  model_path: string;
  /** running | done | error */
  state: string;
  /** ok | corrupt | incomplete | unverifiable | running */
  verdict: string;
  bytes_total: number;
  bytes_done: number;
  percent: number;
  error: string | null;
  shards: VerifyShard[];
}

export interface FitCheck {
  available: boolean;
  level: 'fits' | 'fits_if_alone' | 'needs_offload' | 'too_big' | 'broken' | 'unknown';
  headline?: string;
  estimate?: VramEstimate;
  hardware?: { gpu_total_mb: number; gpu_free_mb: number; ram_total_mb: number; ram_available_mb: number };
  plan?: {
    gpu_need_mb: number; ram_need_mb: number; cpu_moe_layers: number;
    calibration_mb?: number;
    /** Safety margin the verdict used — smaller once the model has been measured. */
    headroom_mb?: number;
    /** True when gpu_need_mb comes from a measurement rather than the formula. */
    measured?: boolean;
  };
  model?: {
    is_moe: boolean;
    expert_count: number | null;
    expert_used_count: number | null;
    n_layers: number | null;
    n_exp_layers: number | null;
    exps_mb: number | null;
    context_length: number | null;
  };
  messages: { severity: 'info' | 'warn' | 'error'; text: string }[];
  suggestions: FitSuggestion[];
}

// ---- HuggingFace types ----

export interface HFFile {
  name: string;
  size: number | null;
}

export interface HFSearchResult {
  repo_id: string;
  likes: number;
  downloads: number;
  tags: string[];
  brand: string;
  series: string;
  files: HFFile[];
}

export interface DownloadJob {
  job_id: string;
  repo_id: string;
  filename: string;
  brand: string;
  series: string;
  base_model: string;
  target_dir: string;
  target_path: string;
  status: 'queued' | 'in_progress' | 'paused' | 'done' | 'failed';
  bytes_downloaded: number;
  total_bytes: number;
  pct: number;
  speed_bps: number;
  eta_seconds: number | null;
  error: string | null;
  created_at: number;
  finished_at: number | null;
}

// ---- first-run setup wizard ----

/** Which step the backend says the user is on. Derived server-side from the
 * real state (binary runs? models on disk? presets?) so the wizard and the
 * dashboard card can never disagree about what is still missing. */
export type SetupStep = 'llama' | 'models_dir' | 'model' | 'preset' | 'done';

export interface SetupState {
  step: SetupStep;
  /** llama.cpp and the models folder are settled — the only two prerequisites.
   * A model and a preset are suggestions, not gates: people arrive with GGUFs
   * already on an external disk. */
  required_done: boolean;
  platform: PlatformInfo;
  llama: {
    bin_path: string;
    bin_exists: boolean;
    /** The binary exists AND answered --version. */
    bin_ok: boolean;
    version: string | null;
    repo_path: string;
    repo_ok: boolean;
    default_repo_path: string;
    clone_url: string;
    candidates: { path: string; source: string }[];
  };
  toolchain: { git: string | null; cmake: string | null; compiler: string | null; make_jobs: number };
  backends: BuildBackend[];
  preferred_backend: string;
  models: { root: string; root_ok: boolean; default_root: string; count: number };
  presets: { count: number };
  build_active: BuildJob | null;
}

// ---- llama.cpp build manager types ----

export interface LlamaVersion {
  build_number: number | null;
  commit: string | null;
  raw: string;
}

export interface BuildBackend {
  id: string;
  label: string;
  cmake_flags: string[];
  available: boolean;
  detail: string;
  supported: boolean;
}

export interface BuildBackends {
  platform: PlatformInfo;
  backends: BuildBackend[];
  preferred: string;
  current: string | null;
}

export interface BuildCheck {
  branch: string;
  head_commit: string | null;
  ahead: number;
  commits: { sha: string; subject: string }[];
}

export interface BuildJob {
  id: number;
  started_at: number;
  finished_at: number | null;
  from_commit: string | null;
  to_commit: string | null;
  status: 'idle' | 'running' | 'success' | 'failed';
  log_path: string | null;
  current_step: string;
  duration_seconds: number;
}

export interface BuildRecord {
  id: number;
  started_at: number;
  finished_at: number | null;
  from_commit: string | null;
  to_commit: string | null;
  status: string;
  log_path: string | null;
}

// ---- llama-bench types ----

export interface BenchResult {
  // Raw rows from llama-bench -o json. Key fields (typical):
  model_filename?: string;
  model_type?: string;
  model_n_params?: number;
  model_size?: number;
  n_threads?: number;
  n_gpu_layers?: number;
  n_batch?: number;
  n_ubatch?: number;
  flash_attn?: number;
  cache_type_k?: string;
  cache_type_v?: string;
  n_prompt?: number;
  n_gen?: number;
  n_depth?: number;
  test?: string;          // 'pp512', 'tg128', 'pp512+tg128', etc.
  avg_ns?: number;
  avg_ts?: number;        // t/s average
  stddev_ts?: number;
  [k: string]: unknown;
}

export interface BenchRunBody {
  model_path: string;
  n_prompts?: number[];
  n_gens?: number[];
  pg_pairs?: [number, number][];
  n_gpu_layers?: number;
  batch_size?: number;
  ubatch_size?: number;
  threads?: number | null;
  flash_attn?: boolean;
  cache_type_k?: string;
  cache_type_v?: string;
  n_depth?: number;
  repetitions?: number;
  extra_flags?: string[];
}

export interface BenchJob {
  id: number;
  model_path: string;
  model_name: string;
  started_at: number;
  finished_at: number | null;
  status: 'idle' | 'running' | 'success' | 'failed' | 'cancelled';
  params: BenchRunBody | null;
  results: BenchResult[];
  log_path: string | null;
  error: string | null;
  build_number: number | null;
  build_commit: string | null;
  duration_seconds: number;
}

export interface BenchRecord {
  id: number;
  model_path: string;
  model_name: string | null;
  build_number: number | null;
  build_commit: string | null;
  started_at: number;
  finished_at: number | null;
  status: string;
  params: BenchRunBody | null;
  results: BenchResult[];
  log_path: string | null;
  error: string | null;
}

// ---- What's New (feature tracker) types ----

export interface FeatureCard {
  id: number;
  scan_id: number;
  created_at: number;
  title_tr: string;
  what_tr: string;
  how_tr: string;
  why_tr: string;
  flags: string[];
  architectures: string[];
  source_urls: string[];
  confidence: 'high' | 'medium' | 'low';
  seen: number;
  seen_at: number | null;
  from_commit: string | null;
  to_commit: string | null;
  build_number: number | null;
}

/** One What's New card judged against the preset in the editor. `add_flags`
 *  is what may actually be appended; the other buckets explain the rest. */
export interface FeatureHint {
  card: FeatureCard;
  match: 'architecture' | 'flags';
  add_flags: string[];
  present: string[];
  managed: string[];
  needs_value: string[];
  unknown: string[];
}

export interface FeatureScan {
  id: number;
  created_at: number;
  from_commit: string | null;
  to_commit: string | null;
  build_number: number | null;
  new_flags: { flag: string; usage: string }[];
  removed_flags: string[];
  commits: { sha: string; subject: string }[];
  releases: { tag: string; name: string; body: string }[];
  status: 'pending' | 'summarized' | 'failed' | 'empty';
  error: string | null;
  seen: number;
}

export interface FeatureAbRun {
  id: number;
  feature_id: number | null;
  model_path: string;
  flags: string[];
  created_at: number;
  status: 'running' | 'success' | 'failed';
  error: string | null;
  off: { id: number; status: string; results: BenchResult[] } | null;
  on: { id: number; status: string; results: BenchResult[] } | null;
}

export interface Guide {
  id?: number;
  created_at?: number;
  build_number?: number | null;
  commit_sha?: string | null;
  status: 'none' | 'running' | 'success' | 'failed';
  content_md?: string | null;
  error?: string | null;
}

export function formatUptime(seconds: number | null): string {
  if (seconds == null) return '—';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}
