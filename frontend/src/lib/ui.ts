/** Shared tiny UI helpers used across routes. */

export interface ModelSource {
  model_path?: string | null;
  hf_file?: string | null;
  hf_repo?: string | null;
}

/** Human-readable model label: GGUF basename without extension,
 * falling back to HF file/repo, then to the given name. */
export function modelLabel(cfg: ModelSource | null | undefined, fallback: string): string {
  if (cfg?.model_path) {
    const base = cfg.model_path.split('/').pop() ?? cfg.model_path;
    return base.replace(/\.gguf$/i, '');
  }
  if (cfg?.hf_file) return cfg.hf_file.replace(/\.gguf$/i, '');
  if (cfg?.hf_repo) return cfg.hf_repo;
  return fallback;
}
