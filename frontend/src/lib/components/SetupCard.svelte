<script lang="ts">
  /**
   * First-run nudge on the dashboard. A fresh install lands on an empty
   * dashboard with no way to tell what is missing — this card answers "what do
   * I do now?" with one sentence and one button, and disappears the moment the
   * answer is "nothing". Deliberately shows a single step at a time: a
   * checklist of everything that is not yet done reads as a chore list.
   *
   * The verdict comes from /api/setup/state, the same endpoint the wizard at
   * /setup renders, so the card and the wizard can never disagree. Nothing is
   * persisted, so the card also reappears if the user later deletes their
   * models or moves the binary.
   */
  import { onMount } from 'svelte';
  import { api, type PresetStatus, type SetupState } from '$lib/api';
  import { t } from '$lib/i18n.svelte';

  interface Props {
    /** Statuses the dashboard already polls — avoids a second poll here. */
    statuses?: Record<string, PresetStatus>;
    /** One-line hardware summary shown while picking a model. */
    hardware?: string | null;
  }
  let { statuses = {}, hardware = null }: Props = $props();

  let setup = $state<SetupState | null>(null);
  let dismissed = $state(false);

  async function probe() {
    try { setup = await api.setupState(); } catch { setup = null; }
  }

  onMount(() => {
    probe();
    // Cheap and rare: re-probe every 15 s so the card follows along while the
    // user fixes things in another tab (Setup, Settings, Download).
    const timer = setInterval(probe, 15000);
    return () => clearInterval(timer);
  });

  const anyRunning = $derived(Object.values(statuses).some((s) => s.running));

  type Step = {
    n: number;
    title: string;
    detail: string;
    action: { label: string; href?: string; onclick?: () => void };
    secondary?: { label: string; href: string };
  } | null;

  const step = $derived.by<Step>(() => {
    if (!setup) return null;   // still loading, or backend unreachable
    if (setup.step === 'llama') return {
      n: 1,
      title: t('Install llama.cpp'),
      detail: t('It runs your models. LlamaDeck can clone and build it for you, or use one you already have.'),
      action: { label: t('Open setup'), href: '/setup' },
      secondary: { label: t('Build it from source'), href: '/build' },
    };
    if (setup.step === 'models_dir') return {
      n: 2,
      title: t('Choose a models folder'),
      detail: t('Where your GGUF files live. LlamaDeck indexes this folder and downloads into it.'),
      action: { label: t('Open setup'), href: '/setup' },
    };
    if (setup.step === 'model') return {
      n: 3,
      title: t('Add a model'),
      detail: hardware
        ? t('Download a GGUF, or copy in ones you already have. Your machine: {hw}.', { hw: hardware })
        : t('Download a GGUF from HuggingFace, or copy in ones you already have.'),
      action: { label: t('Find a model'), href: '/download' },
      secondary: { label: t('I have my own models'), href: '/setup' },
    };
    if (setup.step === 'preset') return {
      n: 4,
      title: t('Create your first preset'),
      detail: t('Pick a model, say what you want it for, and LlamaDeck fills in the settings. You can change every one of them afterwards.'),
      action: { label: t('Start the wizard'), href: '/presets?new=1' },
    };
    if (!anyRunning) return {
      n: 5,
      title: t('Start it'),
      detail: t('Your preset is ready. Starting it gives you an OpenAI-compatible endpoint on your machine.'),
      action: { label: t('Go to Server'), href: '/server' },
    };
    return null;
  });
</script>

{#if step && !dismissed}
  <section class="rounded-lg border border-cyan-900/60 bg-cyan-950/20 p-5">
    <div class="flex items-start gap-4">
      <div class="flex-1 min-w-0">
        <div class="text-[11px] font-mono uppercase tracking-wider text-cyan-500/80">
          {t('Setup')} · {t('step {n} of 5', { n: step.n })}
        </div>
        <h2 class="mt-1 text-lg font-semibold text-slate-100">{step.title}</h2>
        <p class="mt-1 max-w-2xl text-sm leading-relaxed text-slate-400">{step.detail}</p>
        <div class="mt-3 flex flex-wrap items-center gap-3">
          <a
            href={step.action.href}
            class="rounded bg-cyan-700/40 border border-cyan-600 px-3 py-1.5 text-sm text-slate-100 hover:bg-cyan-700/60"
          >{step.action.label}</a>
          {#if step.secondary}
            <a href={step.secondary.href} class="text-sm text-slate-400 underline decoration-dotted hover:text-slate-200">
              {step.secondary.label}
            </a>
          {/if}
        </div>
      </div>
      <button
        onclick={() => (dismissed = true)}
        title={t('Hide until the next reload')}
        aria-label={t('Hide until the next reload')}
        class="shrink-0 rounded px-2 py-1 text-slate-600 hover:text-slate-300"
      >✕</button>
    </div>
  </section>
{/if}
