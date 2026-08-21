"""Summarizing a build update with a LOCAL model.

Every case here is a failure that actually killed a scan on this box:

  scans 2-6  the prompt (21869 tokens) was larger than the llama-server
             context the preset happened to be started with (3072/6144)
  scan 7     the answer stopped mid-string at char 3962, and the all-or-nothing
             parse threw away the six complete cards that had arrived
  scan 8     settings said `Muse-Glimmer-30B-UD-Q4_K_XL` while the endpoint
             served `/ml/ai/models/gguf/Ornith-1.0-35B/Ornith-…-Q6_K_XL.gguf`

Hosted APIs hit none of these, which is why they went unnoticed until the local
provider was wired up.
"""
from __future__ import annotations

import json

import pytest

from lld import features
from lld.features import (
    ContextTooSmall,
    FeatureError,
    FeatureTracker,
    TruncatedOutput,
    extract_json_object,
    parse_feature_cards,
    resolve_model_id,
    strip_reasoning,
)


def _card(title="Flash attention landed", **over):
    c = {
        "title_tr": title,
        "what_tr": "what landed",
        "how_tr": "--flash-attn on",
        "why_tr": "faster prompt processing",
        "flags": ["--flash-attn on"],
        "architectures": ["qwen3"],
        "source_urls": [],
        "confidence": "high",
    }
    c.update(over)
    return c


# ---------- model id resolution (scan 8) ----------

class TestResolveModelId:
    def test_exact_id_is_kept(self):
        assert resolve_model_id("gpt-4o", ["gpt-4o", "o3"]) == "gpt-4o"

    def test_name_matches_the_served_gguf_path(self):
        served = ["/ml/ai/models/gguf/Ornith-1.0-35B/Ornith-1.0-35B-UD-Q6_K_XL.gguf"]
        assert resolve_model_id("Ornith-1.0-35B-UD-Q6_K_XL", served) == served[0]

    def test_single_served_model_wins_over_a_stale_setting(self):
        """The live failure: the preset was restarted with another model and the
        setting still named the old one. One model is loaded; use it."""
        served = ["/ml/ai/models/gguf/Ornith-1.0-35B/Ornith-1.0-35B-UD-Q6_K_XL.gguf"]
        assert resolve_model_id("Muse-Glimmer-30B-UD-Q4_K_XL", served) == served[0]

    def test_unknown_model_among_many_names_the_alternatives(self):
        with pytest.raises(FeatureError, match="not served"):
            resolve_model_id("nope", ["a", "b"])

    def test_no_probe_result_leaves_the_setting_alone(self):
        assert resolve_model_id("whatever", []) == "whatever"


# ---------- reading the answer out ----------

class TestStripReasoning:
    def test_balanced_think_block(self):
        assert strip_reasoning("<think>hmm</think>{\"a\":1}") == '{"a":1}'

    def test_orphan_close_tag(self):
        """Some chat templates emit the opener themselves, so only the closer
        reaches the client."""
        assert strip_reasoning("planning...</think>ANSWER") == "ANSWER"

    def test_unterminated_think_leaves_no_answer(self):
        assert strip_reasoning("prefix<think>still thinking") == "prefix"


class TestExtractJsonObject:
    def test_ignores_prose_around_the_object(self):
        text = 'Sure! Here you go:\n{"features": []}\nHope that helps.'
        assert json.loads(extract_json_object(text)) == {"features": []}

    def test_braces_inside_strings_do_not_end_the_object(self):
        text = '{"how_tr": "use {n} slots \\" here"}'
        assert json.loads(extract_json_object(text))["how_tr"] == 'use {n} slots " here'


class TestParseFeatureCards:
    def test_plain_object(self):
        cards, salvaged = parse_feature_cards(json.dumps({"features": [_card()]}))
        assert len(cards) == 1 and not salvaged

    def test_fenced_with_commentary(self):
        raw = "Here:\n```json\n" + json.dumps({"features": [_card()]}) + "\n```"
        cards, salvaged = parse_feature_cards(raw)
        assert len(cards) == 1 and not salvaged

    def test_truncated_array_keeps_the_complete_cards(self):
        """Scan 7: the response died inside the third card's `what_tr`."""
        good = json.dumps({"features": [_card("one"), _card("two")]})
        raw = good[: good.rindex("]")] + ', {"title_tr": "three", "what_tr": "half of a sen'
        cards, salvaged = parse_feature_cards(raw)
        assert salvaged
        assert [c["title_tr"] for c in cards] == ["one", "two"]

    def test_nothing_parseable_raises(self):
        with pytest.raises(FeatureError):
            parse_feature_cards("I cannot help with that.")


# ---------- card validation ----------

class TestNormalizeCards:
    def test_one_bad_card_does_not_sink_the_batch(self):
        cards = FeatureTracker._normalize_cards(
            [_card("good"), _card("bad", why_tr="   "), "not an object"]
        )
        assert [c["title_tr"] for c in cards] == ["good"]

    def test_all_bad_raises(self):
        with pytest.raises(FeatureError, match="no usable cards"):
            FeatureTracker._normalize_cards([_card(why_tr=None)])

    def test_empty_response_stays_empty(self):
        assert FeatureTracker._normalize_cards([]) == []

    def test_blank_flag_is_dropped(self):
        """`"".split()[0]` is an IndexError in the insert loop downstream."""
        card = FeatureTracker._normalize_cards([_card(flags=["", "  ", "--flash-attn"])])[0]
        assert card["flags"] == ["--flash-attn"]

    def test_code_fences_are_stripped_from_prose_fields(self):
        card = FeatureTracker._normalize_cards(
            [_card(how_tr="```bash\nllama-server --flash-attn on\n```")]
        )[0]
        assert card["how_tr"] == "llama-server --flash-attn on"

    def test_confidence_is_defaulted_not_rejected(self):
        assert FeatureTracker._normalize_cards([_card(confidence="very")])[0]["confidence"] == "medium"


# ---------- fitting the prompt to the context (scans 2-6) ----------

@pytest.fixture
def scan():
    return {
        "from_commit": "aaa", "to_commit": "bbb",
        "new_flags": [{"flag": f"--flag{i}", "usage": f"--flag{i} does thing {i}"}
                      for i in range(60)],
        "removed_flags": ["--old"],
        "commits": [{"sha": f"{i:07x}", "subject": f"server: change number {i}"}
                    for i in range(300)],
        "releases": [{"tag": "b1000", "name": "rel", "body": "note. " * 3000}],
    }


class TestPromptShrinking:
    def test_each_level_is_smaller_than_the_last(self, scan):
        ft = FeatureTracker()
        sizes = [
            len(ft._build_prompt(scan, ["qwen3"], {"--flash-attn": "--flash-attn on"}, lvl))
            for lvl in range(len(features._DETAIL_LEVELS))
        ]
        assert sizes == sorted(sizes, reverse=True), sizes
        assert sizes[-1] < sizes[0] / 3

    def test_lowest_level_drops_release_notes(self, scan):
        p = FeatureTracker()._build_prompt(scan, [], {}, len(features._DETAIL_LEVELS) - 1)
        assert "note. note." not in p

    def test_card_budget_follows_the_prompt_size(self, scan):
        ft = FeatureTracker()
        assert "AT MOST 8 cards" in ft._build_prompt(scan, [], {}, 0)
        assert "AT MOST 4 cards" in ft._build_prompt(scan, [], {}, 3)

    def test_level_is_clamped(self, scan):
        FeatureTracker()._build_prompt(scan, [], {}, 99)  # must not IndexError


class TestContextErrors:
    def test_llama_cpp_overflow_400_becomes_typed(self):
        body = json.dumps({"error": {
            "code": 400,
            "message": "request (21869 tokens) exceeds the available context size (3072 tokens)",
            "type": "exceed_context_size_error",
            "n_prompt_tokens": 21869, "n_ctx": 3072,
        }})
        err = FeatureTracker._openai_400(body, None)
        assert isinstance(err, ContextTooSmall)
        assert err.n_ctx == 3072 and err.n_prompt == 21869

    def test_other_400s_stay_plain(self):
        body = json.dumps({"error": {"code": 400, "message": "model 'x' not found"}})
        err = FeatureTracker._openai_400(body, None)
        assert isinstance(err, FeatureError) and not isinstance(err, ContextTooSmall)

    def test_unparseable_400_body(self):
        assert isinstance(FeatureTracker._openai_400("<html>nope</html>", None), FeatureError)


class TestReadCompletion:
    def test_plain_content(self):
        text, finish = FeatureTracker._read_completion(
            {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]})
        assert (text, finish) == ("hello", "stop")

    def test_content_parts_are_joined(self):
        text, _ = FeatureTracker._read_completion(
            {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]})
        assert text == "ab"

    def test_falls_back_to_reasoning_content(self):
        text, _ = FeatureTracker._read_completion(
            {"choices": [{"message": {"content": "", "reasoning_content": "answer"}}]})
        assert text == "answer"

    def test_all_budget_spent_on_thinking_is_named_as_truncation(self):
        with pytest.raises(TruncatedOutput):
            FeatureTracker._read_completion(
                {"choices": [{"message": {"content": "<think>and on and on"},
                              "finish_reason": "length"}]})

    def test_garbage_shape(self):
        with pytest.raises(FeatureError, match="unexpected"):
            FeatureTracker._read_completion({"error": "nope"})


# ---------- the adaptive loop, end to end ----------

@pytest.fixture
def local_endpoint(monkeypatch):
    """Stand in for a llama-server: a fixed context window and a scripted
    sequence of replies."""
    def _setup(n_ctx, replies):
        monkeypatch.setattr(features, "_probe_cache", {})
        monkeypatch.setattr(
            FeatureTracker, "_openai_conf",
            lambda self: ("http://local", "some-model", {}),
        )

        async def fake_probe(base, headers=None, timeout=8.0):
            return {"models": ["some-model"], "n_ctx": n_ctx, "native": True}

        async def fake_count(text, base, headers=None, native=False):
            return len(text) // 4          # deterministic stand-in for /tokenize

        calls = []

        async def fake_completion(self, prompt, **kw):
            calls.append(prompt)
            reply = replies[min(len(calls) - 1, len(replies) - 1)]
            if isinstance(reply, Exception):
                raise reply
            return reply, "stop"

        monkeypatch.setattr(features, "probe_endpoint", fake_probe)
        monkeypatch.setattr(features, "count_tokens", fake_count)
        monkeypatch.setattr(FeatureTracker, "_openai_completion", fake_completion)
        return calls
    return _setup


class TestAdaptiveSummary:
    @pytest.mark.asyncio
    async def test_roomy_context_uses_the_full_prompt(self, scan, local_endpoint):
        calls = local_endpoint(120_000, [json.dumps({"features": [_card()]})])
        cards = await FeatureTracker()._summarize_openai_adaptive(scan, ["qwen3"], {})
        assert len(cards) == 1
        assert len(calls) == 1
        assert "0000000 server: change number 0" in calls[0]

    @pytest.mark.asyncio
    async def test_small_context_shrinks_before_calling(self, scan, local_endpoint):
        """Scans 2-6 burned a slow local generation to be told the prompt was
        too big. The size is known up front, so pick a level that fits."""
        calls = local_endpoint(4096, [json.dumps({"features": [_card()]})])
        cards = await FeatureTracker()._summarize_openai_adaptive(scan, [], {})
        assert len(cards) == 1
        assert len(calls) == 1, "must not spend a call discovering the overflow"
        full = FeatureTracker()._build_prompt(scan, [], {}, 0)
        assert len(calls[0]) < len(full)
        assert len(calls[0]) // 4 + features._MIN_COMPLETION < 4096

    @pytest.mark.asyncio
    async def test_overflow_at_runtime_retries_smaller(self, scan, local_endpoint):
        """A server that reports no n_ctx can still refuse the prompt."""
        calls = local_endpoint(None, [
            ContextTooSmall("too big", n_ctx=4096, n_prompt=9000),
            json.dumps({"features": [_card()]}),
        ])
        cards = await FeatureTracker()._summarize_openai_adaptive(scan, [], {})
        assert len(cards) == 1
        assert len(calls) == 2 and len(calls[1]) < len(calls[0])

    @pytest.mark.asyncio
    async def test_truncated_answer_retries_at_lower_detail(self, scan, local_endpoint):
        calls = local_endpoint(None, [
            TruncatedOutput("cut off"),
            json.dumps({"features": [_card()]}),
        ])
        assert len(await FeatureTracker()._summarize_openai_adaptive(scan, [], {})) == 1
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_hopeless_context_says_what_to_change(self, scan, local_endpoint):
        """When even the minimum prompt leaves no room to answer, the error has
        to tell the user to raise --ctx-size rather than read as a crash."""
        local_endpoint(2048, [json.dumps({"features": [_card()]})])
        with pytest.raises(ContextTooSmall, match="ctx-size"):
            await FeatureTracker()._summarize_openai_adaptive(scan, [], {})

    @pytest.mark.asyncio
    async def test_every_level_failing_reports_the_last_reason(self, scan, local_endpoint):
        local_endpoint(None, [TruncatedOutput("cut off")])
        with pytest.raises(TruncatedOutput):
            await FeatureTracker()._summarize_openai_adaptive(scan, [], {})

    @pytest.mark.asyncio
    async def test_empty_feature_list_is_a_valid_answer(self, scan, local_endpoint):
        """"Nothing noteworthy landed" must not look like a failure."""
        local_endpoint(120_000, [json.dumps({"features": []})])
        assert await FeatureTracker()._summarize_openai_adaptive(scan, [], {}) == []


# ---------- the HTTP call itself ----------

class _FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    @property
    def text(self):
        return self._payload if isinstance(self._payload, str) else json.dumps(self._payload)

    def json(self):
        return json.loads(self._payload) if isinstance(self._payload, str) else self._payload


@pytest.fixture
def fake_post(monkeypatch):
    """Replace the chat-completions POST with a scripted sequence, and record
    the payloads that were sent."""
    def _setup(responses, n_ctx=None, served=("some-model",)):
        sent = []

        async def fake_probe(base, headers=None, timeout=8.0):
            return {"models": list(served), "n_ctx": n_ctx, "native": bool(n_ctx)}

        async def fake_count(text, base, headers=None, native=False):
            return len(text) // 4

        class _FakeClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None):
                # A copy: the caller mutates its payload when it re-resolves
                # the model id, which would rewrite history here.
                sent.append((url, dict(json or {})))
                r = responses[min(len(sent) - 1, len(responses) - 1)]
                if isinstance(r, Exception):
                    raise r
                return r

        monkeypatch.setattr(features, "probe_endpoint", fake_probe)
        monkeypatch.setattr(features, "count_tokens", fake_count)
        monkeypatch.setattr(features.httpx, "AsyncClient", _FakeClient)
        monkeypatch.setattr(FeatureTracker, "_RETRY_BACKOFF", 0.0)
        monkeypatch.setattr(
            FeatureTracker, "_openai_conf",
            lambda self: ("http://local", "some-model", {}),
        )
        return sent
    return _setup


def _ok(text):
    return _FakeResponse(200, {"choices": [{"message": {"content": text},
                                            "finish_reason": "stop"}]})


class TestOpenAiCompletion:
    @pytest.mark.asyncio
    async def test_happy_path(self, fake_post):
        sent = fake_post([_ok("hi")])
        text, finish = await FeatureTracker()._openai_completion("prompt")
        assert (text, finish) == ("hi", "stop")
        assert sent[0][0] == "http://local/chat/completions"

    @pytest.mark.asyncio
    async def test_sends_the_resolved_model_id(self, fake_post):
        """Scan 8's 400 came from posting the name in settings verbatim."""
        served = ("/ml/ai/models/gguf/Ornith-1.0-35B/Ornith-1.0-35B-UD-Q6_K_XL.gguf",)
        sent = fake_post([_ok("hi")], served=served)
        await FeatureTracker()._openai_completion("prompt")
        assert sent[0][1]["model"] == served[0]

    @pytest.mark.asyncio
    async def test_max_tokens_fits_the_context(self, fake_post):
        sent = fake_post([_ok("hi")], n_ctx=8192)
        await FeatureTracker()._openai_completion("x" * 4000)   # 1000 tokens
        assert sent[0][1]["max_tokens"] == 8192 - 1000 - features._CTX_MARGIN

    @pytest.mark.asyncio
    async def test_max_tokens_is_capped_on_a_huge_context(self, fake_post):
        sent = fake_post([_ok("hi")], n_ctx=200_000)
        await FeatureTracker()._openai_completion("prompt")
        assert sent[0][1]["max_tokens"] == features._MAX_COMPLETION

    @pytest.mark.asyncio
    async def test_refuses_before_calling_when_there_is_no_room_to_answer(self, fake_post):
        sent = fake_post([_ok("hi")], n_ctx=4096)
        with pytest.raises(ContextTooSmall, match="ctx-size"):
            await FeatureTracker()._openai_completion("x" * 14_000)   # 3500 tokens
        assert not sent

    @pytest.mark.asyncio
    async def test_retries_while_the_model_is_still_loading(self, fake_post):
        """llama-server answers 503 until the weights are resident — a scan
        started right after a preset restart used to just fail."""
        sent = fake_post([_FakeResponse(503, "loading model"), _ok("hi")])
        assert (await FeatureTracker()._openai_completion("p"))[0] == "hi"
        assert len(sent) == 2

    @pytest.mark.asyncio
    async def test_retries_a_dropped_connection(self, fake_post):
        sent = fake_post([features.httpx.ConnectError("refused"), _ok("hi")])
        assert (await FeatureTracker()._openai_completion("p"))[0] == "hi"
        assert len(sent) == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_the_attempt_budget(self, fake_post):
        sent = fake_post([_FakeResponse(503, "still loading")])
        with pytest.raises(FeatureError, match="503"):
            await FeatureTracker()._openai_completion("p")
        assert len(sent) == 3

    @pytest.mark.asyncio
    async def test_auth_failure_is_not_retried(self, fake_post):
        sent = fake_post([_FakeResponse(401, "bad key")])
        with pytest.raises(FeatureError, match="401"):
            await FeatureTracker()._openai_completion("p")
        assert len(sent) == 1

    @pytest.mark.asyncio
    async def test_context_overflow_400_is_not_retried(self, fake_post):
        body = json.dumps({"error": {
            "message": "request (21869 tokens) exceeds the available context size (3072 tokens)",
            "type": "exceed_context_size_error", "n_ctx": 3072, "n_prompt_tokens": 21869}})
        sent = fake_post([_FakeResponse(400, body)])
        with pytest.raises(ContextTooSmall):
            await FeatureTracker()._openai_completion("p")
        assert len(sent) == 1

    @pytest.mark.asyncio
    async def test_a_swapped_model_is_re_probed_not_failed(self, monkeypatch, fake_post):
        """The probe is cached for 30s; restarting the preset with another model
        inside that window used to produce scan 8's `model … not found`."""
        sent = fake_post(
            [_FakeResponse(400, json.dumps({"error": {"message": "model 'old' not found"}})),
             _ok("hi")],
            served=("stale-model",),
        )
        calls = {"n": 0}
        real_probe = features.probe_endpoint

        async def counting_probe(base, headers=None, timeout=8.0):
            calls["n"] += 1
            info = await real_probe(base, headers, timeout)
            return {**info, "models": ["fresh-model"]} if calls["n"] > 1 else info

        monkeypatch.setattr(features, "probe_endpoint", counting_probe)
        assert (await FeatureTracker()._openai_completion("p"))[0] == "hi"
        assert sent[0][1]["model"] == "stale-model"
        assert sent[1][1]["model"] == "fresh-model"


# ---------- "auto": no model id typed by hand ----------

class TestAutoModel:
    def test_empty_setting_takes_the_loaded_model(self):
        """The local case: the server knows what it loaded, so asking the user
        to retype its name only creates a way to get it wrong."""
        served = ["/ml/ai/models/gguf/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-UD-Q4_K_XL.gguf"]
        assert resolve_model_id("", served) == served[0]

    def test_whitespace_is_empty(self):
        assert resolve_model_id("   ", ["only-one"]) == "only-one"

    def test_several_models_and_no_choice_asks_for_one(self):
        with pytest.raises(FeatureError, match="pick one"):
            resolve_model_id("", ["a", "b"])

    def test_unreadable_model_list_says_so(self):
        with pytest.raises(FeatureError, match="model list"):
            resolve_model_id("", [])

    def test_status_is_ready_without_a_model_name(self, monkeypatch):
        """summary_status used to require llm_model, so 'auto' would read as
        an unconfigured provider and never even try."""
        from lld import settings as settings_mod

        cfg = settings_mod.load_settings().model_copy(update={
            "llm_provider": "openai",
            "llm_base_url": "http://127.0.0.1:8083",
            "llm_model": "",
        })
        monkeypatch.setattr(features, "load_settings", lambda: cfg)
        st = features.summary_status()
        assert st["mode"] == "openai" and st["model"] is None and st["detail"] is None

    def test_status_still_needs_a_base_url(self, monkeypatch):
        from lld import settings as settings_mod

        cfg = settings_mod.load_settings().model_copy(update={
            "llm_provider": "openai", "llm_base_url": "", "llm_model": "",
        })
        monkeypatch.setattr(features, "load_settings", lambda: cfg)
        assert features.summary_status()["mode"] == "none"
