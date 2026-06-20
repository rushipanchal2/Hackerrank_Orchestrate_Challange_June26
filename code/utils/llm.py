"""LiteLLM-backed LLM client with cooldown-based routing and dual-key Groq support.

Model priority (free tier only):
  GROUP gemini  — gemini/gemini-2.5-flash          (vision, GEMINI_API_KEY)
                  gemini/gemini-2.5-flash-lite      (vision, GEMINI_API_KEY, lighter quota)
  GROUP groq1   — llama-4-scout (vision)            (GROQ_API_KEY)
                  llama-3.3-70b (text-only)
                  llama-3.1-8b  (text-only)
  GROUP groq2   — same models, GROQ_API_KEY_2       (separate TPM bucket = 2× effective TPM)

Routing:
  - Each entry has a unique cooldown key (group:model) so groq1 and groq2
    rate limits are tracked independently.
  - Available models (cooldown expired) are always tried before cooling ones.
  - api_key is passed explicitly per call — LiteLLM env vars are not relied on.
  - On 429: set per-entry cooldown from retry-after, move to next available entry.
  - If all entries cooling: wait for soonest, retry up to 3×.
  - Thread-local last_model: 2 parallel threads track their own last-used model.
"""

import os
import re
import threading
import time

import litellm
from litellm import completion, RateLimitError, BadRequestError, AuthenticationError

litellm.set_verbose = False
litellm.suppress_debug_info = True
os.environ.setdefault("LITELLM_LOG", "ERROR")
import logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

# ── Model registry ────────────────────────────────────────────────────────────

def _build_model_list() -> list[dict]:
    gemini_key  = os.getenv("GEMINI_API_KEY", "").strip()
    groq_key1   = os.getenv("GROQ_API_KEY", "").strip()
    groq_key2   = os.getenv("GROQ_API_KEY_2", "").strip()

    entries = []

    if gemini_key:
        entries += [
            {"group": "gemini", "vision": True,
             "model": "gemini/gemini-2.5-flash",      "api_key": gemini_key},
            {"group": "gemini", "vision": True,
             "model": "gemini/gemini-2.5-flash-lite",  "api_key": gemini_key},
        ]
    if groq_key1:
        entries += [
            {"group": "groq1", "vision": True,
             "model": "groq/meta-llama/llama-4-scout-17b-16e-instruct", "api_key": groq_key1},
            {"group": "groq1", "vision": False,
             "model": "groq/llama-3.3-70b-versatile",  "api_key": groq_key1},
            {"group": "groq1", "vision": False,
             "model": "groq/llama-3.1-8b-instant",     "api_key": groq_key1},
        ]
    if groq_key2:
        entries += [
            {"group": "groq2", "vision": True,
             "model": "groq/meta-llama/llama-4-scout-17b-16e-instruct", "api_key": groq_key2},
            {"group": "groq2", "vision": False,
             "model": "groq/llama-3.3-70b-versatile",  "api_key": groq_key2},
            {"group": "groq2", "vision": False,
             "model": "groq/llama-3.1-8b-instant",     "api_key": groq_key2},
        ]
    return entries


_ACTIVE_MODELS: list[dict] = _build_model_list()

# ── Per-entry unique key (group:model) for independent cooldown tracking ──────

def _entry_key(entry: dict) -> str:
    return f"{entry['group']}:{entry['model']}"

# ── State ─────────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_cooldowns: dict[str, float] = {}   # entry_key → time.time() when available
_warned:    set[str]         = set() # suppress repeated WARN lines per entry
_thread_local = threading.local()   # per-thread last_model

# ── Token usage accounting (real numbers for the operational report) ──────────
_usage_lock = threading.Lock()
_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def _add_usage(resp) -> None:
    """Accumulate real prompt/completion token counts from a LiteLLM response."""
    try:
        u = resp.usage
        pt = getattr(u, "prompt_tokens", 0) or 0
        ct = getattr(u, "completion_tokens", 0) or 0
    except Exception:
        return
    with _usage_lock:
        _usage["prompt_tokens"] += pt
        _usage["completion_tokens"] += ct
        _usage["calls"] += 1


def get_usage() -> dict:
    """Return a snapshot of accumulated token usage since the last reset."""
    with _usage_lock:
        return dict(_usage)


def reset_usage() -> None:
    with _usage_lock:
        _usage.update(prompt_tokens=0, completion_tokens=0, calls=0)


def _set_cooldown(entry: dict, err_str: str) -> None:
    waits = re.findall(r"try again in (\d+(?:\.\d+)?)\s*s", err_str.lower())
    wait = max((float(w) for w in waits), default=15) + 2
    with _lock:
        _cooldowns[_entry_key(entry)] = time.time() + wait


def _is_available(entry: dict) -> bool:
    return time.time() >= _cooldowns.get(_entry_key(entry), 0)


def _sorted_entries() -> list[dict]:
    """Available entries first, then sorted by soonest cooldown."""
    now = time.time()
    avail   = [e for e in _ACTIVE_MODELS if _cooldowns.get(_entry_key(e), 0) <= now]
    cooling = sorted(
        [e for e in _ACTIVE_MODELS if _cooldowns.get(_entry_key(e), 0) > now],
        key=lambda e: _cooldowns.get(_entry_key(e), 0),
    )
    return avail + cooling


def _soonest_wait() -> float:
    now = time.time()
    waits = [max(_cooldowns.get(_entry_key(e), 0) - now, 0) for e in _ACTIVE_MODELS]
    return min(waits) if waits else 0


# ── Banner ────────────────────────────────────────────────────────────────────

def startup_banner() -> None:
    if not _ACTIVE_MODELS:
        raise SystemExit(
            "No LLM API keys found.\n"
            "Set GEMINI_API_KEY and/or GROQ_API_KEY in .env\n"
            "Optional: GROQ_API_KEY_2 for a second Groq TPM bucket."
        )
    print("\n  [MODELS] Priority order:", flush=True)
    for i, e in enumerate(_ACTIVE_MODELS):
        label = "PRIMARY" if i == 0 else f"FALLBACK-{i}"
        key_hint = f"[{e['group']}]"
        vis = "vision" if e["vision"] else "text"
        print(f"    {i+1}. {e['model']}  {key_hint}  [{label}]  ({vis})", flush=True)
    print(flush=True)


def _announce(msg: str) -> None:
    print(msg, flush=True)
    try:
        from code.utils.logger import _append
        _append(f"## [MODEL EVENT]\n\n{msg}")
    except Exception:
        pass


# ── Message builder ───────────────────────────────────────────────────────────

def _build_messages(system: str, text: str,
                    images: list | None, vision: bool) -> list[dict]:
    if not vision or not images:
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": text},
        ]
    content: list = []
    for img in images:
        if img.get("exists") and img.get("base64_str"):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img['base64_str']}"},
            })
            content.append({"type": "text", "text": f"[Image ID: {img['image_id']}]"})
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": content},
    ]


# ── Core call ─────────────────────────────────────────────────────────────────

def _call(system: str, text: str, images: list | None,
          max_tokens: int) -> tuple[str, str]:
    """Call LiteLLM with cooldown-aware fallback. Returns (response_text, model_label)."""
    has_images = bool(images)
    last_exc: Exception | None = None
    prev_model: str = getattr(_thread_local, "last_model", "")

    for attempt in range(4):
        ordered = _sorted_entries()

        if all(not _is_available(e) for e in ordered):
            if attempt >= 3:
                break
            wait = _soonest_wait()
            print(f"  [WAIT] All models cooling. Retrying in {wait:.0f}s "
                  f"(attempt {attempt+1}/3) …", flush=True)
            time.sleep(max(wait, 1) + 1)
            continue

        for entry in ordered:
            if not _is_available(entry):
                continue

            model       = entry["model"]
            api_key     = entry["api_key"]
            is_vision   = entry["vision"]
            use_vision  = has_images and is_vision
            ekey        = _entry_key(entry)

            messages = _build_messages(
                system, text,
                images if use_vision else None,
                vision=use_vision,
            )

            try:
                resp = completion(
                    model=model,
                    api_key=api_key,        # explicit per-entry key
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0,
                    timeout=60,
                )
                result = resp.choices[0].message.content or ""
                _add_usage(resp)

                label = f"{model} [{entry['group']}]"
                if prev_model and prev_model != label:
                    suffix = " (text-only)" if has_images and not is_vision else ""
                    _announce(f"  [MODEL SWITCH] {prev_model} → {label}{suffix}")
                elif not prev_model:
                    print(f"  [MODEL] Using: {label}", flush=True)

                with _lock:
                    _warned.discard(ekey)
                _thread_local.last_model = label
                return result, label

            except (RateLimitError, BadRequestError, AuthenticationError) as exc:
                last_exc = exc
                _set_cooldown(entry, str(exc))
                if ekey not in _warned:
                    short = str(exc)[:120].replace("\n", " ")
                    print(f"  [WARN] {model} [{entry['group']}]: {short}", flush=True)
                    with _lock:
                        _warned.add(ekey)
                continue

            except Exception as exc:
                last_exc = exc
                print(f"  [ERR] {model}: {str(exc)[:120]}", flush=True)
                continue

        # All available entries tried; loop will wait and retry
        if attempt < 3:
            wait = _soonest_wait()
            print(f"  [WAIT] All models cooling. Retrying in {max(wait,1):.0f}s "
                  f"(attempt {attempt+1}/3) …", flush=True)
            time.sleep(max(wait, 1) + 1)

    raise SystemExit(
        f"All models limit exhausted after retries. Last error: {last_exc}\n"
        "Keys tried: GEMINI_API_KEY, GROQ_API_KEY" +
        (", GROQ_API_KEY_2" if os.getenv("GROQ_API_KEY_2") else " (tip: add GROQ_API_KEY_2 for 2× TPM)") + "\n"
        "Please wait for rate limit reset or add API credits."
    )


# ── Public API ────────────────────────────────────────────────────────────────

def text_call(system: str, user: str, max_tokens: int = 512, **_kw) -> str:
    text, _ = _call(system, user, None, max_tokens)
    return text


def vision_call(system: str, images: list, text: str,
                max_tokens: int = 768, **_kw) -> str:
    result, _ = _call(system, text, images, max_tokens)
    return result


def active_model() -> str:
    return getattr(_thread_local, "last_model",
                   _ACTIVE_MODELS[0]["model"] if _ACTIVE_MODELS else "unknown")


def get_available_providers() -> list[str]:
    return [e["model"] for e in _ACTIVE_MODELS]
