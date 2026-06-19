"""Provider-agnostic LLM client for the claim pipeline.

Pick the provider with the LLM_PROVIDER env var:

    auto       (default) ping providers in order and use the first that responds
    gemini     Google Gemini   (free tier, vision)   — GEMINI_API_KEY
    groq       Groq Llama-4    (free tier, vision)   — GROQ_API_KEY
    anthropic  Anthropic Claude (paid API, vision)   — ANTHROPIC_API_KEY

Multi-threading: each thread can be assigned a specific provider via the
`provider` parameter on text_call()/vision_call(). When `provider=None` the
globally selected provider is used. If a call fails, automatic fallback tries
the next available provider and logs the switch to terminal + log file.

Optional model overrides: GEMINI_MODEL, GROQ_MODEL, ANTHROPIC_MODEL.
"""

import os
import threading

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_AUTO_ORDER = ["gemini", "groq", "anthropic"]

_CFG = {
    "gemini": {
        "key": "GEMINI_API_KEY",
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "kind": "openai",
    },
    "groq": {
        "key": "GROQ_API_KEY",
        "model": os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        "base_url": "https://api.groq.com/openai/v1",
        "kind": "openai",
    },
    "anthropic": {
        "key": "ANTHROPIC_API_KEY",
        "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        "base_url": None,
        "kind": "anthropic",
    },
}

_clients: dict = {}
_clients_lock = threading.Lock()
_active: str | None = None
_available: list[str] = []   # providers that passed ping, in order


def _has_key(provider: str) -> bool:
    return bool(os.getenv(_CFG[provider]["key"], "").strip())


def _client(provider: str):
    with _clients_lock:
        if provider not in _clients:
            cfg = _CFG[provider]
            if cfg["kind"] == "openai":
                from openai import OpenAI
                _clients[provider] = OpenAI(
                    api_key=os.getenv(cfg["key"]), base_url=cfg["base_url"]
                )
            else:
                import anthropic
                _clients[provider] = anthropic.Anthropic(
                    api_key=os.getenv(cfg["key"])
                )
        return _clients[provider]


def _retryable_excs() -> tuple:
    excs: list = []
    try:
        import openai
        excs += [
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ]
    except Exception:
        pass
    try:
        import anthropic
        excs += [
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        ]
    except Exception:
        pass
    return tuple(excs) or (Exception,)


# ── content builders ──────────────────────────────────────────────────────────

def _openai_user_content(text: str, images: list | None) -> list:
    parts: list = []
    for img in images or []:
        if img.get("exists") and img.get("base64_str"):
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img['base64_str']}"},
            })
            parts.append({"type": "text", "text": f"[Image ID: {img['image_id']}]"})
    parts.append({"type": "text", "text": text})
    return parts


def _anthropic_content(text: str, images: list | None) -> list:
    content: list = []
    for img in images or []:
        if img.get("exists") and img.get("base64_str"):
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img["base64_str"],
                },
            })
            content.append({"type": "text", "text": f"[Image ID: {img['image_id']}]"})
    content.append({"type": "text", "text": text})
    return content


# ── core call (with transient-only retry) ─────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_retryable_excs()),
    reraise=True,
)
def _call_once(provider: str, system: str, text: str, images: list | None,
               max_tokens: int, json_mode: bool) -> str:
    cfg = _CFG[provider]
    if cfg["kind"] == "openai":
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = _client(provider).chat.completions.create(
            model=cfg["model"],
            temperature=0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": _openai_user_content(text, images)},
            ],
            **kwargs,
        )
        return resp.choices[0].message.content or ""
    # anthropic
    resp = _client(provider).messages.create(
        model=cfg["model"],
        max_tokens=max_tokens,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": _anthropic_content(text, images)}],
    )
    return resp.content[0].text


def _call_with_fallback(preferred: str, system: str, text: str, images: list | None,
                        max_tokens: int, json_mode: bool) -> str:
    """Try preferred provider first, then fall back through available providers."""
    candidates = [preferred] + [p for p in _available if p != preferred]
    last_exc: Exception | None = None
    for provider in candidates:
        try:
            result = _call_once(provider, system, text, images, max_tokens, json_mode)
            if provider != preferred:
                # Switched provider mid-run — announce it
                model = _CFG[provider]["model"]
                msg = (f"  [PROVIDER SWITCH] {preferred} → {provider} "
                       f"({model}) — {type(last_exc).__name__}: {str(last_exc)[:80]}")
                print(msg, flush=True)
                try:
                    from code.utils.logger import write_provider_switch
                    write_provider_switch(
                        from_provider=preferred,
                        to_provider=provider,
                        reason=f"{type(last_exc).__name__}: {str(last_exc)[:140]}",
                        model=model,
                    )
                except Exception:
                    pass
            return result
        except Exception as exc:
            last_exc = exc
            print(f"  [WARN] {provider} failed: {type(exc).__name__}: {str(exc)[:80]}",
                  flush=True)
    raise RuntimeError(
        f"All providers failed. Last error from {candidates[-1]}: {last_exc}"
    )


# ── provider selection ────────────────────────────────────────────────────────

def _ping(provider: str) -> None:
    cfg = _CFG[provider]
    if cfg["kind"] == "openai":
        _client(provider).chat.completions.create(
            model=cfg["model"], max_tokens=4,
            messages=[{"role": "user", "content": "ping"}],
        )
    else:
        _client(provider).messages.create(
            model=cfg["model"], max_tokens=4,
            messages=[{"role": "user", "content": "ping"}],
        )


def select_provider(verbose: bool = True) -> str:
    """Resolve and cache the active provider. Raises SystemExit if none work."""
    global _active, _available
    if _active:
        return _active

    requested = os.getenv("LLM_PROVIDER", "auto").strip().lower()

    if requested and requested != "auto":
        if requested not in _CFG:
            raise SystemExit(f"Unknown LLM_PROVIDER={requested!r} "
                             f"(use one of: auto, {', '.join(_CFG)})")
        if not _has_key(requested):
            raise SystemExit(f"LLM_PROVIDER={requested} but {_CFG[requested]['key']} "
                             f"is not set in .env")
        _active = requested
        _available = [requested]
        if verbose:
            model = _CFG[requested]["model"]
            print(f"  [MODEL] {requested} → {model}", flush=True)
        return _active

    # auto: try each provider that has a key, in order
    errors: list[str] = []
    working: list[str] = []
    for p in _AUTO_ORDER:
        if not _has_key(p):
            continue
        try:
            _ping(p)
            working.append(p)
            if verbose:
                model = _CFG[p]["model"]
                label = "PRIMARY" if not working or len(working) == 1 else "FALLBACK"
                print(f"  [MODEL] {p} ({model}) — {label}", flush=True)
        except Exception as e:
            errors.append(f"{p}: {type(e).__name__}: {str(e)[:140]}")
            if verbose:
                print(f"  [MODEL] {p} — UNAVAILABLE ({type(e).__name__})", flush=True)

    if not working:
        if errors:
            raise SystemExit("No working LLM provider. Tried:\n  " + "\n  ".join(errors))
        raise SystemExit("No LLM API keys found — set GEMINI_API_KEY, GROQ_API_KEY, "
                         "or ANTHROPIC_API_KEY in .env")

    _active = working[0]
    _available = working

    if verbose and len(working) > 1:
        print(f"  [MODEL] Active provider: {_active} | Fallbacks: {working[1:]}",
              flush=True)
    elif verbose:
        print(f"  [MODEL] Active provider: {_active} (no fallbacks available)",
              flush=True)

    return _active


def active() -> str:
    return _active or select_provider(verbose=False)


def active_model() -> str:
    return _CFG[active()]["model"]


def get_available_providers() -> list[str]:
    """Return all working providers discovered during select_provider()."""
    return list(_available)


# ── public API used by strategies/nodes ───────────────────────────────────────

def text_call(system: str, user: str, max_tokens: int = 1024,
              json_mode: bool = True, provider: str | None = None) -> str:
    p = provider or active()
    return _call_with_fallback(p, system, user, None, max_tokens, json_mode)


def vision_call(system: str, images: list, text: str, max_tokens: int = 2048,
                json_mode: bool = True, provider: str | None = None) -> str:
    p = provider or active()
    return _call_with_fallback(p, system, text, images, max_tokens, json_mode)
