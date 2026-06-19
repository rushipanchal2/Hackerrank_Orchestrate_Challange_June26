# Claude Code fix prompt — paste this into `claude` (auto mode) at the repo root

> Run from `hackerrank-orchestrate-june26/` with the venv active.
> Paste everything in the fenced block below as a single message.

```
We have a bug: `python code/evaluate.py --strategy both --dataset sample` fails on EVERY
case with `RetryError[<Future ... raised BadRequestError>]`. The real 400 message is never
shown because tenacity retries the error and then wraps it. Fix the code so the real error
surfaces and the run succeeds. Make these changes, then verify.

ROOT CAUSE (already diagnosed — do not re-investigate, just fix):
1. The @retry decorators in `code/strategies/single_shot.py` and `code/graph/nodes.py` use
   `retry_if_exception_type((anthropic.RateLimitError, anthropic.APIStatusError))`.
   `BadRequestError`/`AuthenticationError`/`NotFoundError`/`PermissionDeniedError` are all
   subclasses of `APIStatusError`, so non-retryable 4xx client errors get retried 3x and the
   original message is hidden inside `RetryError`. This is why no message is visible.
2. `code/utils/image_loader.py` re-encodes JPEGs at quality 85 but never downsizes dimensions.
   The `test` dataset contains a 7908x5931 image (~10.5MB base64) that exceeds Anthropic's
   5MB-per-image limit and will return a real 400 on the full run.

FIX 1 — stop retrying client errors and surface the real message.
In BOTH `code/strategies/single_shot.py` and `code/graph/nodes.py`, change every @retry
decorator so it ONLY retries transient errors and re-raises the original exception:

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        )),
        reraise=True,
    )

`reraise=True` is required so the underlying error (with its message) propagates instead of
`RetryError`. Do NOT keep `anthropic.APIStatusError` in the retry tuple.

FIX 2 — downsize images before encoding so they stay under the API limit.
In `code/utils/image_loader.py`, after `img = Image.open(resolved).convert("RGB")`, resize so
the longest edge is at most 1568px (Anthropic's recommended max; keeps requests well under 5MB)
before saving to the JPEG buffer:

    MAX_EDGE = 1568
    if max(img.size) > MAX_EDGE:
        img.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)

Keep quality=85. This must not change behaviour for already-small images.

FIX 3 — add a fast preflight so the real failure is obvious immediately.
At the top of `main()` in `code/evaluate.py` (right after args are parsed and the API key is
loaded), add a single cheap API call that fails loudly with the real message before the full
loop runs:

    import anthropic
    try:
        anthropic.Anthropic().messages.create(
            model="claude-sonnet-4-6", max_tokens=8,
            messages=[{"role": "user", "content": "ok"}],
        )
    except anthropic.APIStatusError as e:
        print(f"\nPREFLIGHT FAILED ({type(e).__name__}, HTTP {e.status_code}): {e.message}\n")
        if e.status_code == 400 and "credit" in str(e).lower():
            print("-> Your Anthropic account has no usable credit / billing. "
                  "Add credits at console.anthropic.com -> Plans & Billing, then re-run.")
        raise SystemExit(1)

(If the most likely real cause — a 400 about credit balance — applies, this preflight will
print it in plain English instead of the opaque RetryError.)

FIX 4 — checkpointer is a context manager, not a saver (fixes a TypeError at compile()).
In `code/graph/graph.py`, `SqliteSaver.from_conn_string("checkpoints.db")` returns a
`_GeneratorContextManager`, not a saver, and it does NOT raise — so the bad object reaches
`g.compile()` and raises `TypeError: Invalid checkpointer ... Received _GeneratorContextManager`.
Build the saver from a sqlite3 connection directly:

    if checkpointer is None:
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
            checkpointer = SqliteSaver(conn)
        except Exception:
            try:
                from langgraph.checkpoint.memory import InMemorySaver as _MemSaver
            except ImportError:
                from langgraph.checkpoint.memory import MemorySaver as _MemSaver
            checkpointer = _MemSaver()

ACCEPTANCE CRITERIA:
- `grep -n "APIStatusError" code/strategies/single_shot.py code/graph/nodes.py` returns nothing
  inside any retry= tuple.
- Running `python code/evaluate.py --strategy A --dataset sample --delay 0.5` either completes,
  or prints a clear single-line error (e.g. the credit-balance message) — NOT a `RetryError`.
- For each fix, show me the git diff of the edited files when done.
```
