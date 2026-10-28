#!/usr/bin/env python3
"""Does Clover-35B-A3B-Preview on SGLang behave? Checks the model listing, thinking off and on (reasoning kept
out of the answer), a tool call and its round trip, streaming and Dutch against a running server, optionally
plus needle-in-a-haystack retrieval at long context.

  tools/check.py                         # functional checks
  tools/check.py --needle 32000 131072   # also hide a key in N tokens of filler and ask for it back
  tools/check.py --url http://10.0.0.158:8888

Exit code 1 if anything fails. Standard library only.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request


class Server:
    def __init__(self, url: str, model: str):
        self.base, self.model = url.rstrip("/"), model

    def _open(self, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data, {"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as err:
            sys.exit(f"{self.base}{path}: HTTP {err.code} {err.read().decode(errors='replace')[:400]}")
        except urllib.error.URLError as err:
            sys.exit(f"{self.base}{path}: {err.reason} (is the server up? docker compose ps)")

    def models(self) -> list[str]:
        payload = json.load(self._open("/v1/models"))
        return [item.get("id", "") for item in payload.get("data", [])]

    def ask(self, prompt: str, **body) -> dict:
        """One non-streamed turn; returns the message plus usage."""
        reply = json.load(self._open("/v1/chat/completions",
                                      {"model": self.model, "messages": [{"role": "user", "content": prompt}], **body}))
        return {**reply["choices"][0]["message"], "usage": reply.get("usage", {})}

    def stream(self, prompt: str, **body) -> tuple[str, str]:
        """One streamed turn; returns (reasoning, content) as they arrived in the deltas."""
        parts = {"reasoning": [], "content": []}
        body = {"model": self.model, "stream": True, "messages": [{"role": "user", "content": prompt}], **body}
        for raw in self._open("/v1/chat/completions", body):
            event = raw.decode().removeprefix("data:").strip()
            if not event or event == "[DONE]":
                continue
            for choice in json.loads(event).get("choices", []):
                delta = choice.get("delta", {})
                parts["reasoning"].append(delta.get("reasoning") or delta.get("reasoning_content") or "")
                parts["content"].append(delta.get("content") or "")
        return "".join(parts["reasoning"]), "".join(parts["content"])


def reasoning(msg: dict) -> str:
    return (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()


def clean(text: str | None) -> str:
    return (text or "").strip()


def model_listed(srv: Server):
    names = srv.models()
    return srv.model in names, f"ids {names}"


OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def thinking_off(srv: Server):
    msg = srv.ask("What is 17 + 25? Digits only.", max_tokens=4096, temperature=0, **OFF)
    answer = clean(msg["content"])
    ok = "42" in answer and "<think>" not in answer and not reasoning(msg)
    return ok, f"answer {answer!r}, reasoning {len(reasoning(msg))} chars"


def thinking_on(srv: Server):
    """Thinking is on by default: the reasoning must land in reasoning_content, not in the answer."""
    msg = srv.ask("A notebook costs 4 euros and a pen costs 1.50 euros. "
                  "How much do 3 notebooks and 2 pens cost, in euros? Reply with the number only.",
                  max_tokens=16384, temperature=0.6)
    answer = clean(msg["content"])
    ok = "15" in answer and "<think>" not in answer and "</think>" not in answer and bool(reasoning(msg))
    return ok, f"answer {answer!r}, reasoning {len(reasoning(msg))} chars"


EVENT_TOOL = {"type": "function", "function": {
    "name": "create_event",
    "description": "Create a calendar event.",
    "parameters": {"type": "object", "required": ["title", "date"], "properties": {
        "title": {"type": "string"},
        "date": {"type": "string", "description": "ISO date, YYYY-MM-DD"}}}}}
EVENT_ASK = "Add a calendar event titled Sprint Review on 12 March 2027."


def tool_call(srv: Server):
    msg = srv.ask(EVENT_ASK, tools=[EVENT_TOOL], tool_choice="auto", max_tokens=4096, temperature=0, **OFF)
    calls = msg.get("tool_calls") or []
    if not calls:
        return False, f"no tool call; content {clean(msg.get('content'))[:160]!r}"
    fn = calls[0].get("function", {})
    raw = fn.get("arguments", {})
    args = json.loads(raw) if isinstance(raw, str) else raw
    title = str(args.get("title", ""))
    ok = fn.get("name") == "create_event" and args.get("date") == "2027-03-12" and "Sprint Review" in title
    return ok, f"{fn.get('name')}({json.dumps(args, ensure_ascii=False)})"


def tool_round_trip(srv: Server):
    """The tool result goes back as a tool message; the model must answer from it, not call again."""
    call = {"id": "call_1", "type": "function",
            "function": {"name": "create_event", "arguments": json.dumps({"title": "Sprint Review", "date": "2027-03-12"})}}
    messages = [{"role": "user", "content": EVENT_ASK},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"ok": True, "event_id": "EV-4471"})}]
    reply = json.load(srv._open("/v1/chat/completions", {"model": srv.model, "messages": messages, "tools": [EVENT_TOOL],
                                                         "max_tokens": 4096, "temperature": 0, **OFF}))
    msg = reply["choices"][0]["message"]
    answer = clean(msg.get("content"))
    return "EV-4471" in answer and not msg.get("tool_calls"), f"answer {answer[:90]!r}"


def streaming(srv: Server):
    think, answer = srv.stream("What is the capital of Belgium? One word.", max_tokens=4096, temperature=0, **OFF)
    text = clean(answer)
    ok = "Brussels" in text and "<think>" not in text + think
    return ok, f"reasoning {len(think)} chars, answer {text[:70]!r}"


def dutch(srv: Server):
    """Answers in Dutch. The fact itself is model knowledge, so it is shown, not checked."""
    msg = srv.ask("Wat is de hoofdstad van de provincie West-Vlaanderen? Antwoord in één Nederlandse zin.",
                  max_tokens=4096, temperature=0, **OFF)
    answer = clean(msg["content"])
    return "hoofdstad" in answer.lower(), f"answer {answer[:90]!r}"


def speed(srv: Server):
    """Output tokens per second for one request, prefill included. Shown, not checked."""
    start = time.time()
    msg = srv.ask("Write a Python function that parses ISO 8601 durations like P3DT4H5M into seconds, "
                  "with a docstring and three doctests.", max_tokens=1024, temperature=0, **OFF)
    n, took = msg["usage"].get("completion_tokens", 0), time.time() - start
    return None if not n else True, f"{n} tokens in {took:.1f} s = {n / took:.1f} tok/s"


def needle(srv: Server, tokens: int):
    """Hide a key among ~`tokens` tokens of synthetic service logs, at a random depth, and ask for it."""
    rng = random.Random(tokens)
    key = f"{rng.randrange(10**7, 10**8)}"
    lines = [f"2026-10-08T{rng.randrange(24):02}:{rng.randrange(60):02}:{rng.randrange(60):02}Z "
             f"worker-{rng.randrange(64):02} {rng.choice(('GET', 'POST', 'PUT'))} /api/v2/"
             f"{rng.choice(('orders', 'users', 'stock', 'invoices'))}/{rng.randrange(10**6)} -> "
             f"{rng.choice((200, 200, 200, 201, 404, 503))} in {rng.randrange(2, 900)} ms"
             for _ in range(tokens // 40)]
    depth = rng.uniform(0.1, 0.9)
    lines.insert(int(len(lines) * depth), f"NOTICE maintenance key for tonight is {key}")
    start = time.time()
    msg = srv.ask("\n".join(lines) + "\n\nWhat is tonight's maintenance key? Reply with the key only.",
                  max_tokens=4096, temperature=0, **OFF)
    got, n = clean(msg["content"]), msg["usage"].get("prompt_tokens", 0)
    return key in got, f"{n:,} tokens, depth {depth:.0%}, {time.time() - start:.0f} s: expected {key}, got {got!r}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("CLOVER_URL", "http://127.0.0.1:8888"))
    ap.add_argument("--model", default=os.environ.get("CLOVER_MODEL", "clover"))
    ap.add_argument("--needle", type=int, nargs="*", default=[], metavar="TOKENS")
    args = ap.parse_args()

    srv = Server(args.url, args.model)
    checks = [(fn.__name__.replace("_", " "), fn) for fn in
              (model_listed, thinking_off, thinking_on, tool_call, tool_round_trip, streaming, dutch, speed)]
    checks += [(f"needle {n:,}", lambda s, n=n: needle(s, n)) for n in args.needle]
    failures = 0
    for name, check in checks:
        ok, detail = check(srv)
        failures += ok is False
        print(f"{'skip' if ok is None else 'ok  ' if ok else 'FAIL'}  {name:20} {detail}", flush=True)
    sys.exit(failures > 0)


if __name__ == "__main__":
    main()
