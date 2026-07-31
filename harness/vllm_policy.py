"""vLLM-client policy for the hierarchical loop (RESEARCH_PLAN Phase 1).

The training policy (Qwen3-4B-Instruct-2507) is served by a local vLLM
OpenAI-compatible server; this policy is just an HTTP client behind the same
`act(ctx) -> {think, subgoal}` interface as ScriptedPolicy / QwenPolicy. It is
**torch-free** (stdlib urllib only), so it lives in the harness env and never imports
the serving stack — generation happens in the separate `vllm` env / server process.

Robust like the other policies: pulls the first JSON object out of the reply
(`extract_decision`) and falls back to a safe `explore` on any parse or HTTP failure.
Records per-turn stats — including completion-token count and generation seconds — so
the Phase-1 rollout batch can report the matched-compute axis (tokens) and throughput
(RESEARCH_PLAN §7.1).
"""
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List

from agent import extract_decision

_FORMAT_REMINDER = (
    'Respond with ONE JSON object only, e.g. '
    '{"think": "need wood for a table", '
    '"subgoal": {"name": "mine", "args": {"resource": "wood", "count": 3}}}'
)


class VLLMPolicy:
    """OpenAI-compatible chat client for a local vLLM server. Set temperature>0 and
    vary `seed` across rollouts for RL diversity later."""

    def __init__(self, model: str = "Qwen/Qwen3-4B-Instruct-2507",
                 base_url: str = "http://localhost:8000/v1",
                 temperature: float = 0.7, max_tokens: int = 256,
                 timeout: float = 120.0):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.stats = {"turns": 0, "parsed": 0, "fallback": 0, "http_error": 0,
                      "gen_tokens": 0, "prompt_tokens": 0, "gen_seconds": 0.0}
        self.raws: List[str] = []

    def _chat(self, system_prompt: str, user_prompt: str,
              append_reminder: bool = True) -> str:
        user = user_prompt + ("\n\n" + _FORMAT_REMINDER if append_reminder else "")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        self.stats["gen_seconds"] += time.time() - t0
        usage = body.get("usage", {})
        self.stats["gen_tokens"] += int(usage.get("completion_tokens", 0))
        self.stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        return body["choices"][0]["message"]["content"]

    def act(self, ctx) -> Dict[str, Any]:
        self.stats["turns"] += 1
        try:
            raw = self._chat(ctx.system_prompt, ctx.turn_prompt)
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as e:
            self.stats["http_error"] += 1
            self.stats["fallback"] += 1
            self.raws.append(f"<http_error: {e}>")
            return {"think": f"server error ({e}); exploring",
                    "subgoal": {"name": "explore", "args": {}}, "_raw": ""}
        self.raws.append(raw)
        try:
            decision = extract_decision(raw)
            if not isinstance(decision, dict) or "subgoal" not in decision:
                raise ValueError("no 'subgoal' key")
            self.stats["parsed"] += 1
        except ValueError as e:
            self.stats["fallback"] += 1
            decision = {"think": f"unparseable ({e}); exploring",
                        "subgoal": {"name": "explore", "args": {}}}
        decision["_raw"] = raw
        return decision
