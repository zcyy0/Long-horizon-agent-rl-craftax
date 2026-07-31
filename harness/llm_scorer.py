"""Candidate scorer: per-candidate length-normalized log-prob ℓ(h,a) from the vLLM server
(REVISED_RESEARCH_PLAN §7.5). The shared A+B substrate — the frozen reference and the PPO
policy are BOTH categorical policies over the grounded candidate set, and that categorical
needs ℓ for every candidate, not just the one taken.

Mechanism: tokenize (system,user)→prompt and (system,user,assistant=completion)→full with the
LOCAL tokenizer (exact span, and identical tokenization to train_ppo_actor so rollout/train
log-probs stay consistent), then one batched `/v1/completions` request with `prompt_logprobs`
over the full token sequences; sum the log-probs on the completion span and length-normalize.

vLLM 0.24 note: `prompt_logprobs` forces `skip_reading_prefix_cache=True`, so each scored
sequence re-prefills the shared prompt (a latency cost, not an error) — scripts/scoring_probe_demo
measures it. Runs in the craftax venv (transformers is CPU-only here; no model is loaded, just
the tokenizer + HTTP).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Sequence


class ScoringError(RuntimeError):
    pass


class LLMScorer:
    def __init__(self, model: str = "Qwen/Qwen3-4B-Instruct-2507",
                 base_url: str = "http://localhost:8000/v1", timeout: float = 180.0):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = model
        self.url = base_url.rstrip("/") + "/completions"
        self.timeout = timeout
        self.stats = {"calls": 0, "scoring_prompt_tokens": 0, "score_seconds": 0.0}

    # --- tokenization -------------------------------------------------------
    # Render the template to TEXT, then tokenize — the same two-step train_ppo_actor.build_ids
    # and train_distill.cand_nll use, so scored and trained token sequences are identical.
    # (Not `apply_chat_template(tokenize=True)`: on transformers 5.x that returns a
    # BatchEncoding, not a token list, which silently makes every span empty.)
    def _ids(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _prompt_ids(self, system: str, user: str) -> List[int]:
        return self._ids(self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False))

    def _full_ids(self, system: str, user: str, completion: str) -> List[int]:
        return self._ids(self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user},
             {"role": "assistant", "content": completion}],
            add_generation_prompt=False, tokenize=False))

    # --- HTTP ---------------------------------------------------------------
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    @staticmethod
    def _tok_logprob(entry: Any, tid: int) -> float:
        """Pull the log-prob of token `tid` from one prompt_logprobs list element, tolerant of
        vLLM response shapes: {"<id>": {"logprob": x}} or {"<id>": x} or {id: Logprob}."""
        if entry is None:
            raise ScoringError("prompt_logprobs entry is None inside the completion span")
        v = entry.get(str(tid), entry.get(tid))
        if v is None:
            raise ScoringError(f"token {tid} absent from its prompt_logprobs entry")
        if isinstance(v, dict):
            return float(v["logprob"])
        return float(v)

    def score(self, system: str, user: str, completions: Sequence[str],
              return_raw: bool = False) -> List[Dict[str, Any]]:
        """Length-normalized ℓ for each completion given the same (system,user) prompt.
        Returns [{ell, sum_logprob, n_tokens}], one per completion, in input order."""
        if not completions:
            return []
        prompt_ids = self._prompt_ids(system, user)
        lp = len(prompt_ids)
        seqs, spans = [], []
        for c in completions:
            full = self._full_ids(system, user, c)
            seqs.append(full)
            spans.append((lp, len(full)))         # completion span = [lp, len(full))
        payload = {"model": self.model, "prompt": seqs, "max_tokens": 1,
                   "temperature": 0.0, "prompt_logprobs": 0}
        self.stats["calls"] += 1
        t0 = time.time()
        try:
            body = self._post(payload)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ScoringError(f"scoring request failed: {e}") from e
        self.stats["score_seconds"] += time.time() - t0
        self.stats["scoring_prompt_tokens"] += sum(len(s) for s in seqs)

        choices = sorted(body["choices"], key=lambda c: c.get("index", 0))
        if len(choices) != len(completions):
            raise ScoringError(f"got {len(choices)} choices for {len(completions)} completions")
        out = []
        for i, ch in enumerate(choices):
            plps = ch.get("prompt_logprobs")
            if plps is None:
                raise ScoringError("response has no prompt_logprobs — is the field supported?")
            lo, hi = spans[i]
            total = sum(self._tok_logprob(plps[pos], seqs[i][pos]) for pos in range(lo, hi))
            n = max(hi - lo, 1)
            rec = {"ell": total / n, "sum_logprob": total, "n_tokens": hi - lo}
            if return_raw:
                rec["_span_ids"] = seqs[i][lo:hi]
            out.append(rec)
        return out
