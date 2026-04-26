---

## Scoring candidate solutions — sequence log-probability

- Self-refinement (and best-of-N, self-consistency) all need a way to **compare two candidate sequences** and pick the better one. We need a scoring function.
- The simplest and **cheapest** score — one we get **for free** from generation — is the model's own log-probability of the sequence:

$$
\log \pi_\theta(y \mid x) \;=\; \sum_{t=1}^{|y|} \log \pi_\theta\!\left(y_t \mid x,\, y_{<t}\right)
$$

- **Intuition**: "how confident was the model, on average, at each step of producing this sequence." A higher (less negative) log-prob means the sequence is more **natural** under the model.
- Every per-token $\log \pi_\theta(y_t \mid x, y_{<t})$ was already computed by the forward pass during generation — scoring is essentially **free**.

- In code:

```python
logits    = model(input_ids).logits[:, :-1, :]            # shift for next-token
log_probs = F.log_softmax(logits, dim=-1)
targets   = input_ids[:, 1:].unsqueeze(-1)
token_lp  = torch.gather(log_probs, dim=-1, index=targets).squeeze(-1)  # (B, L-1)
seq_logp  = (token_lp * completion_mask).sum(dim=-1)      # (B,) — the score
```

---

## Length normalization — and what log-prob cannot measure

- **Problem**: every per-token log-prob is negative, so the raw sum is **systematically biased toward shorter sequences**. A terse answer always has a higher raw log-prob than a detailed chain-of-thought that reaches the same conclusion.
- **Fix**: divide by sequence length to get the **mean log-prob** — equivalently, the log of the geometric mean token probability:

$$
\text{score}(y \mid x) \;=\; \frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta\!\left(y_t \mid x,\, y_{<t}\right)
$$

- A related, equivalent quantity is **perplexity**: $\text{PPL}(y) = \exp\!\left(-\tfrac{1}{|y|} \sum_t \log \pi_\theta(y_t \mid x, y_{<t})\right)$. Lower perplexity = the model is "less surprised" per token by the sequence.

<center>
    <img src="images/seq_logprob_scoring.svg" width=1000>
</center>

- **Caveat 1 — log-prob is confidence, not correctness**. A fluent, confident-but-wrong answer often scores higher than a hesitant, correct one. The model scores how much *itself* "believes" the sequence, not whether the sequence is right.
- **Caveat 2 — better options exist when available.** When possible, combine log-prob with (a) a **verifier** for tasks with ground truth (math, code), (b) a **reward model** for open-ended tasks, or (c) **LLM-as-judge** — the model scoring its own candidates with an explicit rubric. Log-prob is the cheap fallback, not the best signal.

---
