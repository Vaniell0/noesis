"""Portability metrics for A0.6 (intra-model state swap) and A0.7 tier-1
(inter-checkpoint state transfer).

Five metrics, each answering a specific question about a state-swap
experiment where model M is fed prompt B but with the WKV state that
model M produced on prompt A ("cross" continuation):

1. ``task_lexicon_hit_rate``  — "does the *content* of A leak into
   B's continuation?" Content-word overlap between donor prompt text
   and generated continuation text.
2. ``topk_jaccard``            — "at token level, how much do the
   next-token distributions overlap?" Graded replacement for
   ``argmax_flip`` in ``../A0_state_probe/a05_intervene.py``.
3. ``alignment_vs_donor``     — "does the cross continuation pull
   toward A or toward B?" Signed alignment from cumulative KL to each
   clean baseline. −1 = fully aligned with donor A, +1 = fully aligned
   with recipient B, 0 = equidistant.
4. ``first_divergence_step``  — "immediate or delayed?" First step at
   which the cross continuation's token diverges from the clean-B
   baseline. Distinguishes state-carrying-content (early divergence)
   from state-mostly-noise (late or no divergence).
5. ``surface_garble``          — coherence sanity: unique-token ratio,
   average sentence length. Numeric replacement for the subjective
   "no obvious garbage" rubric in the plan's success criteria.

All functions are stateless and operate on already-materialised outputs
(decoded strings, token id lists, per-step logits vectors, cumulative
KL scalars). The runner in ``a06_run.py`` / ``a07_tier1_run.py`` owns
the model forwards and hands finished artefacts here.

Design constraint: **tokenizer-agnostic content metrics.** Task-lexicon
and surface-garble work on decoded strings, so the same code applies
across RWKV-7 World vocab and any future tokenizer swap. Logit-level
metrics accept torch tensors; they cast to fp32 CPU before comparison.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

import torch


# --------------------------------------------------------------------------- #
# Content-word extraction (used by 1 and 5)
# --------------------------------------------------------------------------- #

# Small stop-word set. The task-lexicon signal is meant to catch *content*
# words (nouns, verbs, distinctive adjectives) that appeared in the donor
# prompt and re-appeared in the continuation. Function words carry no
# portability signal. This list intentionally stays small — for A0.6/A0.7
# the prompts are drawn from open English sources; add more entries when
# extending to other languages.
_STOPWORDS_EN = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "in",
    "on", "at", "to", "from", "by", "for", "with", "without", "into", "onto",
    "up", "down", "out", "over", "under", "again", "further", "as", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "not", "no", "nor", "so", "yes",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "them", "his", "her", "their", "our", "your", "my", "we", "you", "i",
    "who", "whom", "what", "which", "when", "where", "why", "how", "than",
    "here", "there", "just", "very", "much", "more", "most", "some", "any",
    "all", "each", "every", "few", "many", "own", "same", "such", "only",
    "too", "also", "would", "could", "should", "shall", "will", "may",
    "might", "must", "can", "one", "two", "three",
})

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _content_words(text: str, min_len: int = 3) -> "set[str]":
    """Extract lower-cased content-word tokens from ``text``.

    Rules: keep alphanumeric runs of length ≥ ``min_len``, drop
    stopwords. Punctuation and short function words are removed. The
    resulting set is treated as unordered for hit-rate computations.
    """
    words = _TOKEN_RE.findall(text.lower())
    return {w for w in words if len(w) >= min_len and w not in _STOPWORDS_EN}


# --------------------------------------------------------------------------- #
# 1. task_lexicon_hit_rate
# --------------------------------------------------------------------------- #

def task_lexicon_hit_rate(donor_text: str, generated_text: str) -> float:
    """Fraction of donor's content-word lexicon that reappears in
    ``generated_text``.

    Returns a value in [0, 1]. Symmetric-ish semantics: a high hit-rate
    means the continuation "talks about" the same content words as the
    donor prompt. For A0.6 the runner computes this three times:

    - ``hit_rate(donor=A, generated=cross_A→B)``  — the load-bearing
      one. If elevated vs baseline, donor A's content is bleeding into
      B's continuation.
    - ``hit_rate(donor=B, generated=cross_A→B)``  — recipient control.
      A cross that ignores the injected state should still track B's
      lexicon since B was the actual prompt.
    - ``hit_rate(donor=A, generated=clean_B)``   — the null baseline.
      Any overlap here is chance-level given the vocabularies.

    Denominator is ``|content(donor_text)|``. If the donor has no
    content words the metric is undefined; we return 0.0 in that case
    so the runner can filter.
    """
    donor_set = _content_words(donor_text)
    if not donor_set:
        return 0.0
    gen_set = _content_words(generated_text)
    hits = len(donor_set & gen_set)
    return hits / len(donor_set)


# --------------------------------------------------------------------------- #
# 2. topk_jaccard
# --------------------------------------------------------------------------- #

def topk_jaccard(
    logits_a: torch.Tensor, logits_b: torch.Tensor, k: int = 10
) -> float:
    """Jaccard overlap of the top-``k`` token ids in two logit vectors.

    Returns ``|top_k(a) ∩ top_k(b)| / k`` as a float in [0, 1]. Both
    inputs are cast to fp32 CPU before ``topk`` so comparison is
    dtype-agnostic.

    This is the graded generalisation of ``argmax_flip`` from
    ``a05_intervene.py``: ``argmax_flip == 1 - topk_jaccard(k=1)`` up to
    the numeric detail that Jaccard of two singleton sets is 0 or 1.
    Use k=5 or k=10 in A0.6; k=50 was proposed but drops most of the
    signal into the tail.

    Applied per step, giving a 1-D trajectory of overlap values whose
    mean is a graded "did the cross continuation pick similar tokens to
    the clean-B baseline" score.
    """
    if k < 1:
        raise ValueError(f"topk_jaccard: k must be ≥ 1, got {k}")
    a = logits_a.reshape(-1).to(torch.float32).cpu()
    b = logits_b.reshape(-1).to(torch.float32).cpu()
    if a.shape != b.shape:
        raise ValueError(
            f"topk_jaccard: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
        )
    k_eff = min(k, a.numel())
    top_a = set(torch.topk(a, k_eff).indices.tolist())
    top_b = set(torch.topk(b, k_eff).indices.tolist())
    return len(top_a & top_b) / k_eff


def topk_jaccard_trajectory(
    logits_seq_a: Sequence[torch.Tensor],
    logits_seq_b: Sequence[torch.Tensor],
    k: int = 10,
) -> "list[float]":
    """Per-step ``topk_jaccard`` across matched logit sequences.

    Truncates to the shorter of the two if they differ in length so a
    partial run is still comparable. Empty result means no overlap
    could be computed (both sequences empty).
    """
    n = min(len(logits_seq_a), len(logits_seq_b))
    return [topk_jaccard(logits_seq_a[i], logits_seq_b[i], k=k) for i in range(n)]


# --------------------------------------------------------------------------- #
# 3. alignment_vs_donor
# --------------------------------------------------------------------------- #

def alignment_vs_donor(
    cross_cum_kl_to_donor: float,
    cross_cum_kl_to_recipient: float,
    eps: float = 1e-9,
) -> "dict[str, float]":
    """Signed alignment of the cross continuation between donor and
    recipient baselines.

    Given cumulative KLs from the cross continuation to (a) the donor
    prompt's clean continuation (state_A → decode_A) and (b) the
    recipient prompt's clean continuation (state_B → decode_B), returns
    a signed score in [-1, +1]:

    - ``alignment = -1`` — cross is indistinguishable from clean-A;
      the injected state fully dominated. State carries A's semantic
      content into B's prompt slot.
    - ``alignment = +1`` — cross matches clean-B; the model discarded
      the injected state and used B's prompt to reconstruct B's
      continuation. State is not doing portable work.
    - ``alignment = 0`` — cross is equidistant to both baselines.
      Either a novel mixture emerged or state and prompt both
      contribute equally.

    Formula: ``(kl_to_donor − kl_to_recipient) / (kl_to_donor +
    kl_to_recipient + eps)``. Direction chosen so "aligned with donor"
    is negative, matching the intuitive "state pulls toward its origin"
    reading.

    Returns a dict with the two input KLs and the derived alignment so
    the runner can log both raw and derived values.
    """
    d = float(cross_cum_kl_to_donor)
    r = float(cross_cum_kl_to_recipient)
    denom = d + r + eps
    return {
        "kl_to_donor": d,
        "kl_to_recipient": r,
        "alignment": (d - r) / denom,
    }


# --------------------------------------------------------------------------- #
# 4. first_divergence_step
# --------------------------------------------------------------------------- #

def first_divergence_step(
    tokens_ref: Sequence[int],
    tokens_test: Sequence[int],
) -> "Optional[int]":
    """First 0-indexed step at which ``tokens_test`` diverges from
    ``tokens_ref``.

    Returns ``None`` if the two sequences agree over the whole
    compared range. Comparison length is ``min(len(ref), len(test))``.

    Interpretation for A0.6:

    - Small value (0–3) — state affects the very next tokens; the
      injected state's influence is immediate, dominating even the
      first-word choice.
    - Middle value (4–20) — state has bandwidth to influence the
      middle of the continuation once the model settles into a
      trajectory.
    - ``None`` or near the horizon — state was overridden by the
      prompt almost immediately; no lasting effect.

    The value is a diagnostic complement to ``alignment_vs_donor``:
    alignment tells us *where* the trajectory ended up; first
    divergence tells us *how quickly* it got there.
    """
    n = min(len(tokens_ref), len(tokens_test))
    for i in range(n):
        if tokens_ref[i] != tokens_test[i]:
            return i
    return None


# --------------------------------------------------------------------------- #
# 5. surface_garble
# --------------------------------------------------------------------------- #

_SENT_SPLIT_RE = re.compile(r"[.!?]+\s+|[.!?]+$")


def surface_garble(text: str) -> "dict[str, float]":
    """Numeric coherence proxy for a generated continuation.

    Returns three signals:

    - ``unique_token_ratio``  — ``|unique_words| / |total_words|`` on
      the whitespace-split, lower-cased sequence. Very low
      (< 0.15) means the model is stuck in a repetition loop —
      classic broken-state failure mode.
    - ``avg_sentence_length`` — mean of word-per-sentence counts,
      splitting on ``.!?``. Very small (< 2) or very large (> 60)
      suggests structural breakdown.
    - ``coherence_flag``      — 1.0 if both above bounds are within
      ``[0.20, 0.95]`` for unique ratio and ``[3, 50]`` for sentence
      length; else 0.0. Coarse-grained but reproducible replacement
      for a subjective "does it look OK" judgement.

    Not a language-quality metric — a fluent paragraph passes just as
    easily as a fluent nonsense paragraph. But injured-state failure
    modes (loops, single-token spam, no punctuation) are exactly the
    ones this catches cheaply.
    """
    if not text.strip():
        return {"unique_token_ratio": 0.0, "avg_sentence_length": 0.0,
                "coherence_flag": 0.0}

    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return {"unique_token_ratio": 0.0, "avg_sentence_length": 0.0,
                "coherence_flag": 0.0}

    unique_ratio = len(set(w.lower() for w in words)) / len(words)

    sentences = [s for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    if sentences:
        sent_lengths = [len([w for w in re.split(r"\s+", s.strip()) if w])
                        for s in sentences]
        avg_sent_len = sum(sent_lengths) / len(sent_lengths)
    else:
        avg_sent_len = float(len(words))

    coherent = (0.20 <= unique_ratio <= 0.95) and (3.0 <= avg_sent_len <= 50.0)
    return {
        "unique_token_ratio": float(unique_ratio),
        "avg_sentence_length": float(avg_sent_len),
        "coherence_flag": 1.0 if coherent else 0.0,
    }


__all__ = [
    "task_lexicon_hit_rate",
    "topk_jaccard",
    "topk_jaccard_trajectory",
    "alignment_vs_donor",
    "first_divergence_step",
    "surface_garble",
]
