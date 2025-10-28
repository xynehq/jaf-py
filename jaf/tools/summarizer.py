"""
Summarizer tools for JAF.

Capabilities:
- summarize_text: Summarize long text using either a context-provided LLM/summarizer or a local extractive method.
  - method='auto' tries context first (llm/summarizer) then falls back to extractive frequency-based summarization
  - method='llm' forces using context-backed summarizer/LLM if available
  - method='extractive' forces local extractive summarization
- Supports chunking large inputs and two-pass summarization (summarize chunks, then summarize the summaries)
- Optional bullet point output

All tools return JSON strings for structured consumption.

Context expectations (any one is sufficient for LLM-based summarization):
- context.summarizer with .summarize(text, max_words|target_sentences) or .complete(prompt)
- context.llm, context.llm_client, or context.model with .summarize or .complete
"""

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from ..core.tools import function_tool


# ----------------------------
# Helpers: tokenization & scoring
# ----------------------------

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

_STOPWORDS = {
    # English minimal stopword set (can be extended or language-specific in the future)
    "a","an","the","and","or","but","if","then","than","that","this","those","these","of","to","in","on","at","for",
    "from","by","with","as","is","am","are","was","were","be","been","being","it","its","into","over","after","before",
    "up","down","out","off","so","no","not","too","very","can","will","just","do","does","did","doing","have","has","had",
    "having","you","your","yours","he","she","they","we","i","me","him","her","them","us","my","our","their","yourselves",
    "himself","herself","themselves","ourselves"
}


def _split_sentences(text: str) -> List[str]:
    # Conservative sentence splitting; preserve punctuation
    text = text.strip()
    if not text:
        return []
    sents = _SENT_SPLIT_RE.split(text)
    # Clean and keep non-empty
    return [s.strip() for s in sents if s and s.strip()]


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _score_sentences(sentences: List[str]) -> List[Tuple[int, float]]:
    # Frequency-based scoring: sum of non-stopword token frequencies per sentence
    freq: Dict[str, int] = {}
    for s in sentences:
        for w in _word_tokens(s):
            if w in _STOPWORDS or len(w) == 1:
                continue
            freq[w] = freq.get(w, 0) + 1

    if not freq:
        # Avoid division by zero; give equal scores
        return [(i, 1.0) for i in range(len(sentences))]

    max_f = max(freq.values())
    # Normalize frequencies and score sentences
    norm_freq = {w: f / max_f for w, f in freq.items()}

    scores: List[Tuple[int, float]] = []
    for i, s in enumerate(sentences):
        s_score = 0.0
        words = _word_tokens(s)
        for w in words:
            if w in norm_freq:
                s_score += norm_freq[w]
        # normalize by length to prefer dense sentences
        length = max(5, len(words))
        s_score = s_score / length
        scores.append((i, s_score))
    return scores


def _extractive_summarize(
    text: str,
    target_sentences: int = 5,
    bullets: bool = False
) -> Dict[str, Any]:
    sentences = _split_sentences(text)
    if not sentences:
        return {"summary": "", "sentences": 0, "bullets": [] if bullets else None}

    if target_sentences <= 0:
        target_sentences = 1

    if len(sentences) <= target_sentences:
        summ_sents = sentences
    else:
        scored = _score_sentences(sentences)
        # pick top-k by score, then reorder by original position to preserve narrative
        top = sorted(scored, key=lambda x: x[1], reverse=True)[:target_sentences]
        chosen_idx = sorted([i for i, _ in top])
        summ_sents = [sentences[i] for i in chosen_idx]

    summary = " ".join(summ_sents)
    result: Dict[str, Any] = {"summary": summary, "sentences": len(summ_sents)}
    if bullets:
        result["bullets"] = [s.strip() for s in summ_sents]
    return result


def _chunk_text_by_chars(text: str, max_chars: int) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # try to cut at a sentence boundary
        slice_text = text[start:end]
        # if we didn't end on punctuation, extend to next sentence end if available
        if end < len(text):
            punct_pos = re.search(r'[.!?]\s', text[end:end+200] if end + 200 <= len(text) else text[end:])
            if punct_pos:
                end += punct_pos.end()
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


async def _summarize_via_context(
    text: str,
    context: Any,
    target_sentences: int = 5,
    max_words: Optional[int] = None
) -> Optional[str]:
    """
    Try to use a context-provided LLM/summarizer client.
    Common shapes:
    - context.summarizer.summarize(text, max_words=?, target_sentences=?)
    - context.llm.complete(prompt) or context.llm.summarize(text)
    - context.llm_client or context.model with summarize/complete
    """
    if context is None:
        return None

    # 1) summarizer client
    for attr in ("summarizer", "summary_client"):
        client = getattr(context, attr, None)
        if client:
            for m in ("summarize", "summarise"):
                method = getattr(client, m, None)
                if callable(method):
                    try:
                        kw = {}
                        if max_words:
                            kw["max_words"] = max_words
                        kw["target_sentences"] = target_sentences
                        res = method(text, **kw)
                        if hasattr(res, "__await__"):
                            res = await res  # type: ignore
                        if isinstance(res, str):
                            return res
                        if isinstance(res, dict) and "summary" in res:
                            return str(res["summary"])
                    except Exception:
                        pass

    # 2) LLM client with summarize or complete
    for attr in ("llm", "llm_client", "model"):
        client = getattr(context, attr, None)
        if client:
            # direct summarize
            for m in ("summarize", "summarise"):
                method = getattr(client, m, None)
                if callable(method):
                    try:
                        res = method(text, target_sentences=target_sentences)
                        if hasattr(res, "__await__"):
                            res = await res  # type: ignore
                        if isinstance(res, str):
                            return res
                    except Exception:
                        pass
            # generic completion prompt
            prompt = (
                "Summarize the following text succinctly. "
                f"Target approximately {target_sentences} sentences. "
                "Text:\n\n" + text
            )
            for m in ("complete", "chat", "generate"):
                method = getattr(client, m, None)
                if callable(method):
                    try:
                        res = method(prompt)
                        if hasattr(res, "__await__"):
                            res = await res  # type: ignore
                        if isinstance(res, str) and len(res.strip()) > 0:
                            return res.strip()
                        # try dict with content
                        if isinstance(res, dict):
                            cand = res.get("content") or res.get("text")
                            if isinstance(cand, str) and cand.strip():
                                return cand.strip()
                    except Exception:
                        pass

    return None


def _two_pass_extractive(text: str, target_sentences: int, bullets: bool, chunk_chars: int) -> Dict[str, Any]:
    # First pass: chunk and summarize each chunk
    chunks = _chunk_text_by_chars(text, chunk_chars)
    per_chunk: List[str] = []
    for c in chunks:
        r = _extractive_summarize(c, target_sentences=max(1, target_sentences // 2 or 1), bullets=False)
        per_chunk.append(r["summary"])
    # Second pass: summarize the summaries
    combined = " ".join(per_chunk)
    final = _extractive_summarize(combined, target_sentences=target_sentences, bullets=bullets)
    final["chunks"] = len(chunks)
    return final


# ----------------------------
# Tool: summarize_text
# ----------------------------

@function_tool(timeout=60.0)
async def summarize_text(
    text: str,
    method: str = "auto",  # auto | extractive | llm
    target_sentences: int = 5,
    bullets: bool = False,
    max_chars_per_chunk: int = 8000,
    context=None,
) -> str:
    """Summarize long text using LLM (if provided in context) or local extractive method.

    Args:
        text: Input text to summarize
        method: 'auto' (default), 'extractive', or 'llm'
        target_sentences: Approximate number of sentences in output (default 5)
        bullets: If true, also return a bulleted list of key sentences
        max_chars_per_chunk: Chunk size for two-pass extractive summarization (default 8000 chars)

    Returns:
        JSON: {
          "type":"summary",
          "method":"extractive|context_llm",
          "sentences": n,
          "summary":"...",
          "bullets":[...],           # when requested
          "chunks": k                # when chunking applied
        } or {"error":"..."}
    """
    try:
        text = (text or "").strip()
        if not text:
            return json.dumps({"error": "Empty text"})

        use_llm = (method.lower() == "llm") or (method.lower() == "auto")

        # Try LLM/context route first if allowed
        if use_llm:
            try:
                llm_summary = await _summarize_via_context(text, context, target_sentences=target_sentences)
                if llm_summary and llm_summary.strip():
                    out = {
                        "type": "summary",
                        "method": "context_llm",
                        "summary": llm_summary.strip()
                    }
                    if bullets:
                        # naive sentence split for bullets
                        sents = _split_sentences(llm_summary)
                        out["bullets"] = sents[:max(1, target_sentences)]
                        out["sentences"] = len(sents)
                    else:
                        out["sentences"] = len(_split_sentences(llm_summary))
                    return json.dumps(out, ensure_ascii=False)
            except Exception:
                # fallback to extractive
                pass

        # Extractive route
        # Use two-pass if text is long
        if len(text) > max_chars_per_chunk * 1.25:
            result = _two_pass_extractive(text, target_sentences=target_sentences, bullets=bullets, chunk_chars=max_chars_per_chunk)
            result.update({"type": "summary", "method": "extractive"})
            return json.dumps(result, ensure_ascii=False)
        else:
            result = _extractive_summarize(text, target_sentences=target_sentences, bullets=bullets)
            result.update({"type": "summary", "method": "extractive"})
            return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Summarization failed: {str(e)}"})


def create_summarizer_tools():
    """Return list of Summarizer tools for easy agent registration."""
    return [summarize_text]