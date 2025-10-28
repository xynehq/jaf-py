"""
Knowledge Base (KB) tools for JAF.

These tools retrieve context from a vector DB / document store, and optionally ingest content.
They are designed to work with a pluggable KB/retriever provided via the agent context, without
hard dependencies on specific vendors. The tools attempt to adapt to common method names used by
popular libraries (e.g., vector_store.search, retriever.search/similarity_search, upsert/add_texts).

Expected context attributes (any one):
- context.kb
- context.retriever
- context.vector_store
- context.document_store

Commonly supported methods (any one for search):
- search(query, top_k=..., filters=..., namespace=...)
- similarity_search(query, k=..., filter=..., namespace=...)
- query(query, top_k=...)
- retrieve(query, top_k=...)
- get_relevant_documents(query)  # returns documents

Commonly supported methods (any one for ingest):
- add_texts(texts: List[str], metadatas: Optional[List[dict]] = None, namespace: Optional[str] = None)
- upsert(items=[{"id": "...", "text": "...", "metadata": {...}, "namespace": "..."}])
- add(documents=[{"text": "...", "metadata": {...}, "namespace": "..."}])
- add_documents(list_of_documents)

All tools return JSON strings for structured consumption.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from ..core.tools import function_tool


def _get_kb_obj(context: Any) -> Optional[Any]:
    """Probe the context for an available KB-like object."""
    if context is None:
        return None
    for attr in ("kb", "retriever", "vector_store", "document_store"):
        if hasattr(context, attr):
            return getattr(context, attr)
    return None


def _to_list_dict(obj: Any) -> List[Dict[str, Any]]:
    """Normalize search results to [{'id', 'text', 'score', 'metadata'}]."""
    results: List[Dict[str, Any]] = []

    def as_text(x: Any) -> Optional[str]:
        if x is None:
            return None
        if isinstance(x, str):
            return x
        # Common attributes
        for attr in ("text", "page_content", "content", "document", "body"):
            if hasattr(x, attr):
                v = getattr(x, attr)
                if isinstance(v, str):
                    return v
        # Dict-like
        if isinstance(x, dict):
            for k in ("text", "page_content", "content", "document", "body"):
                if k in x and isinstance(x[k], str):
                    return x[k]
        # Fallback string repr (avoid huge dumps)
        return str(x)

    def as_meta(x: Any) -> Dict[str, Any]:
        if x is None:
            return {}
        if isinstance(x, dict):
            # Heuristic: prefer 'metadata' key if nested
            if "metadata" in x and isinstance(x["metadata"], dict):
                return dict(x["metadata"])
            # else use the dict itself minus common content keys
            meta = dict(x)
            for k in ("text", "page_content", "content", "document", "body", "id", "score"):
                if k in meta:
                    meta.pop(k, None)
            return meta
        # Common attributes
        if hasattr(x, "metadata") and isinstance(getattr(x, "metadata"), dict):
            return dict(getattr(x, "metadata"))
        # LangChain: Document with .metadata
        if hasattr(x, "__dict__"):
            # try best-effort extraction of fields other than content-like
            meta = {}
            for k, v in x.__dict__.items():
                if k not in ("text", "page_content", "content", "document", "body", "id", "score"):
                    meta[k] = v
            return meta
        return {}

    def as_id(x: Any) -> Optional[str]:
        if x is None:
            return None
        if isinstance(x, dict):
            vid = x.get("id") or x.get("doc_id") or x.get("_id")
            return str(vid) if vid is not None else None
        for attr in ("id", "doc_id", "_id"):
            if hasattr(x, attr):
                return str(getattr(x, attr))
        return None

    def as_score(x: Any) -> Optional[float]:
        if x is None:
            return None
        if isinstance(x, dict) and "score" in x:
            try:
                return float(x["score"])
            except Exception:
                return None
        if hasattr(x, "score"):
            try:
                return float(getattr(x, "score"))
            except Exception:
                return None
        # Sometimes similarity distance; skip if not present
        return None

    # Normalize to list
    if obj is None:
        return results
    if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], list):
        obj = obj["results"]
    if not isinstance(obj, list):
        obj = [obj]

    for item in obj:
        # Handle (doc, score) tuples
        if isinstance(item, tuple) and len(item) >= 1:
            doc = item[0]
            score = None
            if len(item) > 1:
                try:
                    score = float(item[1])  # may be similarity or distance
                except Exception:
                    score = None
            results.append({
                "id": as_id(doc),
                "text": as_text(doc),
                "score": score if score is not None else as_score(doc),
                "metadata": as_meta(doc),
            })
        else:
            results.append({
                "id": as_id(item),
                "text": as_text(item),
                "score": as_score(item),
                "metadata": as_meta(item),
            })

    return results


def _search_adapter(
    kb: Any,
    query: str,
    top_k: int,
    filters: Optional[Dict[str, Any]],
    namespace: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Try multiple common search methods, normalize the results.
    """
    kwargs_common = {}
    if namespace is not None:
        kwargs_common["namespace"] = namespace

    # Try exact method names
    for method_name, kwargs_map in [
        ("search", {"top_k": top_k, "filters": filters, **kwargs_common}),
        ("similarity_search", {"k": top_k, "filter": filters, **kwargs_common}),
        ("query", {"top_k": top_k, **kwargs_common}),
        ("retrieve", {"top_k": top_k, "filters": filters, **kwargs_common}),
        ("get_relevant_documents", {}),  # usually only query param
    ]:
        if hasattr(kb, method_name):
            try:
                method = getattr(kb, method_name)
                if method_name == "get_relevant_documents":
                    res = method(query)
                else:
                    res = method(query, **{k: v for k, v in kwargs_map.items() if v is not None})
                return _to_list_dict(res)
            except Exception:
                # Try next method
                pass

    # If kb has inner store
    for inner_attr in ("vector_store", "store", "client"):
        inner = getattr(kb, inner_attr, None)
        if inner is not None and inner is not kb:
            try:
                return _search_adapter(inner, query, top_k, filters, namespace)
            except Exception:
                pass

    raise RuntimeError("No compatible search method found on KB object")


def _ingest_adapter(
    kb: Any,
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]],
    namespace: Optional[str],
) -> Dict[str, Any]:
    """
    Try multiple common ingest methods and return {"inserted": n, "ids": [...]}
    """
    meta = metadatas if isinstance(metadatas, list) else None
    if meta is not None and len(meta) != len(texts):
        # Align metadata length if shorter by padding with {}
        if len(meta) < len(texts):
            meta = meta + [{} for _ in range(len(texts) - len(meta))]
        else:
            meta = meta[:len(texts)]

    # add_texts
    if hasattr(kb, "add_texts"):
        try:
            if namespace is not None:
                ids = kb.add_texts(texts=texts, metadatas=meta, namespace=namespace)
            else:
                ids = kb.add_texts(texts=texts, metadatas=meta)
            return {"inserted": len(texts), "ids": list(ids) if ids is not None else []}
        except Exception:
            pass

    # upsert
    if hasattr(kb, "upsert"):
        try:
            items = []
            for i, t in enumerate(texts):
                item = {"text": t}
                if meta:
                    item["metadata"] = meta[i]
                if namespace is not None:
                    item["namespace"] = namespace
                items.append(item)
            res = kb.upsert(items=items)
            # Try to find returned IDs
            ids = []
            try:
                if isinstance(res, dict):
                    ids = res.get("ids") or res.get("upserted_ids") or []
            except Exception:
                pass
            return {"inserted": len(texts), "ids": ids}
        except Exception:
            pass

    # add / add_documents
    for m in ("add", "add_documents"):
        if hasattr(kb, m):
            try:
                docs = []
                for i, t in enumerate(texts):
                    d = {"text": t}
                    if meta:
                        d["metadata"] = meta[i]
                    if namespace is not None:
                        d["namespace"] = namespace
                    docs.append(d)
                method = getattr(kb, m)
                res = method(docs)
                ids = []
                try:
                    if isinstance(res, dict):
                        ids = res.get("ids") or []
                except Exception:
                    pass
                return {"inserted": len(texts), "ids": ids}
            except Exception:
                pass

    # If kb has inner store
    for inner_attr in ("vector_store", "store", "client"):
        inner = getattr(kb, inner_attr, None)
        if inner is not None and inner is not kb:
            try:
                return _ingest_adapter(inner, texts, meta, namespace)
            except Exception:
                pass

    raise RuntimeError("No compatible ingest method found on KB object")


@function_tool(timeout=30.0)
async def kb_search(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    namespace: Optional[str] = None,
    filters_json: Optional[str] = None,
    include_metadata: bool = True,
    context=None,
) -> str:
    """Search the knowledge base via context-provided retriever/vector store.

    Args:
        query: Natural language query
        top_k: Number of results to return
        min_score: Minimum similarity score threshold (if scores available)
        namespace: Optional namespace/collection identifier
        filters_json: Optional JSON dict for metadata filters (implementation-specific)
        include_metadata: Whether to include metadata in results

    Returns:
        JSON: {"type":"kb_search","query":"...","top_k":...,"results":[{"id":...,"text":"...","score":...,"metadata":{...}}]}
    """
    try:
        kb = _get_kb_obj(context)
        if kb is None:
            return json.dumps({"error": "No KB/retriever found on context (expected one of: kb, retriever, vector_store, document_store)"})

        filters: Optional[Dict[str, Any]] = None
        if filters_json:
            try:
                f = json.loads(filters_json)
                if isinstance(f, dict):
                    filters = f
            except Exception:
                return json.dumps({"error": "Invalid filters_json; must be a JSON object"})

        raw = _search_adapter(kb, query=query, top_k=int(top_k), filters=filters, namespace=namespace)
        # Apply score threshold if present
        out = []
        for r in raw:
            if r.get("score") is not None and min_score is not None:
                try:
                    if float(r["score"]) < float(min_score):
                        continue
                except Exception:
                    pass
            if not include_metadata:
                r = {k: v for k, v in r.items() if k != "metadata"}
            out.append(r)

        return json.dumps({"type": "kb_search", "query": query, "top_k": top_k, "results": out}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"KB search failed: {str(e)}"})


@function_tool(timeout=60.0)
async def kb_ingest_texts(
    texts_json: str,
    metadatas_json: Optional[str] = None,
    namespace: Optional[str] = None,
    context=None,
) -> str:
    """Ingest a list of texts (with optional metadata) into the KB.

    Args:
        texts_json: JSON list of text strings
        metadatas_json: Optional JSON list of metadata dicts (same length as texts)
        namespace: Optional namespace/collection identifier

    Returns:
        JSON: {"type":"kb_ingest","inserted": n, "ids": [...]} or {"error":"..."}
    """
    try:
        kb = _get_kb_obj(context)
        if kb is None:
            return json.dumps({"error": "No KB/retriever found on context (expected one of: kb, retriever, vector_store, document_store)"})

        try:
            texts = json.loads(texts_json)
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                return json.dumps({"error": "texts_json must be a JSON list of strings"})
        except Exception as e:
            return json.dumps({"error": f"Invalid texts_json: {str(e)}"})

        metadatas = None
        if metadatas_json:
            try:
                tmp = json.loads(metadatas_json)
                if not isinstance(tmp, list) or not all(isinstance(m, dict) for m in tmp):
                    return json.dumps({"error": "metadatas_json must be a JSON list of objects"})
                metadatas = tmp
            except Exception as e:
                return json.dumps({"error": f"Invalid metadatas_json: {str(e)}"})

        res = _ingest_adapter(kb, texts=texts, metadatas=metadatas, namespace=namespace)
        return json.dumps({"type": "kb_ingest", **res}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"KB ingest failed: {str(e)}"})


def create_knowledge_base_tools():
    """Return list of Knowledge Base tools for easy agent registration."""
    return [kb_search, kb_ingest_texts]