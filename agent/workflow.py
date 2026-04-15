from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def build_source_snippets(retrieved: Sequence[dict[str, Any]], reranked_texts: Sequence[str], limit: int = 2) -> str:
    snippets: list[str] = []
    for passage in reranked_texts[:limit]:
        source = "Guideline"
        for item in retrieved:
            if item["content"] == passage:
                source = item.get("source", "Guideline")
                break
        snippets.append(f"[{source}] {passage[:180]}...")
    return "\n".join(snippets)


def run_patient_workflow(
    *,
    raw_inputs: dict[str, Any],
    model_features: dict[str, Any],
    probability: float,
    index: Any,
    chunks: Sequence[dict[str, Any]],
    graph_builder: Callable[[], Any] | None = None,
    factor_fn: Callable[[dict[str, Any]], list[str]] | None = None,
    doctor_fn: Callable[[dict[str, Any], float], list[str]] | None = None,
    search_fn: Callable[..., list[dict[str, Any]]] | None = None,
    rerank_fn: Callable[[str, list[str]], list[str]] | None = None,
    prompt_fn: Callable[..., str] | None = None,
    response_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    if graph_builder is None:
        from agent.langgraph_flow import build_graph

        graph_builder = build_graph
    if factor_fn is None:
        from agent.utils import extract_factors

        factor_fn = extract_factors
    if doctor_fn is None:
        from agent.doctor import recommend_department

        doctor_fn = recommend_department
    if search_fn is None:
        from agent.rag_faiss import search

        search_fn = search
    if rerank_fn is None:
        from agent.reranker import rerank

        rerank_fn = rerank
    if prompt_fn is None:
        from agent.prompt import build_prompt

        prompt_fn = build_prompt
    if response_fn is None:
        from agent.llm import generate_ai_response

        response_fn = generate_ai_response

    prompt_data = dict(raw_inputs)
    prompt_data.update(model_features)

    workflow = graph_builder()
    final_state = workflow.invoke(
        {
            "data": raw_inputs,
            "model_features": model_features,
            "prompt_data": prompt_data,
            "predict": lambda _: probability,
            "extract_factors": factor_fn,
            "doctor_fn": doctor_fn,
            "search": search_fn,
            "rerank": rerank_fn,
            "build_prompt": prompt_fn,
            "generate_response": response_fn,
            "index": index,
            "chunks": chunks,
        }
    )

    reranked_texts = final_state["reranked"]
    retrieved = final_state["retrieved"]

    return {
        "probability": final_state["prob"],
        "factors": final_state["factors"],
        "specialists": final_state["departments"],
        "ai_response": final_state["response"],
        "retrieved": retrieved,
        "reranked_texts": reranked_texts,
        "context": final_state["context"],
        "source_context": build_source_snippets(retrieved, reranked_texts),
    }
