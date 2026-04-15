from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    data: dict[str, Any]
    model_features: dict[str, Any]
    prompt_data: dict[str, Any]
    prob: float
    factors: list[str]
    query: str
    retrieved: list[dict[str, Any]]
    reranked: list[str]
    context: str
    response: str
    departments: list[str]
    predict: Any
    extract_factors: Any
    doctor_fn: Any
    search: Any
    rerank: Any
    build_prompt: Any
    generate_response: Any
    index: Any
    chunks: Any


def ml_node(state: GraphState) -> GraphState:
    features = state.get("model_features")
    if features is None:
        raise ValueError("model_features missing in state")

    state["prob"] = state["predict"](features)
    return state


def factor_node(state: GraphState) -> GraphState:
    factors = state["extract_factors"](state["data"])
    state["factors"] = factors
    state["query"] = " ".join(factors)
    return state


def doctor_node(state: GraphState) -> GraphState:
    state["departments"] = state["doctor_fn"](state["model_features"], state["prob"])
    return state


def rag_node(state: GraphState) -> GraphState:
    results = state["search"](state["query"], state["index"], state["chunks"])
    reranked = state["rerank"](state["query"], [r["content"] for r in results])
    state["retrieved"] = results
    state["reranked"] = reranked
    state["context"] = "\n".join(reranked)
    return state


def llm_node(state: GraphState) -> GraphState:
    prompt = state["build_prompt"](
        state["prompt_data"],
        state["prob"],
        state["factors"],
        state["context"],
        state["departments"],
    )
    state["response"] = state["generate_response"](prompt)
    return state


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("ml", ml_node)
    graph.add_node("factors", factor_node)
    graph.add_node("doctor", doctor_node)
    graph.add_node("rag", rag_node)
    graph.add_node("llm", llm_node)

    graph.set_entry_point("ml")
    graph.add_edge("ml", "factors")
    graph.add_edge("factors", "doctor")
    graph.add_edge("doctor", "rag")
    graph.add_edge("rag", "llm")
    graph.set_finish_point("llm")

    return graph.compile()
