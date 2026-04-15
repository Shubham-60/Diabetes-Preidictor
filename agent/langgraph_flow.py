from __future__ import annotations

from typing import Any, cast

from langgraph.graph import StateGraph
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    data: dict[str, Any]
    model_features: dict[str, Any]
    prompt_data: dict[str, Any]
    rag_profile: str
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
    state_dict = cast(dict[str, Any], state)
    features = state_dict.get("model_features")
    if features is None:
        raise ValueError("model_features missing in state")

    state_dict["prob"] = state_dict["predict"](features)
    return state


def factor_node(state: GraphState) -> GraphState:
    state_dict = cast(dict[str, Any], state)
    factors = state_dict["extract_factors"](state_dict["data"])
    state_dict["factors"] = factors
    query_parts = [" ".join(factors)]
    rag_profile = state_dict.get("rag_profile")
    if rag_profile:
        query_parts.append(rag_profile)
    state_dict["query"] = " ".join(part for part in query_parts if part).strip()
    return state


def doctor_node(state: GraphState) -> GraphState:
    state_dict = cast(dict[str, Any], state)
    state_dict["departments"] = state_dict["doctor_fn"](state_dict["model_features"], state_dict["prob"])
    return state


def rag_node(state: GraphState) -> GraphState:
    state_dict = cast(dict[str, Any], state)
    results = state_dict["search"](state_dict["query"], state_dict["index"], state_dict["chunks"])
    reranked = state_dict["rerank"](state_dict["query"], [r["content"] for r in results])
    state_dict["retrieved"] = results
    state_dict["reranked"] = reranked
    state_dict["context"] = "\n".join(reranked)
    return state


def llm_node(state: GraphState) -> GraphState:
    state_dict = cast(dict[str, Any], state)
    prompt = state_dict["build_prompt"](
        state_dict["prompt_data"],
        state_dict["prob"],
        state_dict["factors"],
        state_dict["context"],
        state_dict["departments"],
    )
    state_dict["response"] = state_dict["generate_response"](prompt)
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
