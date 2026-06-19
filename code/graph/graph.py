"""LangGraph graph definition for the claim evaluation pipeline."""

from langgraph.graph import END, START, StateGraph

from code.graph.nodes import (
    analyze_images,
    extract_claim,
    format_output,
    load_context,
    make_fallback_decision,
    route_on_images,
    synthesize_decision,
)
from code.graph.state import ClaimState


def build_graph(checkpointer=None):
    """Build and compile the claim evaluation graph.

    checkpointer: pass a saver instance or None for MemorySaver (thread-safe default).
    SQLiteSaver is intentionally NOT the default here — it is not safe for concurrent
    threads sharing the same connection. main.py passes a MemorySaver per-thread.
    """
    if checkpointer is None:
        try:
            from langgraph.checkpoint.memory import InMemorySaver as _MemSaver
        except ImportError:
            from langgraph.checkpoint.memory import MemorySaver as _MemSaver
        checkpointer = _MemSaver()

    g = StateGraph(ClaimState)

    g.add_node("load_context", load_context)
    g.add_node("extract_claim", extract_claim)
    g.add_node("analyze_images", analyze_images)
    g.add_node("synthesize_decision", synthesize_decision)
    g.add_node("make_fallback_decision", make_fallback_decision)
    g.add_node("format_output", format_output)

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "extract_claim")
    g.add_edge("extract_claim", "analyze_images")

    g.add_conditional_edges(
        "analyze_images",
        route_on_images,
        {
            "synthesize_decision": "synthesize_decision",
            "make_fallback_decision": "make_fallback_decision",
        },
    )

    g.add_edge("synthesize_decision", "format_output")
    g.add_edge("make_fallback_decision", "format_output")
    g.add_edge("format_output", END)

    return g.compile(checkpointer=checkpointer)
