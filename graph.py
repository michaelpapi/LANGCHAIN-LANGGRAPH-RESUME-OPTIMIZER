from langgraph.graph import StateGraph, START, END

from nodes import embed_documents, analyze_resume, generate_suggestions, check_reanalyze


def create_resume_graph():
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("embed_docs", embed_documents)
    graph.add_node("analyze_resume", analyze_resume)
    graph.add_node("generate_suggestions", generate_suggestions)
    graph.add_node("check_reanalyze", check_reanalyze)


    # Edges
    graph.set_entry_point("embed_docs")
    # graph.add_edge(START, "embed_docs")
    graph.add_edge("embed_docs", "analyze_resume")
    graph.add_edge("analyze_resume", "generate_suggestions")
    graph.add_edge("generate_suggestions", "check_reanalyze")

    # a branch that loops back if reanalyze returns as true
    graph.add_conditional_edges(
        "check_reanalyze",
        lambda state: "reanalyze" if state.get("reanalyze") else "end",
        {
            "reanalyze": "analyze_resume",
            "end": END
        }
    )

    return graph.compile()
