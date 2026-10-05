from functools import partial
from langgraph.graph import END, START, StateGraph

from src.plover_extraction.config import PipelineConfig
from src.plover_extraction.llm import LLMClient
from src.workflow.state import ExtractionState
from src.workflow.nodes.prepare_doc import prepare_doc
from src.workflow.nodes.raw_extractor import raw_extract
from src.plover_extraction.config import load_config
from src.plover_extraction.llm import OpenAICompatibleClient


def sub_graph(
    pipeline_config: PipelineConfig,
    client: LLMClient,
):
    workflow = StateGraph(ExtractionState)

    workflow.add_node(
        "prepare_doc",
        partial(
            prepare_doc,
            pipeline_config=pipeline_config,
        ),
    )

    workflow.add_node(
        "raw_extract",
        partial(
            raw_extract,
            pipeline_config=pipeline_config,
            client=client,
        ),
    )

    workflow.add_edge(START, "prepare_doc")
    workflow.add_edge("prepare_doc", "raw_extract")
    workflow.add_edge("raw_extract", END)

    return workflow.compile()

if __name__ == "__main__":
    config = load_config("config.example.yaml")
    client = OpenAICompatibleClient(config.llm)

    graph = sub_graph(config, client)
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)