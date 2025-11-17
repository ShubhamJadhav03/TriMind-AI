import operator
import json
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Annotated, List
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch, TavilyExtract
from datetime import datetime
from langgraph.types import Command
from ai_launchpad.langgraph_module.multi_agent.supervisor.utils import truncate_messages

# --------------------------------------------------------
# Setup
# --------------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the researcher system prompt
researcher_prompt = open("prompts/researcher.md", "r").read()


# --------------------------------------------------------
# 1️⃣ TOOL: Web Search
# --------------------------------------------------------
@tool
async def search_web(query: str, num_results: int = 3):

    # return title, url, content_preview
    """Search the web and return summarized search results."""
    web_search = TavilySearch(max_results=min(num_results, 3), topic="general")
    raw = web_search.invoke(input={"query": query})

    # Normalize possible return types
    if isinstance(raw, str):
        try:
            raw_parsed = json.loads(raw)
            raw = raw_parsed
        except Exception:
            logger.warning("TavilySearch returned a plain string, not JSON.")
            return {
                "query": query,
                "results": [{"title": None, "url": None, "content_preview": raw}],
            }

    # Determine structure
    if isinstance(raw, dict) and "results" in raw:
        search_results = raw["results"]
    elif isinstance(raw, list):
        search_results = raw
    else:
        logger.warning(f"Unexpected TavilySearch output type: {type(raw)}")
        return {"query": query, "results": []}

    processed_results = {"query": query, "results": []}
    for result in search_results:
        if isinstance(result, dict):
            title = result.get("title")
            url = result.get("url")
            content_preview = result.get("content", "")
        else:
            title = None
            url = None
            content_preview = str(result)

        processed_results["results"].append({
            "title": title,
            "url": url,
            "content_preview": content_preview
        })

    return processed_results


# --------------------------------------------------------
# 2️⃣ TOOL: Webpage Content Extractor
# --------------------------------------------------------
@tool
async def extract_content_from_webpage(urls: List[str]):
    """Extract readable content from one or more webpages."""

    # Use TavilyExtract to get content
    web_extract = TavilyExtract()
    raw = web_extract.invoke(input={"urls": urls})

    logger.info(f"TavilyExtract returned type: {type(raw)}")

    # Try parsing JSON string
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            raw = parsed
        except Exception:
            logger.warning("TavilyExtract returned non-JSON string.")
            return [{"url_list": urls, "content": raw}]

    # Extract results safely
    if isinstance(raw, dict) and "results" in raw:
        results = raw["results"]
        return results if isinstance(results, list) else [results]

    if isinstance(raw, list):
        return raw

    # Fallback: wrap anything else into a list
    return [raw]


# --------------------------------------------------------
# 3️⃣ TOOL: Generate Research Report
# --------------------------------------------------------
class ResearchReport(BaseModel):
    topic: str
    report: str


@tool
async def generate_research_report(
    topic: str,
    report: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Generate a structured research report for a given topic."""
    research_report = ResearchReport.model_validate({
        "topic": topic,
        "report": report
    })

    return Command(update={
        "research_reports": [research_report],
        "messages": [ToolMessage(
            name="generate_research_report",
            content=research_report.model_dump_json(),
            tool_call_id=tool_call_id,
        )],
    })


# --------------------------------------------------------
# 4️⃣ STATE MODEL
# --------------------------------------------------------
class ResearcherState(BaseModel):
    """Researcher agent state model shared with supervisor."""
    messages: Annotated[list, add_messages] = []
    research_reports: Annotated[list, operator.add] = []


# --------------------------------------------------------
# 5️⃣ AGENT SETUP
# --------------------------------------------------------
tools = [
    search_web,
    extract_content_from_webpage,
    generate_research_report,
]

llm = ChatOpenAI(
    name="Researcher",
    model="gpt-4o-mini",
)
llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------------------
# 6️⃣ RESEARCHER NODE
# --------------------------------------------------------
async def researcher(state: ResearcherState):
    """Main logic of the Researcher agent."""
    system_msg = SystemMessage(content=researcher_prompt.format(current_datetime=datetime.now()))

    truncated_messages = truncate_messages(
        messages=state.messages,
        system_message=system_msg,
        max_messages=10,
        max_tokens_approx=100000,
    )

    # Prevent API error from incomplete tool call sequences
    safe_messages = []
    skip_until_index = -1

    for i, msg in enumerate(truncated_messages):
        if i <= skip_until_index:
            continue

        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            required_ids = {t.id for t in msg.tool_calls if hasattr(t, "id") and t.id}
            if required_ids:
                found_ids = set()
                tool_indices = []
                for j in range(i + 1, len(truncated_messages)):
                    next_msg = truncated_messages[j]
                    if isinstance(next_msg, (AIMessage, HumanMessage, SystemMessage)):
                        break
                    if isinstance(next_msg, ToolMessage) and hasattr(next_msg, "tool_call_id"):
                        if next_msg.tool_call_id in required_ids:
                            found_ids.add(next_msg.tool_call_id)
                            tool_indices.append(j)
                if found_ids == required_ids:
                    safe_messages.append(msg)
                    for idx in tool_indices:
                        safe_messages.append(truncated_messages[idx])
                    if tool_indices:
                        skip_until_index = max(tool_indices)
                else:
                    if tool_indices:
                        skip_until_index = max(tool_indices)
                    continue
            else:
                safe_messages.append(msg)
        else:
            safe_messages.append(msg)

    response = llm_with_tools.invoke(safe_messages)
    return {"messages": [response]}


# --------------------------------------------------------
# 7️⃣ ROUTING
# --------------------------------------------------------
async def researcher_router(state: ResearcherState) -> str:
    """Route to tools if the last message includes tool calls."""
    if state.messages and hasattr(state.messages[-1], "tool_calls") and state.messages[-1].tool_calls:
        return "tools"
    return END


# --------------------------------------------------------
# 8️⃣ GRAPH BUILDING
# --------------------------------------------------------
builder = StateGraph(ResearcherState)
builder.add_node(researcher)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("researcher")
builder.add_edge("tools", "researcher")
builder.add_conditional_edges("researcher", researcher_router, {
    "tools": "tools",
    END: END,
})

graph = builder.compile()
# Optional: persistent memory
# graph = builder.compile(checkpointer=MemorySaver())

# --------------------------------------------------------
# Done ✅
# --------------------------------------------------------
# Optional visualization:
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())
