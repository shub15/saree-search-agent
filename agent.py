"""
agent.py
--------
Tool-calling agent for saree similarity search.

Uses the modern LangChain 1.x pattern:
  - ChatGoogleGenerativeAI with .bind_tools()
  - Manual tool-calling loop (no AgentExecutor needed)
  - Shared results_store list (closure) for the Streamlit UI

This pattern works with any LangChain version >= 0.3 and requires
no additional packages like langgraph.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from embedder import FashionEmbedder
from search import SareeIndex, SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are Aria, an expert fashion consultant specialising in Indian sarees. \
You assist users in finding visually similar sarees from a curated catalogue \
of over a thousand unique designs from Byrappa Silks.

## Your Capabilities
- When a user shares a saree image (file path or URL), you search the catalogue \
  for the closest visual matches using fine-grained AI embeddings.
- You understand nuanced differences in fabric, weave, print, colour \
  combination, border work and pallu patterns.

## Behaviour Rules
1. If the user mentions an image path or URL, ALWAYS call `find_similar_sarees`.
2. After getting results, write a warm, concise response:
   - Mention what you noticed about the query saree (colour, apparent style).
   - Name the top 1-2 closest matches and their prices.
   - Invite the user to explore further or upload another image.
3. If no image is provided, engage conversationally and ask the user \
   to share a saree image so you can search the catalogue.
4. Keep responses under 150 words — the UI displays the result images separately.
5. Never fabricate product names, prices or links beyond what the tool returns.
"""

# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------
def create_search_tool(
    embedder: FashionEmbedder,
    index: SareeIndex,
    results_store: List[Dict[str, Any]],
) -> Callable:
    """
    Returns a LangChain @tool that:
      1. Loads and embeds the query image.
      2. Searches the FAISS index.
      3. Writes rich results into results_store (for the Streamlit UI).
      4. Returns a JSON summary string to the LLM.
    """

    @tool
    def find_similar_sarees(image_source: str, top_k: int = 5) -> str:
        """
        Search the saree catalogue for images visually similar to the given image.

        Parameters
        ----------
        image_source : str
            The file path or HTTP(S) URL of the query saree image.
        top_k : int, optional
            Number of similar sarees to return (default 5, max 10).

        Returns
        -------
        str
            JSON with count, matches (name/sku/price/similarity/link), and a summary.
        """
        top_k = max(1, min(int(top_k), 10))

        try:
            pil_image = FashionEmbedder.load_image(image_source)
        except Exception as exc:
            err = f"Could not load image from '{image_source}': {exc}"
            logger.error(err)
            return json.dumps({"error": err, "count": 0, "matches": []})

        query_vec = embedder.embed_query(pil_image)
        matches: List[SearchResult] = index.search(query_vec, top_k=top_k)

        # Write into shared store so the Streamlit UI can display thumbnails
        results_store.clear()
        results_store.extend([m.to_dict() for m in matches])

        payload = {
            "count": len(matches),
            "matches": [
                {
                    "rank": m.rank,
                    "name": m.name or m.filename,
                    "sku": m.sku,
                    "discounted_price": m.discounted_price,
                    "similarity": m.similarity,
                    "website_link": m.website_link,
                }
                for m in matches
            ],
            "summary": (
                f"Found {len(matches)} similar sarees. "
                f"Top match: '{matches[0].name or matches[0].filename}' "
                f"(SKU {matches[0].sku}, \u20b9{matches[0].discounted_price}, "
                f"similarity {matches[0].similarity:.0%})"
                if matches
                else "No matches found."
            ),
        }
        return json.dumps(payload, ensure_ascii=False)

    return find_similar_sarees


# ---------------------------------------------------------------------------
# Agent class (manual tool-calling loop — works with LangChain 1.x)
# ---------------------------------------------------------------------------
class SareeAgent:
    """
    Wraps a Gemini LLM + find_similar_sarees tool in a simple
    tool-calling loop that works with any LangChain version >= 0.3.

    Usage:
        agent = SareeAgent(embedder, index, results_store, api_key)
        reply = agent.chat("Find sarees like this", chat_history, image_path)
    """

    MAX_ITERATIONS = 5

    def __init__(
        self,
        embedder: FashionEmbedder,
        index: SareeIndex,
        results_store: List[Dict[str, Any]],
        google_api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        api_key = google_api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "No Google API key found. "
                "Set GOOGLE_API_KEY in your environment or Streamlit secrets."
            )

        self._tool = create_search_tool(embedder, index, results_store)
        self._tool_map = {self._tool.name: self._tool}

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.4,
        )
        self._llm = llm.bind_tools([self._tool])

    def chat(
        self,
        user_input: str,
        history: Optional[List[BaseMessage]] = None,
    ) -> str:
        """
        Run one conversational turn and return the assistant's text reply.

        Parameters
        ----------
        user_input : str
            The user's message (may include an image path/URL reference).
        history : list of BaseMessage, optional
            Previous turns in LangChain message format.
        """
        messages: List[BaseMessage] = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + (history or [])
            + [HumanMessage(content=user_input)]
        )

        for iteration in range(self.MAX_ITERATIONS):
            response: AIMessage = self._llm.invoke(messages)

            # No tool calls → final text answer
            if not response.tool_calls:
                return response.content or "I found some results for you!"

            # Execute every requested tool call
            messages.append(response)
            for tc in response.tool_calls:
                tool_fn = self._tool_map.get(tc["name"])
                if tool_fn is None:
                    result = json.dumps({"error": f"Unknown tool: {tc['name']}"})
                else:
                    try:
                        result = tool_fn.invoke(tc["args"])
                    except Exception as exc:
                        result = json.dumps({"error": str(exc)})

                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

        # Fallback if we hit max iterations without a text response
        return "I completed the search. Please see the results displayed below."


# ---------------------------------------------------------------------------
# Convenience builder (used by app.py)
# ---------------------------------------------------------------------------
def build_agent(
    embedder: FashionEmbedder,
    index: SareeIndex,
    results_store: List[Dict[str, Any]],
    google_api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
) -> SareeAgent:
    return SareeAgent(
        embedder=embedder,
        index=index,
        results_store=results_store,
        google_api_key=google_api_key,
        model_name=model_name,
    )
