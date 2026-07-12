"""
AI Agent - Orchestrates multiple tools using LangGraph.

This is the most advanced component of our system. Instead of the
user manually choosing which endpoint to call, the agent:
1. Analyzes the user's question
2. Decides which tools to use
3. Calls them in the right order
4. Combines results into one comprehensive answer

LangGraph concepts used:
- StateGraph: defines the workflow as a graph
- State: data that flows between nodes
- Nodes: functions that process/transform state
- Conditional edges: LLM decides which node runs next
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    """
    State that flows through the agent graph.

    Every node can read from and write to this state.
    TypedDict ensures type safety — we know exactly
    what data is available at each step.
    """
    question: str
    collection_name: str | None
    needs_rag: bool
    needs_news: bool
    needs_ratios: bool
    rag_result: str
    rag_sources: list
    news_result: list
    ratios_result: list
    final_answer: str


class FinancialAgent:
    """
    Multi-tool financial research agent.

    The agent uses the LLM to classify the question,
    then routes to the appropriate tools, then synthesizes
    everything into a final answer.
    """

    def __init__(self, rag_service):
        """
        Takes the shared RAGService so the agent uses
        the same models and vector store as the rest of the app.
        """
        self.rag_service = rag_service
        self.llm_service = rag_service.llm_service
        self.news_service = None  # lazy import to avoid circular deps

        # Build the graph
        self.graph = self._build_graph()

    def _get_news_service(self):
        if self.news_service is None:
            from app.services.news_service import NewsService
            self.news_service = NewsService()
        return self.news_service

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Graph structure:
            classify → [rag_search, news_fetch, ratio_calc] → synthesize → END

        The classify node uses the LLM to decide which tools are needed.
        Then all needed tools run. Finally, synthesize combines everything.
        """
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("classify", self._classify)
        graph.add_node("rag_search", self._rag_search)
        graph.add_node("news_fetch", self._news_fetch)
        graph.add_node("synthesize", self._synthesize)

        # Set entry point
        graph.set_entry_point("classify")

        # Conditional edges from classify
        graph.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "rag": "rag_search",
                "news": "news_fetch",
                "synthesize": "synthesize",
            },
        )

        # After RAG, check if we also need news
        graph.add_conditional_edges(
            "rag_search",
            self._route_after_rag,
            {
                "news": "news_fetch",
                "synthesize": "synthesize",
            },
        )

        # After news, always go to synthesize
        graph.add_edge("news_fetch", "synthesize")

        # After synthesize, end
        graph.add_edge("synthesize", END)

        return graph.compile()

    def _classify(self, state: AgentState) -> dict:
        """
        Node 1: Classify the question to determine which tools to use.

        Uses the LLM to analyze the question and decide:
        - Does it need document search (RAG)?
        - Does it need latest news?
        - Does it need ratio calculations?
        """
        classify_prompt = (
            "Analyze this financial question and determine what tools are needed.\n\n"
            f'Question: "{state["question"]}"\n\n'
            f'Document available: {"Yes - " + state["collection_name"] if state.get("collection_name") else "No"}\n\n'
            "Respond with ONLY these three lines, nothing else:\n"
            "NEEDS_RAG: yes or no\n"
            "NEEDS_NEWS: yes or no\n"
            "NEEDS_RATIOS: yes or no\n\n"
            "Rules:\n"
            "- NEEDS_RAG=yes if the question asks about company financials, reports, statements, or specific data\n"
            "- NEEDS_NEWS=yes if the question asks about recent events, current status, or latest developments\n"
            "- NEEDS_RATIOS=yes if the question mentions specific ratios, valuation, or financial metrics to calculate\n"
            "- If no document is available, NEEDS_RAG must be no"
        )

        response = self.llm_service.chat(
            user_message=classify_prompt,
            system_prompt="You are a task classifier. Respond ONLY in the exact format requested.",
        )

        response_lower = response.lower()

        needs_rag = "needs_rag: yes" in response_lower and state.get("collection_name") is not None
        needs_news = "needs_news: yes" in response_lower
        needs_ratios = "needs_ratios: yes" in response_lower

        # If nothing was classified, default to RAG if document exists, else news
        if not needs_rag and not needs_news and not needs_ratios:
            if state.get("collection_name"):
                needs_rag = True
            else:
                needs_news = True

        return {
            "needs_rag": needs_rag,
            "needs_news": needs_news,
            "needs_ratios": needs_ratios,
        }

    def _route_after_classify(self, state: AgentState) -> str:
        """Decide which node to go to after classification."""
        if state["needs_rag"]:
            return "rag"
        if state["needs_news"]:
            return "news"
        return "synthesize"

    def _route_after_rag(self, state: AgentState) -> str:
        """After RAG, check if we also need news."""
        if state["needs_news"]:
            return "news"
        return "synthesize"

    def _rag_search(self, state: AgentState) -> dict:
        """
        Node 2: Search uploaded documents using RAG.
        """
        if not state.get("collection_name"):
            return {
                "rag_result": "No document uploaded to search.",
                "rag_sources": [],
            }

        try:
            result = self.rag_service.query(
                question=state["question"],
                collection_name=state["collection_name"],
                top_k=5,
            )

            return {
                "rag_result": result["answer"],
                "rag_sources": result["sources"],
            }
        except Exception as e:
            return {
                "rag_result": f"Document search failed: {str(e)}",
                "rag_sources": [],
            }

    def _news_fetch(self, state: AgentState) -> dict:
        """
        Node 3: Fetch latest financial news.
        """
        try:
            news = self._get_news_service()

            # Extract company name from question or collection name
            company = (state.get("collection_name") or "").replace("_", " ")
            if not company:
                # Try to extract from question
                company = state["question"]

            result = news.search_news(company, max_results=3)

            return {"news_result": result}
        except Exception as e:
            return {"news_result": [{"title": f"News fetch failed: {str(e)}", "snippet": "", "url": "", "source": "error"}]}

    def _synthesize(self, state: AgentState) -> dict:
        """
        Node 4: Combine all gathered information into one answer.

        This is where the agent shines — it takes RAG results,
        news, and ratios and produces a unified analysis.
        """
        context_parts = []

        # Add RAG results if available
        if state.get("rag_result") and state["rag_result"] != "No document uploaded to search.":
            context_parts.append(
                f"## Document Analysis:\n{state['rag_result']}"
            )

        # Add news if available
        if state.get("news_result"):
            news_text = "\n".join(
                [f"- {a['title']} ({a['source']})" for a in state["news_result"]
                 if a.get("source") != "error"]
            )
            if news_text:
                context_parts.append(f"## Latest News:\n{news_text}")

        # Add ratios if available
        if state.get("ratios_result"):
            ratios_text = "\n".join(
                [f"- {r['ratio']}: {r['value']}" for r in state["ratios_result"]]
            )
            context_parts.append(f"## Financial Ratios:\n{ratios_text}")

        if not context_parts:
            # No tools returned data, just answer directly
            answer = self.llm_service.chat(
                user_message=state["question"],
                system_prompt="You are a financial research analyst. Answer concisely.",
            )
            return {"final_answer": answer}

        combined_context = "\n\n".join(context_parts)

        synthesis_prompt = (
            f"Based on the following information, provide a comprehensive "
            f"answer to the user's question.\n\n"
            f"## Gathered Information:\n{combined_context}\n\n"
            f"## User's Question:\n{state['question']}\n\n"
            f"## Instructions:\n"
            f"- Combine insights from all sources into one cohesive answer\n"
            f"- Start with a direct answer\n"
            f"- Clearly separate document data from news updates\n"
            f"- Keep it under 300 words\n"
            f"- End with key takeaways"
        )

        answer = self.llm_service.chat(
            user_message=synthesis_prompt,
            system_prompt="You are a senior financial research analyst. Synthesize information from multiple sources.",
        )

        return {"final_answer": answer}

    def run(self, question: str, collection_name: str = None) -> dict:
        """
        Run the agent on a question.

        This is the main entry point. Give it a question,
        optionally a document collection, and it figures out
        what to do.
        """
        initial_state = {
            "question": question,
            "collection_name": collection_name,
            "needs_rag": False,
            "needs_news": False,
            "needs_ratios": False,
            "rag_result": "",
            "rag_sources": [],
            "news_result": [],
            "ratios_result": [],
            "final_answer": "",
        }

        # Run the graph
        result = self.graph.invoke(initial_state)

        return {
            "answer": result["final_answer"],
            "sources": result.get("rag_sources", []),
            "news": result.get("news_result", []),
            "tools_used": self._get_tools_used(result),
            "question": question,
        }

    def _get_tools_used(self, state: dict) -> list[str]:
        """Track which tools the agent used — useful for transparency."""
        tools = []
        if state.get("needs_rag"):
            tools.append("document_search")
        if state.get("needs_news"):
            tools.append("news_fetch")
        if state.get("needs_ratios"):
            tools.append("ratio_calculator")
        return tools