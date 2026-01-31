import os
import logging
from typing import List
from interface.base_response_generator import BaseResponseGenerator
from util.invoke_ai import invoke_ai
from util.http_client import get_session

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Use the provided context to provide a concise answer to the user's question.
If you cannot find the answer in the context, say so. Do not make up information.
"""

SYSTEM_PROMPT_WEB_SEARCH = """
You are a helpful assistant. The user asked a question, and you did not find
an answer in the local database. You are now searching the web for an answer.
Provide a concise, factual answer based on the search results.
"""


class ResponseGenerator(BaseResponseGenerator):
    def __init__(self):
        self.crawl4ai_url = os.environ.get("CRAWL4AI_URL", "http://localhost:11235")
        self.crawl4ai_enabled = os.environ.get("CRAWL4AI_ENABLE", "0").lower() in ("1", "true", "yes")

    def _search_crawl4ai_web(self, query: str) -> str:
        """Attempt to search the web via Crawl4AI and return the answer."""
        if not self.crawl4ai_enabled:
            logger.debug("Crawl4AI web search disabled (CRAWL4AI_ENABLE not set)")
            return None

        from util.http_client import get_session
        session = get_session()  # Use singleton session
        try:
            # Try /search endpoint first for web search
            search_url = f"{self.crawl4ai_url.rstrip('/')}/search"
            try:
                resp = session.get(search_url, params={"q": query}, timeout=10)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        if isinstance(data, dict):
                            for key in ("results", "content", "answer"):
                                if key in data:
                                    return data[key]
                        elif isinstance(data, list):
                            return "\n".join(str(r) for r in data[:3])
                    except ValueError:
                        return resp.text
            except Exception as e:
                logger.debug("Crawl4AI /search endpoint failed: %s", e)

            # Fallback: use /ask endpoint with search intent
            ask_url = f"{self.crawl4ai_url.rstrip('/')}/ask"
            try:
                resp = session.get(
                    ask_url,
                    params={"q": f"Search the web for: {query}", "system": SYSTEM_PROMPT_WEB_SEARCH},
                    timeout=10,
                )
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        for key in ("answer", "text", "result", "response"):
                            if key in data:
                                return data[key]
                        return str(data)
                    except ValueError:
                        return resp.text
            except Exception as e:
                logger.debug("Crawl4AI /ask web search fallback failed: %s", e)

        except Exception as e:
            logger.warning("Web search via Crawl4AI failed: %s", e)

        return None

    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response using local context first, then fall back to web search.

        If the local database has no results, attempt to search Crawl4AI for answers.
        """
        context_text = "\n".join(context).strip()

        # If no context found in local database, try web search
        if not context_text:
            logger.info("No local context found for query '%s'; attempting web search via Crawl4AI", query)
            web_answer = self._search_crawl4ai_web(query)
            if web_answer:
                # Audit the web search response
                try:
                    from util.audit import audit_ai_call

                    try:
                        audit_ai_call(
                            SYSTEM_PROMPT_WEB_SEARCH,
                            query,
                            web_answer,
                            served_from_fallback=False,
                            metadata={"source": "crawl4ai_web_search"},
                        )
                    except Exception:
                        logger.debug("Audit write failed for web search (ignored)")
                except Exception:
                    pass
                return web_answer

            # If web search also fails, provide a message
            response = (
                f"Could not find information about '{query}' in the local database. "
                "Web search via Crawl4AI is disabled or unavailable. "
                "Please enable Crawl4AI by setting CRAWL4AI_ENABLE=1 and/or ensure a Crawl4AI server is running."
            )
            # Audit the failed web search attempt
            try:
                from util.audit import audit_ai_call

                try:
                    audit_ai_call(
                        SYSTEM_PROMPT_WEB_SEARCH,
                        query,
                        response,
                        served_from_fallback=True,
                        metadata={"reason": "web_search_unavailable"},
                    )
                except Exception:
                    logger.debug("Audit write failed for web search fallback (ignored)")
            except Exception:
                pass
            return response

        # If we have context, use the standard prompt
        user_message = (
            f"<context>\n{context_text}\n</context>\n"
            f"<question>\n{query}\n</question>"
        )

        return invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_message)
