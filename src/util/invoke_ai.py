import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# Import singleton at runtime to avoid circular imports
def get_singleton_session():
    from util.http_client import get_session
    return get_session()


def invoke_ai(system_message: str, user_message: str, base_url: Optional[str] = None) -> str:
    """Invoke a local Crawl4AI server running at `base_url` using the `/ask` endpoint.

    This is a simple, best-effort integration: it sends the combined messages as
    the `q` parameter and returns the text result. The Crawl4AI server must be
    running (see `./scripts/crawl4ai_postinstall.sh`).
    """

    # Determine base URL and whether Crawl4AI integration is enabled.
    if base_url is None:
        base_url = os.environ.get("CRAWL4AI_URL", "http://localhost:11235")

    enabled = os.environ.get("CRAWL4AI_ENABLE", "0").lower() in ("1", "true", "yes")
    if not enabled:
        logger.debug("Crawl4AI invoke disabled (CRAWL4AI_ENABLE not set). Returning fallback message.")
        # Provide a useful but safe fallback response
        resp_text = (
            "Crawl4AI integration is disabled. Install or enable Crawl4AI to generate LLM responses. "
            "Provided context has been ignored in this fallback."
        )
        # Audit the fallback response if possible
        try:
            from util.audit import audit_ai_call

            try:
                audit_ai_call(system_message, user_message, resp_text, served_from_fallback=True, metadata={"reason": "disabled"})
            except Exception:
                logger.debug("Audit write failed for disabled fallback (ignored)")
        except Exception:
            # If audit module isn't available, skip silently
            pass
        return resp_text

    ask_url = f"{base_url.rstrip('/')}/ask"
    session = get_singleton_session()  # Get singleton session
    try:
        resp = session.get(ask_url, params={"q": user_message, "system": system_message}, timeout=10)
        resp.raise_for_status()
        try:
            data = resp.json()
            for key in ("answer", "text", "result", "response"):
                if key in data:
                    resp_text = data[key]
                    break
            else:
                resp_text = str(data)
        except ValueError:
            resp_text = resp.text

        # Audit successful response (best-effort)
        try:
            from util.audit import audit_ai_call

            try:
                audit_ai_call(system_message, user_message, resp_text, served_from_fallback=False, metadata={"endpoint": ask_url})
            except Exception:
                logger.debug("Audit write failed for successful response (ignored)")
        except Exception:
            pass

        return resp_text
    except requests.exceptions.RequestException as e:
        logger.warning("Failed to invoke local Crawl4AI server at %s: %s", ask_url, e)
        resp_text = (
            "Crawl4AI server unreachable. Install/start a local Crawl4AI server or set CRAWL4AI_ENABLE=0 to disable integration. "
            "Alternatively, install sentence-transformers for local embeddings/responses."
        )
        # Audit the unreachable/fallback event
        try:
            from util.audit import audit_ai_call

            try:
                audit_ai_call(system_message, user_message, resp_text, served_from_fallback=True, metadata={"error": str(e)})
            except Exception:
                logger.debug("Audit write failed for unreachable fallback (ignored)")
        except Exception:
            pass
        return resp_text
