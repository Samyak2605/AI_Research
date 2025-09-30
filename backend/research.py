from typing import List, Dict, Optional

import requests
try:
	from ddgs import DDGS  # preferred package name
except Exception:
	from duckduckgo_search import DDGS  # fallback
import trafilatura


def safe_get(url: str, timeout_s: int = 10) -> Optional[str]:
	try:
		resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"})
		if resp.status_code == 200 and resp.text:
			return resp.text
		return None
	except Exception:
		return None


def extract_text(html: str, url: str) -> Optional[str]:
	try:
		# Trafilatura handles boilerplate removal and readability
		return trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
	except Exception:
		return None


def web_research(query: str, top_k: int = 4, per_url_timeout_s: int = 10) -> List[Dict]:
	results: List[Dict] = []
	try:
		with DDGS() as ddgs:
			for r in ddgs.text(query, max_results=top_k):
				title = r.get("title") or r.get("source") or "Result"
				url = r.get("href") or r.get("url")
				if not url:
					continue
				html = safe_get(url, timeout_s=per_url_timeout_s)
				content = extract_text(html, url) if html else None
				results.append({"title": title, "url": url, "content": content or ""})
	except Exception:
		# Fail open with empty results so the app still answers
		pass
	return results
