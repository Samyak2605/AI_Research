import os
import time
from typing import List, Dict, Optional

import streamlit as st

from backend.gemini_client import GeminiAssistant
from backend.research import web_research, format_snippets_for_citation, select_top_k_snippets
from backend.vector_store import LocalFaissStore


APP_TITLE = "AI Research Assistant (Gemini)"
APP_SUBTITLE = "Ask research questions. Optionally pull in fresh web context."

# Defaults for simplified sidebar
DEFAULT_RESEARCH_ENABLED = False
DEFAULT_RESEARCH_K = 2
DEFAULT_CRAWL_TIMEOUT = 5
DEFAULT_DB_DIR = ".ai_store"
DEFAULT_INDEX_PATH = os.path.join(DEFAULT_DB_DIR, "faiss.index")


def get_api_key_from_inputs() -> Optional[str]:
	# Priority: session input > env var (no secrets file to avoid crashes)
	api_key = st.session_state.get("gemini_api_key")
	if api_key:
		return api_key
	api_key = os.getenv("GEMINI_API_KEY")
	if api_key:
		return api_key
	return None


def init_session_state():
	if "messages" not in st.session_state:
		st.session_state.messages = []  # [{role, content}]
	if "gemini_api_key" not in st.session_state:
		st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY") or None
	if "research_results" not in st.session_state:
		st.session_state.research_results = []
	if "vector_store" not in st.session_state:
		st.session_state.vector_store = None


def ui_header():
	st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")
	left, right = st.columns([0.75, 0.25])
	with left:
		st.title(APP_TITLE)
		st.caption(APP_SUBTITLE)
	with right:
		st.markdown("""
		<style>
			.stButton>button { border-radius: 8px; padding: 0.6rem 1rem; }
			.stChatMessage { border-radius: 10px; }
			.sidebar .stTextInput>div>div>input { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }
		</style>
		""", unsafe_allow_html=True)


def ui_sidebar() -> Dict:
	with st.sidebar:
		st.header("Settings")
		api_key_input = st.text_input(
			"GEMINI_API_KEY",
			value=st.session_state.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", ""),
			type="password",
			placeholder="Enter your key",
		)
		if api_key_input:
			st.session_state.gemini_api_key = api_key_input

		model = st.selectbox(
			"Model",
			["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"],
			index=0,
		)

		st.divider()
		enable_research = st.checkbox("Enable web research (slower)", value=DEFAULT_RESEARCH_ENABLED)
		research_k = st.slider("Results per query", min_value=1, max_value=5, value=DEFAULT_RESEARCH_K)
		crawl_timeout = st.slider("Per-URL timeout (s)", min_value=3, max_value=15, value=DEFAULT_CRAWL_TIMEOUT)

		st.divider()
		fast_mode = st.checkbox("Fast mode (shorter answers, no research)", value=not DEFAULT_RESEARCH_ENABLED)
		if fast_mode:
			enable_research = False

		st.divider()
		if st.button("Clear chat", use_container_width=True):
			st.session_state.messages = []
			st.session_state.research_results = []
			st.experimental_rerun()

		return {
			"api_key": get_api_key_from_inputs(),
			"model": model,
			"enable_research": enable_research,
			"research_k": research_k,
			"crawl_timeout": crawl_timeout,
			"max_retries": 0,
			"temperature": 0.2,
			"top_p": 0.9,
			"persist_research": True,
			"requery_k": 5,
			"fast_mode": fast_mode,
			"max_output_tokens": 512 if fast_mode else 1024,
		}


def render_chat(messages: List[Dict[str, str]]):
	for msg in messages:
		with st.chat_message(msg["role"]):
			st.markdown(msg["content"])
			if msg.get("meta"):
				with st.expander("Sources"):
					for r in msg["meta"]:
						st.markdown(f"- [{r['title']}]({r['url']})")


def build_system_prompt(enable_research: bool) -> str:
	base = (
		"You are an expert AI research assistant. Be concise, cite sources with markdown links, and "
		"when using web context, synthesize key insights before answering."
	)
	if enable_research:
		base += " You may use the provided web snippets as context if relevant."
	return base


def ensure_vector_store(assistant: GeminiAssistant) -> Optional[LocalFaissStore]:
	try:
		os.makedirs(DEFAULT_DB_DIR, exist_ok=True)
		vs = st.session_state.get("vector_store")
		if vs is not None:
			return vs
		# bootstrap with embedding dimension from a sample embedding
		probe = assistant.embed_texts(["probe"])
		dim = len(probe[0]) if probe and probe[0] else 768
		vs = LocalFaissStore(dim=dim, index_path=DEFAULT_INDEX_PATH)
		st.session_state.vector_store = vs
		return vs
	except Exception as _e:
		return None


def persist_research_snippets(assistant: GeminiAssistant, snippets: List[Dict]):
	vs = ensure_vector_store(assistant)
	if not vs or not snippets:
		return
	texts = [(s.get("content") or "").strip() for s in snippets if (s.get("content") or "").strip()]
	if not texts:
		return
	embs = assistant.embed_texts(texts)
	metas = [
		{"title": s.get("title"), "url": s.get("url"), "content": s.get("content")}
		for s in snippets if (s.get("content") or "").strip()
	]
	# filter out failed embeddings
	filtered_embs = []
	filtered_meta = []
	for e, m in zip(embs, metas):
		if e:
			filtered_embs.append(e)
			filtered_meta.append(m)
	if filtered_embs:
		vs.add(filtered_embs, filtered_meta)


def requery_past_research(assistant: GeminiAssistant, query: str, k: int = 5) -> List[Dict]:
	vs = ensure_vector_store(assistant)
	if not vs or not query:
		return []
	q_emb = assistant.embed_texts([query])
	if not q_emb or not q_emb[0]:
		return []
	results = vs.search(q_emb, k=k)
	if not results:
		return []
	return [m for _score, m in results[0]]


def fact_check_answer(assistant: GeminiAssistant, answer: str, snippets: List[Dict]) -> str:
	prompt = (
		"You are a factuality checker. Compare the assistant's answer to the provided web snippets. "
		"Identify any unsupported claims, contradictions, or missing citations. Output a brief checklist "
		"with corrections and cite snippet titles or URLs inline. Keep it concise."
	)
	messages = [
		{"role": "system", "content": prompt},
		{"role": "user", "content": f"Answer to check:\n\n{answer}"},
		{"role": "user", "content": f"Snippets:\n\n{format_snippets_for_citation(select_top_k_snippets(snippets, 5))}"},
	]
	return assistant.generate_answer(messages, system_prompt="", context_snippets=None)


def generate_report(assistant: GeminiAssistant, question: str, answer: str, snippets: List[Dict]) -> str:
	prompt = (
		"Generate a concise research report with sections: Introduction, Key Findings, Methods, Pros/Cons, Sources. "
		"Ground claims in the provided snippets when applicable and include markdown links."
	)
	messages = [
		{"role": "system", "content": prompt},
		{"role": "user", "content": f"Research question:\n{question}"},
		{"role": "user", "content": f"Assistant draft answer:\n{answer}"},
		{"role": "user", "content": f"Snippets:\n\n{format_snippets_for_citation(select_top_k_snippets(snippets, 8))}"},
	]
	return assistant.generate_answer(messages, system_prompt="", context_snippets=None)


def main():
	init_session_state()
	ui_header()
	settings = ui_sidebar()

	api_key = settings["api_key"]
	assistant = None
	api_ready = bool(api_key)
	if api_ready:
		assistant = GeminiAssistant(
			api_key=api_key,
			model_name=settings["model"],
			max_retries=settings["max_retries"],
			temperature=settings["temperature"],
			top_p=settings["top_p"],
			max_output_tokens=settings.get("max_output_tokens", 1024),
		) 
		# prepare vector store once API is ready
		ensure_vector_store(assistant)

	render_chat(st.session_state.messages)

	# Chat input area
	prompt = st.chat_input("Ask a research question...")
	if prompt:
		st.session_state.messages.append({"role": "user", "content": prompt})
		with st.chat_message("user"):
			st.markdown(prompt)

		# Optional web research
		context_snippets: List[str] = []
		research_meta: List[Dict] = []
		if settings["enable_research"]:
			with st.status("Researching the web...", expanded=False) as status:
				results = web_research(
					query=prompt,
					top_k=settings["research_k"],
					per_url_timeout_s=settings["crawl_timeout"],
				)
				context_snippets = [r["content"] for r in results if r.get("content")]
				research_meta = results
				status.update(label="Research complete", state="complete")

			if research_meta:
				with st.expander("Web context used"):
					for r in research_meta:
						st.markdown(f"- [{r['title']}]({r['url']})")

			# persist snippets
			if settings.get("persist_research") and research_meta and assistant:
				persist_research_snippets(assistant, research_meta)

		# Generate answer
		with st.chat_message("assistant"):
			placeholder = st.empty()
			accumulated = ""

			if not api_ready:
				placeholder.markdown("⚠️ Please set your `GEMINI_API_KEY` in the sidebar.")
			else:
				try:
					stream = assistant.stream_answer(
						messages=st.session_state.messages,
						system_prompt=build_system_prompt(settings["enable_research"]),
						context_snippets=context_snippets,
					)
					for chunk in stream:
						accumulated += chunk
						placeholder.markdown(accumulated)
						# smooth streaming
						time.sleep(0.002)
					# store assistant message with meta for sources
					st.session_state.messages.append({"role": "assistant", "content": accumulated, "meta": research_meta})
				except Exception as e:
					placeholder.markdown(f"❌ Error: {str(e)}")

				# Fact-check pass
				if accumulated.strip() and assistant:
					with st.expander("Fact-check report"):
						fc = fact_check_answer(assistant, accumulated, research_meta)
						st.markdown(fc or "No report.")

				# Report generation action
				col1, col2 = st.columns([0.5, 0.5])
				with col1:
					if st.button("Generate Research Report", use_container_width=True):
						report = generate_report(assistant, prompt, accumulated, research_meta)
						st.download_button(
							label="Download Report (Markdown)",
							data=report,
							file_name="research_report.md",
							mime="text/markdown",
						)
				with col2:
					if st.button("Re-query Past Research", use_container_width=True):
						past = requery_past_research(assistant, prompt, k=settings.get("requery_k", 5))
						if past:
							st.subheader("Related past research")
							for r in past:
								st.markdown(f"- [{r.get('title','Source')}]({r.get('url','')})")


if __name__ == "__main__":
	main()
