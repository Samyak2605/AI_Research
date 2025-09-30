import os
import time
from typing import List, Dict, Optional

import streamlit as st

from backend.gemini_client import GeminiAssistant
from backend.research import web_research


APP_TITLE = "AI Research Assistant (Gemini)"
APP_SUBTITLE = "Ask research questions. Optionally pull in fresh web context."

# Defaults for simplified sidebar
DEFAULT_RESEARCH_ENABLED = True
DEFAULT_RESEARCH_K = 3
DEFAULT_CRAWL_TIMEOUT = 8


def get_api_key_from_inputs() -> Optional[str]:
	# Priority: session input > env var > st.secrets
	api_key = st.session_state.get("gemini_api_key")
	if api_key:
		return api_key
	api_key = os.getenv("GEMINI_API_KEY")
	if api_key:
		return api_key
	api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
	return api_key


def init_session_state():
	if "messages" not in st.session_state:
		st.session_state.messages = []  # [{role, content}]
	if "gemini_api_key" not in st.session_state:
		st.session_state.gemini_api_key = None
	if "research_results" not in st.session_state:
		st.session_state.research_results = []


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
		if st.button("Clear chat", use_container_width=True):
			st.session_state.messages = []
			st.session_state.research_results = []
			st.experimental_rerun()

		return {
			"api_key": get_api_key_from_inputs(),
			"model": model,
			"enable_research": DEFAULT_RESEARCH_ENABLED,
			"research_k": DEFAULT_RESEARCH_K,
			"crawl_timeout": DEFAULT_CRAWL_TIMEOUT,
			"max_retries": 0,
			"temperature": 0.2,
			"top_p": 0.9,
		}


def render_chat(messages: List[Dict[str, str]]):
	for msg in messages:
		with st.chat_message(msg["role"]):
			st.markdown(msg["content"])


def build_system_prompt(enable_research: bool) -> str:
	base = (
		"You are an expert AI research assistant. Be concise, cite sources with markdown links, and "
		"when using web context, synthesize key insights before answering."
	)
	if enable_research:
		base += " You may use the provided web snippets as context if relevant."
	return base


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
		) 

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
						time.sleep(0.01)
					st.session_state.messages.append({"role": "assistant", "content": accumulated})
				except Exception as e:
					placeholder.markdown(f"❌ Error: {str(e)}")


if __name__ == "__main__":
	main()
