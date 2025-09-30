import os
import re
import time
from typing import List, Dict, Iterable, Optional

import google.generativeai as genai


MAX_PROMPT_CHARS = 12000  # soft cap to keep under free-tier token limits
DEFAULT_MAX_OUTPUT_TOKENS = 1024


class GeminiAssistant:
	def __init__(
		self,
		api_key: Optional[str] = None,
		model_name: str = "gemini-1.5-flash-latest",
		max_retries: int = 0,
		temperature: float = 0.2,
		top_p: float = 0.9,
		max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
	) -> None:
		key = api_key or os.getenv("GEMINI_API_KEY")
		if not key:
			raise ValueError("GEMINI_API_KEY not provided. Set in env or pass to GeminiAssistant.")
		genai.configure(api_key=key)

		self.model_name = self._resolve_model_name(model_name)
		self.generation_config = {
			"temperature": float(temperature),
			"top_p": float(top_p),
			"max_output_tokens": int(max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS),
		}
		self.max_retries = max(0, int(max_retries))
		self.model = genai.GenerativeModel(self.model_name)

	def embed_texts(self, texts: List[str], model: str = "text-embedding-004") -> List[List[float]]:
		"""Return embeddings for a list of texts using Gemini embedding model.

		Falls back gracefully by returning empty lists on error for each text.
		"""
		embeddings: List[List[float]] = []
		for t in texts:
			try:
				res = genai.embed_content(model=model, content=t or "")
				vec = res.get("embedding") if isinstance(res, dict) else getattr(res, "embedding", None)
				if vec:
					embeddings.append(list(vec))
					continue
			except Exception:
				pass
			# ensure positional alignment
			embeddings.append([])
		return embeddings

	def generate_answer(
		self,
		messages: List[Dict[str, str]],
		system_prompt: str,
		context_snippets: Optional[List[str]] = None,
	) -> str:
		"""Non-streaming single-call generation that returns full text."""
		prompt = self._compose_prompt(system_prompt, messages, context_snippets)
		try:
			response = self.model.generate_content(
				prompt,
				generation_config=self.generation_config,
			)
			text = getattr(response, "text", None)
			return text or ""
		except Exception as e:
			return f"Error: {str(e)}"

	def _resolve_model_name(self, requested: str) -> str:
		"""Resolve to an available non-experimental model that supports generateContent.
		Prefers 1.5 flash/pro latest models and avoids experimental (exp) and 2.5 variants.
		"""
		preferred_order = [
			"gemini-1.5-flash-latest",
			"gemini-1.5-pro-latest",
		]
		try:
			available = list(genai.list_models())
			supported = [
				m.name for m in available
				if getattr(m, "supported_generation_methods", None)
				and "generateContent" in m.supported_generation_methods
			]
		except Exception:
			supported = [
				"models/gemini-1.5-flash-latest",
				"models/gemini-1.5-pro-latest",
			]

		normalized_supported = set()
		for name in supported:
			if any(tag in name for tag in ["exp", "2.5", "vision"]):
				continue
			normalized_supported.add(name)
			if name.startswith("models/"):
				normalized_supported.add(name.replace("models/", ""))

		candidates = [requested]
		if not requested.endswith("-latest"):
			candidates.append(requested + "-latest")
		candidates.extend([
			requested.replace("models/", ""),
			*preferred_order,
		])
		for c in candidates:
			if c in normalized_supported:
				return c if not c.startswith("models/") else c.replace("models/", "")

		for p in preferred_order:
			if p in normalized_supported:
				return p
		for name in normalized_supported:
			return name.replace("models/", "") if name.startswith("models/") else name
		raise ValueError("No supported Gemini model found for generateContent.")

	def _format_chat_history(self, messages: List[Dict[str, str]]) -> str:
		parts: List[str] = []
		for m in messages:
			role = m.get("role", "user")
			content = m.get("content", "")
			if role == "system":
				parts.append(f"System: {content}")
			elif role == "assistant":
				parts.append(f"Assistant: {content}")
			else:
				parts.append(f"User: {content}")
		joined = "\n\n".join(parts)
		return joined[-MAX_PROMPT_CHARS:]

	def _compose_prompt(self, system_prompt: str, messages: List[Dict[str, str]], context_snippets: Optional[List[str]]) -> str:
		context = ""
		if context_snippets:
			bullets = "\n".join([f"- {c[:2000]}" for c in context_snippets if c])
			context = f"\n\nContext (web snippets):\n{bullets}\n\n"
		chat = self._format_chat_history(messages)
		composed = f"{system_prompt}{context}\n\nConversation so far:\n{chat}\n\nAssistant:"
		return composed[:MAX_PROMPT_CHARS]

	def _parse_retry_delay_seconds(self, error_message: str) -> float:
		match = re.search(r"retry_delay.*?seconds: (\d+)", error_message)
		if match:
			try:
				return float(match.group(1))
			except Exception:
				return 5.0
		return 5.0

	def stream_answer(
		self,
		messages: List[Dict[str, str]],
		system_prompt: str,
		context_snippets: Optional[List[str]] = None,
	) -> Iterable[str]:
		prompt = self._compose_prompt(system_prompt, messages, context_snippets)

		attempt = 0
		while True:
			try:
				response = self.model.generate_content(
					prompt,
					generation_config=self.generation_config,
					stream=True,
				)
				for chunk in response:
					text = getattr(chunk, "text", None)
					if text:
						yield text
				return
			except Exception as e:
				msg = str(e)
				is_rate = ("quota" in msg.lower()) or ("rate" in msg.lower()) or ("429" in msg)
				if is_rate and attempt < self.max_retries:
					delay = self._parse_retry_delay_seconds(msg) * (2 ** attempt)
					time.sleep(delay)
					attempt += 1
					continue
				if is_rate:
					yield "\n\n> Rate limit/quota hit. Please wait and try again, or switch model."
				else:
					yield f"\n\n> Error: {msg}"
				return
