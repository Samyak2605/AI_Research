# AI Research Assistant (Gemini)

A Streamlit-based research assistant that uses Google Gemini for generation and optional web research for fresh context.

## Setup

1. Create a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Provide your Gemini API key (free tier available):
   - Option A: Export env var

```bash
export GEMINI_API_KEY="YOUR_KEY"
```

   - Option B: Enter it in the app sidebar when running.

## Run

```bash
streamlit run app.py
```

- In the sidebar, choose model, toggle web research, and set result count.
- Ask questions in the chat input. Streaming responses appear in real time.

## Notes
- Web research uses DuckDuckGo search + Trafilatura extraction. If fetching fails, the assistant still answers.
- No data is stored; refresh or Clear chat to reset session.
