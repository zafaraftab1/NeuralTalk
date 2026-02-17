<h1 align="center">Advanced Streamlit Chat App</h1>
<p align="center"><b>Ollama + LangChain + Streamlit</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-111111?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?style=for-the-badge" />
</p>

---

## What This App Does

A local AI coding/chat interface with:

- Multi-chat history in the sidebar
- Right-aligned user prompts and left-aligned assistant responses
- Centered chat input bar with built-in `+` upload control
- Red Enter/submit button with Enter icon (`⏎`)
- Streaming responses for lower perceived latency
- Markdown + syntax-highlighted code blocks with copy support
- PDF + text file reading into prompt context
- Persistent sidebar settings across app restarts
- Sidebar model selector from installed Ollama models

---

## Defaults

- **Default model:** `deepseek-coder:1.3b`
- **Also used model:** `llama3.2:3b`
- **Default temperature:** `0.2`
- **Default context turns:** `0` (`0 = full same-chat memory`)

---

## Tech Stack

- **UI:** Streamlit
- **LLM Orchestration:** LangChain
- **Local Runtime:** Ollama (`langchain_ollama.OllamaLLM`)
- **Language:** Python 3

---

## Project Structure

```text
.
├── app.py
├── README.md
└── .user_settings.json   # auto-created at runtime
```

---

## Quick Start

### 1) Create and activate environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install streamlit langchain langchain-core langchain-ollama pypdf
```

### 3) Start Ollama

```bash
ollama serve
```

### 4) Pull model (if needed)

```bash
ollama pull deepseek-coder:1.3b
```

### 5) Run app

```bash
streamlit run app.py
```

Open: `http://localhost:8501`

---

## Sidebar Settings

- `Model` (from installed Ollama models)
- `Temperature`
- `Max tokens`
- `System prompt`
- `Context turns (0 = full chat memory)`
- `Stream output`

### Persistence

Settings are saved to `.user_settings.json` and loaded automatically on restart.

### Multi-Model Use

This app supports switching between your installed models directly from the sidebar.
Current setup includes:

- `deepseek-coder:1.3b`
- `llama3.2:3b`

---

## Upload Behavior

Use the `+` inside the chat input to attach files.

Supported extraction:

- Text-like files: `.txt`, `.md`, `.py`, `.json`, `.csv`, `.yaml`, `.yml`, `.xml`, `.html`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.sql`
- PDF files: `.pdf` via `pypdf`

Behavior:

- Extracted text is appended to model prompt context
- Very large extracted text is truncated
- Non-text files fall back to metadata

---

## Code Output Behavior

The app enforces coding-oriented behavior in generation:

- For coding requests, return clean final code
- Avoid extra explanation unless user asks
- Avoid line-by-line comments unless requested
- If response includes fenced code, UI prioritizes code block rendering

---

## Troubleshooting

### Missing package

```bash
pip install <missing-package>
```

### No model response

- Ensure Ollama is running: `ollama serve`
- Verify models: `ollama list`
- Pull missing model: `ollama pull <model_name>`

### UI style not updated

- Refresh browser tab once
- Hard refresh if needed (`Cmd+Shift+R` on macOS)

---

## Security Notes

- Uploaded file text is included in prompts.
- Avoid uploading secrets in untrusted environments.

---

<p align="center"><b>Built for a fast local AI workflow.</b></p>
