# AI Assistant (Streamlit + Ollama + LangChain)

Local chat UI for Ollama models with a Streamlit frontend and LangChain prompt orchestration.

## What It Does

- Runs a local chat assistant against an Ollama model
- Supports multiple chats in the sidebar
- Persists UI settings in `.user_settings.json`
- Streams responses when enabled
- Accepts multiple file attachments in the chat input
- Extracts text from common text files and PDFs
- Tries to force clean fenced-code output for coding requests
- Renders assistant code blocks with syntax highlighting

## Current App Behavior

The app is implemented in [app.py](/Users/zafaraftab/Stream_Lit/app.py). It:

- Defaults to `deepseek-coder:1.3b`
- Connects to Ollama at `http://localhost:11434`
- Uses full same-chat history by default with `context_turns = 0`
- Saves these settings between runs:
  - `model`
  - `temperature`
  - `max_tokens`
  - `system_prompt`
  - `context_turns`
  - `stream_output`

## Project Structure

```text
.
├── app.py
├── assets/
│   └── neuraltalk_logo.png
├── README.md
└── .user_settings.json   # created/updated at runtime
```

## Requirements

- Python 3.10+
- Ollama installed and running locally
- At least one pulled Ollama model

Python packages used by the app:

- `streamlit`
- `langchain-core`
- `langchain-ollama`
- `pypdf`

## Quick Start

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install streamlit langchain-core langchain-ollama pypdf
```

Start Ollama:

```bash
ollama serve
```

Pull the default model if needed:

```bash
ollama pull deepseek-coder:1.3b
```

Run the app:

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## Sidebar Controls

### Chats

- Create a new chat
- Switch between existing chats
- Active chats are ordered by most recently updated

### Settings

- `Model`
- `Temperature`
- `Max tokens`
- `System prompt`

### Advanced

- `Context turns (0 = full chat memory)`
- `Stream output`

## File Attachments

Use the built-in `+` control in the chat input to attach multiple files.

Supported text extraction:

- Text-like files: `.txt`, `.md`, `.py`, `.json`, `.csv`, `.log`, `.yaml`, `.yml`, `.xml`, `.html`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.sql`
- PDF files: `.pdf` through `pypdf`

Behavior:

- Parsed file contents are appended to the prompt sent to the model
- Non-text files fall back to metadata-only context
- Extracted file content is truncated to 10,000 characters
- If only files are attached, the app sends: `Please analyze the attached files.`

## Coding Responses

If the prompt looks like a coding request, the app switches into a stricter code-oriented mode. That mode attempts to:

- request exactly one fenced code block
- strip surrounding explanation when possible
- normalize malformed code fences
- preserve comments only when the prompt asks for comments or explanation

The code-request detection is keyword-based, so it is heuristic rather than guaranteed.

## Notes

- Model discovery in the sidebar uses `ollama list`
- If Ollama is unavailable, the app shows a connection error in the chat
- Chat history is kept in Streamlit session state and is not persisted to disk

## Troubleshooting

If no models appear:

```bash
ollama list
```

If Ollama is not responding:

```bash
ollama serve
```

If PDF parsing fails, ensure `pypdf` is installed:

```bash
pip install pypdf
```

## Security

Attached file contents are inserted into prompts sent to your local model. Do not upload secrets unless that is acceptable for your local environment.
