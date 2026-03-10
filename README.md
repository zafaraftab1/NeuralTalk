<div align="center">

# Neural Talk

### A stylish local AI chat workspace built with Streamlit, Ollama, and LangChain

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-2F6FED?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF5C8A?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-111827?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LangChain-Orchestration-10B981?style=for-the-badge" />
</p>

<p>
  Bright visuals. Local models. Multi-chat workflow. File-aware prompting.
</p>

</div>

---

## Overview

`Neural Talk` is a local-first AI assistant interface for running Ollama models inside a custom Streamlit chat UI.

It is designed around a polished chat experience:

- multi-chat sidebar navigation
- colorful branded interface
- file attachments directly in the composer
- streamed model responses
- code-oriented response cleanup for coding prompts
- persistent user settings between runs

## Why This Project Feels Different

This is not a plain demo chat window. The app is set up as a more presentation-focused workspace with:

- a styled hero section and branded sidebar
- custom user and assistant message bubbles
- syntax-highlighted code blocks
- local Ollama model switching from the sidebar
- prompt augmentation from attached files

## Core Features

### Chat Experience

- Create and switch between multiple chats
- Keep per-session chat history in Streamlit state
- Use full same-chat memory by default
- Stream responses for lower perceived latency

### Model Controls

- Choose from installed Ollama models
- Adjust temperature
- Set max token output
- Customize the system prompt
- Control how many prior turns are sent as context

### File-Aware Prompts

Attach files using the built-in `+` button in the chat input.

Supported parsing:

- Text files: `.txt`, `.md`, `.py`, `.json`, `.csv`, `.log`, `.yaml`, `.yml`, `.xml`, `.html`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.sql`
- PDF files: `.pdf`

Behavior:

- extracted content is appended into the model prompt
- unsupported files fall back to metadata-only context
- extracted text is truncated to `10,000` characters
- file-only submissions automatically become: `Please analyze the attached files.`

### Code-Focused Output

For code-related prompts, the app switches into a stricter output mode that tries to:

- return a single fenced code block
- strip extra explanation around code
- normalize malformed fences
- preserve comments only if the prompt explicitly asks for them

## Current Defaults

The live app behavior in [app.py](/Users/zafaraftab/Stream_Lit/app.py) currently uses:

- Default model: `deepseek-coder:1.3b`
- Base URL: `http://localhost:11434`
- Default temperature: `0.2`
- Default max tokens: `256`
- Default context turns: `0` meaning full same-chat history
- Settings persistence file: `.user_settings.json`

## Project Structure

```text
.
├── app.py
├── assets/
│   └── neuraltalk_logo.png
├── README.md
└── .user_settings.json
```

## Quick Start

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install streamlit langchain-core langchain-ollama pypdf
```

### 3. Start Ollama

```bash
ollama serve
```

### 4. Pull the default model if needed

```bash
ollama pull deepseek-coder:1.3b
```

### 5. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501`

## Sidebar Controls

### Chats

- New chat creation
- Active chat switching
- Recent-chat ordering

### Settings

- `Model`
- `Temperature`
- `Max tokens`
- `System prompt`

### Advanced

- `Context turns`
- `Stream output`

## Notes

- Installed model discovery uses `ollama list`
- If Ollama is unavailable, the app returns a connection error in the chat
- Chat messages are not persisted to disk
- UI settings are persisted to `.user_settings.json`

## Troubleshooting

### No model appears in the selector

```bash
ollama list
```

### Ollama is not responding

```bash
ollama serve
```

### PDF text extraction is unavailable

```bash
pip install pypdf
```

## Security

Attached file contents are included in prompts sent to your local model. Do not upload secrets unless that is acceptable for your environment.

---

<div align="center">
  Built for a more beautiful local AI workflow.
</div>
