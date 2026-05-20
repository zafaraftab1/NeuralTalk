# 🚀 Neural Talk - Complete Startup Guide

## Quick Start (30 seconds)

### Option 1: Using the Startup Script (Recommended)

```bash
chmod +x run.sh
./run.sh
```

### Option 2: Manual Steps

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies
pip install streamlit langchain-core langchain-ollama pypdf

# 3. Run the app
streamlit run app.py
```

---

## Prerequisites

### 1. **Python 3.10+**
Check your Python version:
```bash
python3 --version
```

### 2. **Ollama Installation**
Download and install Ollama from: https://ollama.ai

---

## Full Setup Instructions

### Step 1: Start Ollama Server

Open a **new terminal** and run:

```bash
ollama serve
```

You should see:
```
Loading model...
Listening on 127.0.0.1:11434 (http://localhost:11434)
```

### Step 2: Install a Model (First Time Only)

In **another terminal**, pull the default model:

```bash
ollama pull deepseek-coder:1.3b
```

Or choose another model:
```bash
ollama pull llama2
ollama pull mistral
ollama pull neural-chat
```

View installed models:
```bash
ollama list
```

### Step 3: Activate Virtual Environment

In your project directory:

```bash
source .venv/bin/activate
```

You should see `(.venv)` at the start of your terminal prompt.

### Step 4: Install Dependencies

```bash
pip install streamlit langchain-core langchain-ollama pypdf
```

### Step 5: Run the App

```bash
streamlit run app.py
```

The app will open at: **http://localhost:8501**

---

## Troubleshooting

### ❌ "Ollama is not running"

**Error:**
```
Connection Error: Cannot reach Ollama at http://localhost:11434
```

**Fix:**
- Make sure you ran `ollama serve` in another terminal
- Check: `curl http://localhost:11434/api/version`

---

### ❌ "No models appear in dropdown"

**Error:**
```
Model dropdown is empty
```

**Fix:**
```bash
ollama list                          # See installed models
ollama pull deepseek-coder:1.3b    # Install a model
```

---

### ❌ PDF extraction not working

**Fix:**
```bash
pip install pypdf
```

---

### ❌ Port 8501 already in use

**Fix:**
```bash
streamlit run app.py --server.port 8502
```

---

## Features Overview

### 💬 Chat Interface
- **Left side:** AI responses (light bubble)
- **Right side:** Your prompts (gradient bubble)
- Multiple chat threads in sidebar
- Full conversation history per chat

### ⚙️ Settings (Sidebar)

#### Model Controls
- **Model:** Switch between installed Ollama models
- **Temperature:** 0-1.0 (lower = more deterministic, higher = more creative)
- **Max tokens:** Maximum output length (1-4096)
- **System prompt:** Customize AI behavior

#### Advanced
- **Context turns:** How many previous messages to consider
  - `0` = full chat memory (recommended)
  - `5` = last 5 conversation pairs
- **Stream output:** Real-time response streaming

### 📎 File Attachments

Supported formats:
- Text: `.txt`, `.md`, `.py`, `.json`, `.csv`, `.log`, `.yaml`, `.xml`, `.html`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.sql`
- Documents: `.pdf`

Click the `+` icon in chat input to attach files.

### 🎨 UI Improvements

**Message Alignment (Fixed)**
- User messages: Right-aligned with gradient background
- AI messages: Left-aligned with light background
- Perfect bubble positioning like ChatGPT/Discord

**Code Blocks**
- Syntax highlighting
- Dark theme with blue border
- Proper spacing and formatting

---

## File Structure

```
Stream_Lit/
├── app.py                    # Main application
├── run.sh                    # Startup script
├── STARTUP_GUIDE.md          # This file
├── README.md                 # Project documentation
├── assets/
│   └── neuraltalk_logo.png   # App logo
├── .user_settings.json       # Persistent settings
└── .venv/                    # Python environment
```

---

## Default Settings

| Setting | Value |
|---------|-------|
| Model | deepseek-coder:1.3b |
| Base URL | http://localhost:11434 |
| Temperature | 0.2 |
| Max Tokens | 256 |
| Context Turns | 0 (full history) |
| Stream Output | Yes |

Change settings in the sidebar or they'll persist to `.user_settings.json`

---

## Tips & Best Practices

### 1. **Code Requests**
The app automatically detects coding prompts. When you ask:
- "Write Python code..."
- "Create a function..."
- "Debug this code..."

It switches to code-only response mode.

### 2. **File Analysis**
Attach code files or documents:
```
"Analyze this file" + attach file
"Fix the bugs" + attach python file
```

### 3. **Keep It Fast**
- Use smaller models (1-7B) for speed
- Lower `max_tokens` for faster responses
- Reduce `temperature` for consistency

### 4. **Better Responses**
- Customize the system prompt
- Use specific examples in your question
- Break complex problems into smaller steps

---

## Keyboard Shortcuts

| Action | Key |
|--------|-----|
| Send message | `Enter` or click ⏎ button |
| New chat | Click "New Chat" button |
| Clear input | `Escape` |

---

## Environment Variables (Optional)

```bash
# Use a different Ollama server
export OLLAMA_BASE_URL="http://your-server:11434"

# Set default model
export OLLAMA_MODEL="mistral"
```

---

## Logs & Debugging

View app logs:
```bash
streamlit run app.py --logger.level=debug
```

Check Ollama logs:
```bash
ollama list
ollama ps
```

---

## Next Steps

1. ✅ Start the app: `./run.sh`
2. 💬 Write your first prompt
3. ⚙️ Adjust settings in sidebar
4. 📎 Try attaching a file
5. 🎯 Explore different models

---

## Support

### Common Issues Checklist
- [ ] Python 3.10+ installed
- [ ] Ollama running (`ollama serve`)
- [ ] Model installed (`ollama list`)
- [ ] Port 8501 available
- [ ] Dependencies installed (`pip list | grep streamlit`)
- [ ] Virtual environment activated (`.venv`)

---

## For Developers

### Add a new model
Edit `app.py` line 131:
```python
"model": "your-model-name",
```

### Customize system prompt
Edit `app.py` lines 124-128

### Change UI theme
Edit CSS in `app.py` lines 634-655

### Modify response rules
Edit `CODE_RULES` on line 45

---

**Enjoy your local AI workspace! 🎉**

Last updated: May 2026

