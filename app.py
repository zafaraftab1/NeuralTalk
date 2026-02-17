import time
import uuid
from datetime import datetime
from pathlib import Path
import re
import json
import io
import subprocess
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

st.set_page_config(page_title="Advanced AI Assistant", page_icon="🤖", layout="wide")


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".json",
    ".csv",
    ".log",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".js",
    ".ts",
    ".java",
    ".go",
    ".rs",
    ".sql",
}
MAX_FILE_CHARS = 10000
SETTINGS_PATH = Path(".user_settings.json")
CODE_RULES = (
    "Coding output rules: "
    "Return clean final code only when user asks for code. "
    "Do not add explanatory text. "
    "Do not add line-by-line comments unless user explicitly asks for comments."
)


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_chat(title: str = "New Chat") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "messages": [],
        "created_at": now_ts(),
        "updated_at": now_ts(),
    }


def load_persisted_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def save_persisted_settings() -> None:
    keys = ["temperature", "max_tokens", "system_prompt", "context_turns", "stream_output", "model"]
    data = {k: st.session_state.get(k) for k in keys}
    try:
        SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception:
        # Non-fatal: UI should continue even if file write fails.
        pass


def init_state() -> None:
    if "chats" not in st.session_state:
        first = new_chat()
        st.session_state["chats"] = [first]
        st.session_state["active_chat_id"] = first["id"]

    defaults = {
        "system_prompt": (
            "You are a helpful AI coding assistant. "
            "When the user asks for code, return only the final code in fenced code blocks. "
            "Do not add extra explanation unless the user explicitly asks for explanation."
        ),
        "temperature": 0.2,
        "max_tokens": 256,
        "model": "deepseek-coder:1.3b",
        "base_url": "http://localhost:11434",
        "context_turns": 0,
        "stream_output": True,
        "session_started_at": now_ts(),
        "composer_index": 0,
        "submit_requested": False,
        "settings_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state["settings_loaded"]:
        persisted = load_persisted_settings()
        for key in ["temperature", "max_tokens", "system_prompt", "context_turns", "stream_output", "model"]:
            if key in persisted:
                st.session_state[key] = persisted[key]
        st.session_state["settings_loaded"] = True


@st.cache_data(ttl=20)
def list_ollama_models() -> list[str]:
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
            timeout=4,
        )
        if proc.returncode != 0:
            return []
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if len(lines) <= 1:
            return []
        models = []
        for line in lines[1:]:
            name = line.split()[0].strip()
            if name:
                models.append(name)
        return models
    except Exception:
        return []


def get_active_chat() -> dict:
    active_id = st.session_state["active_chat_id"]
    for chat in st.session_state["chats"]:
        if chat["id"] == active_id:
            return chat

    # Fallback if active chat id is stale
    first = st.session_state["chats"][0]
    st.session_state["active_chat_id"] = first["id"]
    return first


def touch_chat(chat: dict) -> None:
    chat["updated_at"] = now_ts()


def chat_history_text(chat: dict, limit_turns: int) -> str:
    messages = chat["messages"]
    # 0 means use full same-chat history for stronger conversation chaining.
    recent = messages if limit_turns == 0 else messages[-(limit_turns * 2) :]
    if not recent:
        return "(no prior messages)"

    lines = []
    for msg in recent:
        role = msg.get("role", "assistant").capitalize()
        content = msg.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_chain():
    template = (
        "System instruction:\n{system_prompt}\n\n"
        "Recent conversation:\n{history}\n\n"
        "User: {question}\nAssistant:"
    )
    prompt = PromptTemplate.from_template(template)
    llm = OllamaLLM(
        model=st.session_state["model"],
        base_url=st.session_state["base_url"],
        num_predict=int(st.session_state["max_tokens"]),
        temperature=float(st.session_state["temperature"]),
    )
    return prompt | llm


def append_message(chat: dict, role: str, content: str, latency: float | None = None):
    entry = {
        "role": role,
        "content": content,
        "ts": now_ts(),
    }
    if latency is not None:
        entry["latency_seconds"] = round(latency, 2)
    chat["messages"].append(entry)
    touch_chat(chat)



def _read_text_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    mime = (uploaded_file.type or "").lower()

    if suffix == ".pdf" or mime == "application/pdf":
        if PdfReader is None:
            return "PDF parsing unavailable: install `pypdf` to extract PDF text."
        try:
            raw = uploaded_file.getvalue()
            reader = PdfReader(io.BytesIO(raw))
            chunks = []
            total = 0
            for page in reader.pages:
                page_text = (page.extract_text() or "").strip()
                if not page_text:
                    continue
                chunks.append(page_text)
                total += len(page_text)
                if total >= MAX_FILE_CHARS:
                    break
            text = "\n\n".join(chunks)
            if not text:
                return "No extractable text found in PDF."
            if len(text) > MAX_FILE_CHARS:
                return text[:MAX_FILE_CHARS] + "\n...[truncated]"
            return text
        except Exception as exc:
            return f"PDF read error: {exc}"

    is_text_like = suffix in TEXT_EXTENSIONS or mime.startswith("text/") or mime in {
        "application/json",
        "application/xml",
    }

    if not is_text_like:
        return None

    raw = uploaded_file.getvalue()
    if not raw:
        return ""

    text = raw.decode("utf-8", errors="ignore")
    if len(text) > MAX_FILE_CHARS:
        return text[:MAX_FILE_CHARS] + "\n...[truncated]"
    return text


def compose_prompt(user_text: str, files) -> tuple[str, str]:
    attached_names = [f.name for f in files]
    display_text = user_text if user_text else "(file attachment only)"
    if attached_names:
        display_text += "\n\nAttached: " + ", ".join(attached_names)

    if not files:
        return user_text, display_text

    blocks = []
    for f in files:
        text_content = _read_text_file(f)
        meta = f"name={f.name}, type={f.type or 'unknown'}, size={f.size} bytes"
        if text_content is None:
            blocks.append(f"File ({meta}) could not be parsed as text.")
        else:
            blocks.append(f"File ({meta}) content:\n{text_content}")

    if not user_text:
        user_text = "Please analyze the attached files."

    full_prompt = (
        f"{user_text}\n\n"
        "The user attached files. Use these file contents/metadata in your answer:\n\n"
        + "\n\n".join(blocks)
    )
    return full_prompt, display_text


def generate_reply(chat: dict, question: str, on_update=None) -> str:
    chain = build_chain()
    effective_system_prompt = f"{CODE_RULES}\n\n{st.session_state['system_prompt']}"
    inputs = {
        "question": question,
        "system_prompt": effective_system_prompt,
        "history": chat_history_text(chat, int(st.session_state["context_turns"])),
    }

    if st.session_state["stream_output"] and hasattr(chain, "stream"):
        chunks = []
        for chunk in chain.stream(inputs):
            chunks.append(str(chunk))
            if on_update is not None:
                on_update("".join(chunks))
        return "".join(chunks).strip()

    response = chain.invoke(inputs)
    text = str(response).strip()
    if on_update is not None:
        on_update(text)
    return text


def set_chat_title_if_needed(chat: dict, user_text: str) -> None:
    if chat["title"] == "New Chat" and user_text.strip():
        chat["title"] = user_text.strip()[:32]


def render_content_blocks(content: str, target) -> None:
    code_pattern = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
    matches = list(code_pattern.finditer(content))
    if not matches:
        target.markdown(content)
        return

    # If code exists, show only the code blocks to avoid unnecessary text noise.
    for match in matches:
        language = match.group(1).strip() or None
        code_text = match.group(2).rstrip("\n")
        target.code(code_text, language=language)


def render_message(role: str, content: str, latency: float | None = None, target=None) -> None:
    if target is None:
        target = st
    if role == "user":
        col_spacer, col_right = target.columns([0.3, 0.7])
        with col_right:
            safe_text = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            st.markdown(f'<div class="user-pill">{safe_text}</div>', unsafe_allow_html=True)
    else:
        col_left, col_spacer = target.columns([0.78, 0.22])
        with col_left:
            render_content_blocks(content, st)


def request_submit() -> None:
    st.session_state["submit_requested"] = True


def parse_chat_value(chat_value):
    if chat_value is None:
        return "", []
    if isinstance(chat_value, str):
        return chat_value.strip(), []

    text = getattr(chat_value, "text", "")
    files = getattr(chat_value, "files", [])
    if not text and isinstance(chat_value, dict):
        text = chat_value.get("text", "")
        files = chat_value.get("files", [])
    return str(text).strip(), list(files or [])


init_state()
active_chat = get_active_chat()

content_left, content_mid, content_right = st.columns([1.2, 7.6, 1.2])
with content_mid:
    st.markdown(
        '<div class="title-wrap"><h1 class="app-title">AI Assistant (Ollama + LangChain)</h1></div>',
        unsafe_allow_html=True,
    )
st.markdown(
    """
    <style>
    .main .block-container {
        padding-bottom: 8rem;
    }
    [data-testid="stChatInput"] {
        max-width: 920px;
        margin-left: auto;
        margin-right: auto;
    }
    .title-wrap {
        max-width: 920px;
        margin-left: auto;
        margin-right: auto;
    }
    .app-title {
        margin: 0 0 0.2rem 0 !important;
    }
    [data-testid="stChatInput"] textarea {
        min-height: 2.85rem !important;
        font-size: 1rem !important;
        border-radius: 20px !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    [data-testid="stChatInputSubmitButton"] button {
        border-radius: 999px !important;
        width: 2.2rem !important;
        height: 2.2rem !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind] {
        min-height: 2.2rem;
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
        font-size: 0.95rem;
        border-radius: 999px !important;
    }
    div[data-testid="stHorizontalBlock"] input[placeholder="Ask anything..."] {
        height: 2.45rem;
        font-size: 1.03rem;
        border-radius: 999px !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    .user-pill {
        display: inline-block;
        margin-left: auto;
        width: fit-content;
        max-width: 80%;
        padding: 0.55rem 0.8rem;
        border-radius: 16px;
        background: rgba(60, 130, 255, 0.15);
        border: 1px solid rgba(60, 130, 255, 0.35);
        line-height: 1.45;
        word-break: break-word;
    }
    div[data-testid="stCodeBlock"] pre {
        font-size: 0.92rem !important;
        line-height: 1.5 !important;
        border-radius: 10px !important;
        border: 1px solid rgba(120, 120, 120, 0.35) !important;
    }
    div[data-testid="stCodeBlock"] code {
        font-family: "JetBrains Mono", "Fira Code", Menlo, Consolas, monospace !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
components.html(
    """
    <script>
    (function () {
      const doc = window.parent.document;

      const styleEnterButton = () => {
        const btn = doc.querySelector('[data-testid="stChatInputSubmitButton"] button');
        if (!btn) return;
        btn.style.background = "#e53935";
        btn.style.border = "1px solid #e53935";
        btn.style.color = "#ffffff";
        btn.style.borderRadius = "999px";
        btn.style.width = "2.2rem";
        btn.style.height = "2.2rem";
        const svg = btn.querySelector("svg");
        if (svg) svg.style.display = "none";
        let span = btn.querySelector(".enter-icon");
        if (!span) {
          span = doc.createElement("span");
          span.className = "enter-icon";
          btn.appendChild(span);
        }
        span.textContent = "⏎";
        span.style.fontSize = "1rem";
        span.style.fontWeight = "800";
        span.style.lineHeight = "1";
      };

      const oldQuote = doc.getElementById("rotating-quote-banner");
      if (oldQuote) oldQuote.remove();

      styleEnterButton();
      setInterval(styleEnterButton, 1200);
    })();
    </script>
    """,
    height=0,
)

with st.sidebar:
    st.header("Chats")
    if st.button("New Chat", use_container_width=True, type="primary"):
        chat = new_chat()
        st.session_state["chats"].insert(0, chat)
        st.session_state["active_chat_id"] = chat["id"]
        st.session_state["composer_index"] += 1
        st.rerun()

    for chat in sorted(st.session_state["chats"], key=lambda c: c["updated_at"], reverse=True):
        is_active = chat["id"] == st.session_state["active_chat_id"]
        label = f"{'• ' if is_active else ''}{chat['title']}"
        if st.button(label, key=f"chat_{chat['id']}", use_container_width=True):
            st.session_state["active_chat_id"] = chat["id"]
            st.session_state["composer_index"] += 1
            st.rerun()

    st.header("Settings")
    model_options = list_ollama_models()
    if st.session_state["model"] not in model_options:
        model_options = [st.session_state["model"], *model_options]
    if not model_options:
        model_options = [st.session_state["model"]]
    st.selectbox("Model", options=model_options, key="model")
    st.slider("Temperature", 0.0, 1.0, key="temperature", step=0.05)
    st.slider("Max tokens", 1, 4096, key="max_tokens", step=1)
    st.text_area("System prompt", key="system_prompt", height=140)
    st.caption(f"Model in use: {st.session_state['model']}")

    st.header("Advanced")
    st.slider("Context turns (0 = full chat memory)", 0, 20, key="context_turns")
    st.toggle("Stream output", key="stream_output")
    save_persisted_settings()

for msg in active_chat["messages"]:
    with content_mid:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        latency = msg.get("latency_seconds")
        render_message(role, content, latency=latency)

chat_value = st.chat_input("Ask anything...", accept_file="multiple")
user_text, attached_files = parse_chat_value(chat_value)

if chat_value and (user_text.strip() or attached_files):
    active_chat = get_active_chat()
    full_prompt, display_prompt = compose_prompt(user_text.strip(), attached_files)

    set_chat_title_if_needed(active_chat, user_text)
    with content_mid:
        render_message("user", display_prompt)

    with content_mid:
        assistant_placeholder = st.empty()
    with st.spinner("Thinking..."):
        start = time.time()
        try:
            answer = generate_reply(
                active_chat,
                full_prompt,
                on_update=lambda partial: render_message("assistant", partial, target=assistant_placeholder),
            )
        except Exception as exc:
            answer = f"Error: {exc}"
            render_message("assistant", answer, target=assistant_placeholder)
        elapsed = time.time() - start

    append_message(active_chat, "user", display_prompt)
    append_message(active_chat, "assistant", answer, latency=elapsed)
    st.rerun()
