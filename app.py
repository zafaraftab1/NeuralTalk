# An advanced AI assistant demo using Streamlit, Ollama, and LangChain.
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
CODE_REQUEST_HINT = (
    "IMPORTANT: Return exactly one fenced code block and nothing else. "
    "No explanation text, no notes, no analysis."
)
CODE_REQUEST_KEYWORDS = (
    "code",
    "function",
    "script",
    "program",
    "implement",
    "generator",
    "class",
    "api",
    "algorithm",
    "query",
    "sql",
    "fix",
    "debug",
    "refactor",
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


def build_code_chain():
    template = (
        "System instruction:\n{system_prompt}\n\n"
        "User coding request:\n{question}\n\n"
        "Output rules:\n"
        "- Return only one markdown fenced code block.\n"
        "- No explanation text.\n"
        "- No prefixes like 'Sure' or 'Here is'.\n\n"
        "Assistant:"
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


def is_code_request(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    if not text:
        return False
    return any(keyword in text for keyword in CODE_REQUEST_KEYWORDS)


def wants_code_comments(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    if not text:
        return False
    return "comment" in text or "explain" in text


def _normalize_fences(text: str) -> str:
    # Normalize malformed fences such as ` ` ` python into ```python.
    return re.sub(r"`\s*`\s*`", "```", text)


def _sanitize_code(code: str, allow_comments: bool) -> str:
    lines = code.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue

        if not allow_comments:
            if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("--"):
                continue
            if "#" in line:
                head, tail = line.split("#", 1)
                # Drop verbose narrative comments but keep short code comments if present.
                if len(tail.split()) > 4:
                    line = head.rstrip()
                    stripped = line.strip()
                    if not stripped:
                        continue
            if re.search(r"\b(user|assistant|console)\b", stripped, flags=re.IGNORECASE):
                if not re.search(r"[=(){}\[\]:,+\-*/]", stripped):
                    continue
            # Remove obviously broken assignment line like: x =
            if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*$", line):
                continue

        cleaned.append(line.rstrip())

    # Trim extra blank lines at top/bottom.
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned)


def _extract_first_fenced_code(text: str) -> str | None:
    normalized = _normalize_fences(text)
    pattern = re.compile(r"```[ \t]*([a-zA-Z0-9_+\-]*)[ \t]*\n?(.*?)```", re.DOTALL)
    match = pattern.search(normalized)
    if not match:
        return None
    language = (match.group(1) or "").strip()
    code = (match.group(2) or "").strip("\n")
    if language:
        return f"```{language}\n{code}\n```"
    return f"```\n{code}\n```"


def postprocess_code_response(text: str, code_mode: bool, allow_comments: bool = False) -> str:
    if not code_mode:
        return text

    normalized = _normalize_fences(text).strip()

    # Fast path: already valid fenced code from the model.
    full_fence = re.fullmatch(r"```[ \t]*[a-zA-Z0-9_+\-]*[ \t]*\n?.*?```", normalized, re.DOTALL)
    if full_fence:
        return normalized

    # Fallback: extract first fenced block if model added surrounding prose.
    fenced = _extract_first_fenced_code(text)
    if fenced:
        return fenced

    # Last resort: detect likely code and wrap as fenced block.
    lines = [line for line in normalized.splitlines() if line.strip()]
    code_start_tokens = (
        "def ",
        "class ",
        "import ",
        "from ",
        "if ",
        "for ",
        "while ",
        "return ",
        "const ",
        "let ",
        "var ",
        "function ",
        "#include",
        "package ",
        "public ",
        "private ",
        "SELECT ",
        "INSERT ",
        "UPDATE ",
        "DELETE ",
    )
    start_idx = None
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(code_start_tokens):
            start_idx = idx
            break

    if start_idx is not None:
        code = "\n".join(lines[start_idx:]).strip()
        code = _sanitize_code(code, allow_comments=allow_comments)
        return f"```python\n{code}\n```"

    return text


def generate_reply(
    chat: dict,
    question: str,
    on_update=None,
    code_mode: bool = False,
    allow_comments: bool = False,
) -> str:
    try:
        effective_question = (
            f"{question}\n\n{CODE_REQUEST_HINT}"
            if code_mode
            else question
        )
        if code_mode:
            chain = build_code_chain()
            effective_system_prompt = f"{st.session_state['system_prompt']}\n\n{CODE_RULES}"
            inputs = {
                "question": effective_question,
                "system_prompt": effective_system_prompt,
            }
        else:
            chain = build_chain()
            effective_system_prompt = f"{CODE_RULES}\n\n{st.session_state['system_prompt']}"
            inputs = {
                "question": effective_question,
                "system_prompt": effective_system_prompt,
                "history": chat_history_text(chat, int(st.session_state["context_turns"])),
            }

        if st.session_state["stream_output"] and hasattr(chain, "stream"):
            chunks = []
            for chunk in chain.stream(inputs):
                chunks.append(str(chunk))
                if on_update is not None:
                    on_update("".join(chunks))
            text = "".join(chunks).strip()
            text = postprocess_code_response(text, code_mode, allow_comments=allow_comments)
            if on_update is not None:
                on_update(text)
            return text

        response = chain.invoke(inputs)
        text = str(response).strip()
        text = postprocess_code_response(text, code_mode, allow_comments=allow_comments)
        if on_update is not None:
            on_update(text)
        return text
    except Exception as e:
        error_msg = f"Connection Error: Cannot reach Ollama at {st.session_state['base_url']}. Please ensure Ollama is running.\n\nDetails: {str(e)}"
        if on_update is not None:
            on_update(error_msg)
        return error_msg


def set_chat_title_if_needed(chat: dict, user_text: str) -> None:
    if chat["title"] == "New Chat" and user_text.strip():
        chat["title"] = user_text.strip()[:32]


def render_content_blocks(content: str, target) -> None:
    code_pattern = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
    matches = list(code_pattern.finditer(content))

    # Wrap text content in assistant message bubble
    if not matches:
        safe_text = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        target.markdown(
            f'<div class="message-row assistant-row"><div class="assistant-message-bubble">{safe_text}</div></div>',
            unsafe_allow_html=True,
        )
        return

    # If code exists, alternate between text and code blocks
    last_end = 0
    for match in matches:
        # Show text before code block
        before_text = content[last_end:match.start()].strip()
        if before_text:
            safe_text = before_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            target.markdown(
                f'<div class="message-row assistant-row"><div class="assistant-message-bubble">{safe_text}</div></div>',
                unsafe_allow_html=True,
            )

        # Use Streamlit native code rendering for syntax highlighting.
        language = match.group(1).strip() or None
        code_text = match.group(2).rstrip("\n")
        target.code(code_text, language=language)
        last_end = match.end()

    # Show remaining text after last code block
    after_text = content[last_end:].strip()
    if after_text:
        safe_text = after_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        target.markdown(
            f'<div class="message-row assistant-row"><div class="assistant-message-bubble">{safe_text}</div></div>',
            unsafe_allow_html=True,
        )


def render_message(role: str, content: str, latency: float | None = None, target=None) -> None:
    if target is None:
        target = st

    if role == "user":
        safe_text = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        target.markdown(
            f'<div class="message-row user-row"><div class="user-message-bubble">{safe_text}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        render_content_blocks(content, target)


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

chat_rail = st.container()
with chat_rail:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-glow hero-glow-a"></div>
            <div class="hero-glow hero-glow-b"></div>
            <div class="title-wrap">
                <div class="hero-kicker">Local-first creative coding studio</div>
                <h1 class="app-title">Neural Talk</h1>
                <p class="hero-subtitle">A vivid Ollama chat workspace with multi-thread memory, file-aware prompts, and cleaner code responses.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --chat-rail-width: 920px;
        --page-bg: radial-gradient(circle at top left, rgba(255, 144, 91, 0.30), transparent 24%),
                   radial-gradient(circle at 85% 12%, rgba(65, 205, 255, 0.28), transparent 22%),
                   radial-gradient(circle at 50% 100%, rgba(255, 78, 145, 0.22), transparent 30%),
                   linear-gradient(135deg, #fff4dd 0%, #ffe8ef 34%, #eaf5ff 68%, #eefdf6 100%);
        --panel-bg: rgba(255, 255, 255, 0.58);
        --panel-border: rgba(255, 255, 255, 0.62);
        --panel-shadow: 0 24px 80px rgba(194, 88, 51, 0.16);
        --headline: #1f1d2b;
        --text: #433b55;
        --muted: #7a6f8f;
        --user-gradient: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 42%, #ffb347 100%);
        --assistant-gradient: linear-gradient(135deg, rgba(255,255,255,0.88) 0%, rgba(245, 251, 255, 0.88) 100%);
        --assistant-border: rgba(115, 135, 173, 0.18);
        --accent: #ff5e7e;
        --accent-2: #25b7ff;
        --accent-3: #14c38e;
        --code-bg: rgba(21, 29, 48, 0.92);
        --code-border: rgba(86, 163, 255, 0.34);
    }
    html, body, [class*="css"]  {
        font-family: "Space Grotesk", sans-serif;
    }
    .stApp {
        background: var(--page-bg);
        color: var(--text);
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
            linear-gradient(rgba(255,255,255,0.18), rgba(255,255,255,0.18)),
            repeating-linear-gradient(
                90deg,
                rgba(255,255,255,0.06) 0,
                rgba(255,255,255,0.06) 1px,
                transparent 1px,
                transparent 90px
            );
        mask-image: linear-gradient(to bottom, rgba(0,0,0,0.75), rgba(0,0,0,0.2));
    }
    [data-testid="stAppViewContainer"] {
        background: transparent;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .main .block-container {
        max-width: var(--chat-rail-width);
        margin-left: auto;
        margin-right: auto;
        padding-top: 2rem;
        padding-bottom: 8rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(34, 25, 55, 0.86) 0%, rgba(32, 48, 79, 0.78) 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(28px);
    }
    [data-testid="stSidebar"] * {
        color: #fdf7ff;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #e7dcff;
    }
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stDownloadButton > button {
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        background: linear-gradient(135deg, rgba(255,255,255,0.16), rgba(255,255,255,0.05)) !important;
        color: #fff9ff !important;
        box-shadow: none !important;
        transition: transform 140ms ease, background 140ms ease, border-color 140ms ease !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] .stDownloadButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(255, 197, 106, 0.45) !important;
        background: linear-gradient(135deg, rgba(255, 170, 95, 0.34), rgba(255,255,255,0.10)) !important;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextArea label,
    [data-testid="stSidebar"] .stToggle label {
        color: #fffaf0 !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stTextArea textarea,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
        background: rgba(255, 255, 255, 0.10) !important;
        border-color: rgba(255,255,255,0.18) !important;
        color: #ffffff !important;
        border-radius: 16px !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: #d7c7f8 !important;
    }
    [data-testid="stChatInput"] {
        max-width: none !important;
        width: 100% !important;
        margin-left: auto;
        margin-right: auto;
    }
    [data-testid="stChatInput"] > div {
        max-width: var(--chat-rail-width) !important;
        width: 100% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    .title-wrap {
        max-width: var(--chat-rail-width);
        margin-left: auto;
        margin-right: auto;
        position: relative;
        z-index: 1;
    }
    .hero-shell {
        position: relative;
        margin: 0 auto 1.25rem auto;
        padding: 2.25rem 2rem 1.4rem 2rem;
        border-radius: 32px;
        background: linear-gradient(135deg, rgba(255,255,255,0.72), rgba(255,255,255,0.38));
        border: 1px solid var(--panel-border);
        box-shadow: var(--panel-shadow);
        overflow: hidden;
        backdrop-filter: blur(24px);
    }
    .hero-shell::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, rgba(255,255,255,0.18), transparent 40%, rgba(255,255,255,0.24));
        pointer-events: none;
    }
    .hero-glow {
        position: absolute;
        border-radius: 999px;
        filter: blur(8px);
        opacity: 0.9;
    }
    .hero-glow-a {
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(255, 156, 91, 0.46), transparent 70%);
        top: -60px;
        right: -20px;
    }
    .hero-glow-b {
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(45, 186, 255, 0.28), transparent 72%);
        bottom: -70px;
        left: -10px;
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.64);
        border: 1px solid rgba(255,255,255,0.8);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8d4a30;
    }
    .app-title {
        margin: 0.65rem 0 0.2rem 0 !important;
        font-size: clamp(2.4rem, 5vw, 4.5rem) !important;
        line-height: 0.98;
        color: var(--headline);
        letter-spacing: -0.05em;
    }
    .hero-subtitle {
        max-width: 720px;
        margin: 0;
        font-size: 1.02rem;
        line-height: 1.65;
        color: var(--text);
    }
    [data-testid="stChatInput"] textarea {
        min-height: 3.25rem !important;
        font-size: 1rem !important;
        border-radius: 24px !important;
        padding-top: 0.8rem !important;
        padding-bottom: 0.8rem !important;
        background: rgba(255,255,255,0.82) !important;
        border: 1px solid rgba(255,255,255,0.86) !important;
        box-shadow: 0 18px 40px rgba(111, 95, 151, 0.14) !important;
        color: var(--headline) !important;
    }
    [data-testid="stChatInputSubmitButton"] button {
        border-radius: 999px !important;
        width: 2.8rem !important;
        height: 2.8rem !important;
        box-shadow: 0 12px 26px rgba(255, 94, 126, 0.35) !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind] {
        min-height: 2.4rem;
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
    .message-row {
        display: flex;
        width: 100%;
        max-width: var(--chat-rail-width);
        margin: 0.35rem auto;
    }
    .assistant-row {
        justify-content: flex-start;
    }
    .user-row {
        justify-content: flex-end;
    }
    .user-message-bubble {
        display: inline-block;
        width: fit-content;
        max-width: 88%;
        padding: 0.95rem 1.15rem;
        border-radius: 24px 24px 8px 24px;
        background: var(--user-gradient);
        color: white;
        line-height: 1.5;
        word-break: break-word;
        font-size: 0.98rem;
        box-shadow: 0 20px 36px rgba(255, 112, 93, 0.26);
        border: 1px solid rgba(255,255,255,0.24);
    }
    .assistant-message-bubble {
        display: inline-block;
        width: fit-content;
        max-width: 88%;
        padding: 0.95rem 1.15rem;
        border-radius: 24px 24px 24px 8px;
        background: var(--assistant-gradient);
        color: var(--headline);
        line-height: 1.65;
        word-break: break-word;
        font-size: 0.98rem;
        border: 1px solid var(--assistant-border);
        box-shadow: 0 16px 36px rgba(94, 120, 160, 0.12);
        backdrop-filter: blur(10px);
    }
    div[data-testid="stCodeBlock"] pre {
        font-size: 0.92rem !important;
        line-height: 1.6 !important;
        border-radius: 22px !important;
        border: 1px solid var(--code-border) !important;
        background: var(--code-bg) !important;
        box-shadow: 0 22px 48px rgba(17, 25, 40, 0.24) !important;
        padding: 0.3rem !important;
    }
    div[data-testid="stCodeBlock"] code {
        font-family: "JetBrains Mono", "Fira Code", Menlo, Consolas, monospace !important;
        color: #eef7ff !important;
    }
    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }
    [data-testid="stChatInput"] {
        position: relative;
    }
    [data-testid="stChatInput"]::before {
        content: "";
        position: absolute;
        inset: -12px -14px;
        border-radius: 32px;
        background: linear-gradient(135deg, rgba(255,255,255,0.32), rgba(255,255,255,0.08));
        z-index: -1;
        filter: blur(6px);
    }
    @media (max-width: 900px) {
        .main .block-container {
            padding-top: 1.2rem;
        }
        .hero-shell {
            padding: 1.5rem 1.1rem 1.15rem 1.1rem;
            border-radius: 24px;
        }
        .hero-subtitle {
            font-size: 0.95rem;
        }
        .user-message-bubble,
        .assistant-message-bubble {
            max-width: 96%;
        }
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
        btn.style.background = "linear-gradient(135deg, #ff5e7e 0%, #ff8f57 52%, #ffcb52 100%)";
        btn.style.border = "1px solid rgba(255,255,255,0.55)";
        btn.style.color = "#ffffff";
        btn.style.borderRadius = "999px";
        btn.style.width = "2.8rem";
        btn.style.height = "2.8rem";
        btn.style.boxShadow = "0 14px 28px rgba(255, 94, 126, 0.32)";
        const svg = btn.querySelector("svg");
        if (svg) svg.style.display = "none";
        let span = btn.querySelector(".enter-icon");
        if (!span) {
          span = doc.createElement("span");
          span.className = "enter-icon";
          btn.appendChild(span);
        }
        span.textContent = "⏎";
        span.style.fontSize = "1.05rem";
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
    with chat_rail:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        latency = msg.get("latency_seconds")
        render_message(role, content, latency=latency)

chat_value = st.chat_input("Ask anything...", accept_file="multiple")
user_text, attached_files = parse_chat_value(chat_value)

if chat_value and (user_text.strip() or attached_files):
    active_chat = get_active_chat()
    full_prompt, display_prompt = compose_prompt(user_text.strip(), attached_files)
    code_mode = is_code_request(user_text)
    allow_comments = wants_code_comments(user_text)

    set_chat_title_if_needed(active_chat, user_text)
    with chat_rail:
        render_message("user", display_prompt)

    with chat_rail:
        assistant_placeholder = st.empty()

    with st.spinner("Thinking..."):
        start = time.time()
        try:
            answer = generate_reply(
                active_chat,
                full_prompt,
                code_mode=code_mode,
                allow_comments=allow_comments,
                on_update=lambda partial: render_message("assistant", partial, target=assistant_placeholder),
            )
        except Exception as exc:
            answer = f"Error: {exc}"
            with assistant_placeholder.container():
                render_message("assistant", answer, target=st)
        elapsed = time.time() - start

    append_message(active_chat, "user", display_prompt)
    append_message(active_chat, "assistant", answer, latency=elapsed)
    st.rerun()
