import streamlit as st
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.team import Team
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.models.openai import OpenAIChat
from agno.knowledge.embedder.openai import OpenAIEmbedder
import tempfile
import os

# -------------------------
# OPTIONAL DUCKDUCKGO (SAFE)
# -------------------------
try:
    from agno.tools.duckduckgo import DuckDuckGoTools
    WEB_TOOLS = [DuckDuckGoTools()]
except Exception:
    WEB_TOOLS = []


COLLECTION_NAME = "legal_documents"


# -------------------------
# 🔐 SECRET HELPERS
# -------------------------
def get_openai_key():
    return st.secrets.get("OPENAI_API_KEY") or st.session_state.get("openai_api_key")


def get_qdrant_key():
    return st.secrets.get("QDRANT_API_KEY") or st.session_state.get("qdrant_api_key")


def get_qdrant_url():
    return st.secrets.get("QDRANT_URL") or st.session_state.get("qdrant_url")


# -------------------------
# SESSION STATE
# -------------------------
def init_session_state():
    for key in [
        "openai_api_key",
        "qdrant_api_key",
        "qdrant_url",
        "vector_db",
        "legal_team",
        "knowledge_base",
        "processed_files",
    ]:
        if key not in st.session_state:
            st.session_state[key] = set() if key == "processed_files" else None


# -------------------------
# QDRANT INIT
# -------------------------
def init_qdrant():
    if not all([get_qdrant_key(), get_qdrant_url(), get_openai_key()]):
        return None

    os.environ["OPENAI_API_KEY"] = get_openai_key()

    try:
        return Qdrant(
            collection=COLLECTION_NAME,
            url=get_qdrant_url(),
            api_key=get_qdrant_key(),
            embedder=OpenAIEmbedder(
                id="text-embedding-3-small",
                api_key=get_openai_key(),
            ),
        )
    except Exception as e:
        st.error(f"🔴 Qdrant connection failed: {e}")
        return None


# -------------------------
# DOCUMENT INGEST
# -------------------------
def process_document(uploaded_file, vector_db: Qdrant):
    os.environ["OPENAI_API_KEY"] = get_openai_key()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    knowledge_base = Knowledge(vector_db=vector_db)

    with st.spinner("📤 Indexing document…"):
        knowledge_base.add_content(path=path)

    os.unlink(path)
    return knowledge_base


# -------------------------
# MAIN APP
# -------------------------
def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    init_session_state()

    st.title("AI Legal Agent Team 👨‍⚖️")

    # -------- SIDEBAR --------
    with st.sidebar:
        st.header("🔑 API Configuration")

        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key or "",
        )

        st.session_state.qdrant_api_key = st.text_input(
            "Qdrant API Key",
            type="password",
            value=st.session_state.qdrant_api_key or "",
        )

        st.session_state.qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url or "",
        )

        if not st.session_state.vector_db:
            st.session_state.vector_db = init_qdrant()

        if st.session_state.vector_db:
            st.success("✅ Connected to Qdrant")

    # -------- MAIN --------
    if not st.session_state.vector_db:
        st.info("👈 Add API keys to begin")
        return

    uploaded_file = st.file_uploader("Upload Legal Document", type=["pdf"])

    if not uploaded_file:
        st.info("Upload a PDF to start")
        return

    if uploaded_file.name not in st.session_state.processed_files:
        kb = process_document(uploaded_file, st.session_state.vector_db)
        st.session_state.knowledge_base = kb
        st.session_state.processed_files.add(uploaded_file.name)

        legal_researcher = Agent(
            name="Legal Researcher",
            role="Legal research specialist",
            model=OpenAIChat(id="gpt-5"),
            tools=WEB_TOOLS,
            knowledge=kb,
            search_knowledge=True,
            markdown=True,
        )

        contract_analyst = Agent(
            name="Contract Analyst",
            role="Contract analysis specialist",
            model=OpenAIChat(id="gpt-5"),
            knowledge=kb,
            search_knowledge=True,
            markdown=True,
        )

        legal_strategist = Agent(
            name="Legal Strategist",
            role="Legal strategy specialist",
            model=OpenAIChat(id="gpt-5"),
            knowledge=kb,
            search_knowledge=True,
            markdown=True,
        )

        st.session_state.legal_team = Team(
            name="Legal Team",
            model=OpenAIChat(id="gpt-5"),
            members=[legal_researcher, contract_analyst, legal_strategist],
            knowledge=kb,
            search_knowledge=True,
            markdown=True,
        )

    query = st.text_area("Ask a legal question about the document")

    if st.button("Analyze") and query:
        os.environ["OPENAI_API_KEY"] = get_openai_key()
        response: RunOutput = st.session_state.legal_team.run(query)
        st.markdown(response.content or "")


if __name__ == "__main__":
    main()
