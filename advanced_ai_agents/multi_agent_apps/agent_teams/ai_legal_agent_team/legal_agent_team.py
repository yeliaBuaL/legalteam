import streamlit as st
from agno.agent import Agent
from agno.team import Team
from agno.run.agent import RunOutput
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.models.openai import OpenAIChat
from agno.knowledge.embedder.openai import OpenAIEmbedder
import tempfile
import os

# ----------------------------------
# OPTIONAL DUCKDUCKGO (SAFE)
# ----------------------------------
try:
    from agno.tools.duckduckgo import DuckDuckGoTools
    WEB_TOOLS = [DuckDuckGoTools()]
except Exception:
    WEB_TOOLS = []

COLLECTION_NAME = "legal_documents"

# ----------------------------------
# SECRET HELPERS (LOCAL + CLOUD)
# ----------------------------------
def get_openai_key():
    return st.secrets.get("OPENAI_API_KEY") or st.session_state.get("openai_api_key")

def get_qdrant_key():
    return st.secrets.get("QDRANT_API_KEY") or st.session_state.get("qdrant_api_key")

def get_qdrant_url():
    return st.secrets.get("QDRANT_URL") or st.session_state.get("qdrant_url")

# ----------------------------------
# SESSION STATE
# ----------------------------------
def init_session_state():
    defaults = {
        "openai_api_key": None,
        "qdrant_api_key": None,
        "qdrant_url": None,
        "vector_db": None,
        "knowledge_base": None,
        "legal_team": None,
        "processed_files": set(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ----------------------------------
# QDRANT INIT
# ----------------------------------
def init_qdrant():
    if not all([get_openai_key(), get_qdrant_key(), get_qdrant_url()]):
        return None

    os.environ["OPENAI_API_KEY"] = get_openai_key()

    return Qdrant(
        collection=COLLECTION_NAME,
        url=get_qdrant_url(),
        api_key=get_qdrant_key(),
        embedder=OpenAIEmbedder(
            id="text-embedding-3-small",
            api_key=get_openai_key(),
        ),
    )

# ----------------------------------
# DOCUMENT INGEST
# ----------------------------------
def ingest_pdf(uploaded_file, vector_db):
    os.environ["OPENAI_API_KEY"] = get_openai_key()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    kb = Knowledge(vector_db=vector_db)
    kb.add_content(path=path)

    os.unlink(path)
    return kb

# ----------------------------------
# ANALYSIS PROMPTS
# ----------------------------------
ANALYSIS_PROMPTS = {
    "Contract Review": "Review the contract and identify key terms, obligations, and potential issues.",
    "Legal Research": "Research relevant legal cases and precedents related to this document.",
    "Risk Assessment": "Analyze potential legal risks and liabilities in this document.",
    "Compliance Check": "Check this document for regulatory or legal compliance issues.",
}

# ----------------------------------
# MAIN APP
# ----------------------------------
def main():
    st.set_page_config(page_title="AI Legal Agent Team", layout="wide")
    init_session_state()

    st.title("AI Legal Agent Team 👨‍⚖️")

    # ---------- SIDEBAR ----------
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

        st.divider()

        st.header("🔍 Analysis Options")

        analysis_type = st.selectbox(
            "Analysis Type",
            [
                "Contract Review",
                "Legal Research",
                "Risk Assessment",
                "Compliance Check",
                "Custom Query",
            ],
        )

        st.subheader("🤖 Select Agents")
        agent_selection = {
            "Legal Researcher": st.checkbox("Legal Researcher", True),
            "Contract Analyst": st.checkbox("Contract Analyst", True),
            "Legal Strategist": st.checkbox("Legal Strategist", True),
        }

        selected_agents = [k for k, v in agent_selection.items() if v]

        if not selected_agents:
            st.warning("Select at least one agent")

    # ---------- MAIN ----------
    if not st.session_state.vector_db:
        st.info("👈 Enter API keys to begin")
        return

    uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])
    if not uploaded_file:
        st.info("👈 Upload a document to start analysis")
        return

    if uploaded_file.name not in st.session_state.processed_files:
        with st.spinner("📤 Processing document…"):
            kb = ingest_pdf(uploaded_file, st.session_state.vector_db)
            st.session_state.knowledge_base = kb
            st.session_state.processed_files.add(uploaded_file.name)

    kb = st.session_state.knowledge_base
    model = OpenAIChat(id="gpt-5")

    # ---------- AGENTS ----------
    all_agents = {
        "Legal Researcher": Agent(
            "Legal Researcher",
            model=model,
            tools=WEB_TOOLS,
            knowledge=kb,
            search_knowledge=True,
            markdown=True,
        ),
        "Contract Analyst": Agent(
            "Contract Analyst",
            model=model,
            knowledge=kb,
            search_knowledge=True,
            markdown=True,
        ),
        "Legal Strategist": Agent(
            "Legal Strategist",
            model=model,
            knowledge=kb,
            search_knowledge=True,
            markdown=True,
        ),
    }

    active_agents = [all_agents[name] for name in selected_agents]

    team = Team(
        name="Legal Team",
        model=model,
        members=active_agents,
        knowledge=kb,
        search_knowledge=True,
        markdown=True,
    )

    # ---------- QUERY ----------
    if analysis_type == "Custom Query":
        query = st.text_area("Enter your custom legal question")
    else:
        query = ANALYSIS_PROMPTS[analysis_type]
        st.info(query)

    if st.button("Analyze"):
        if not selected_agents:
            st.warning("Select at least one agent")
            return

        if not query:
            st.warning("Enter a query")
            return

        with st.spinner("🧠 Analyzing document…"):
            response: RunOutput = team.run(query)
            st.markdown(response.content or "")

# ----------------------------------
if __name__ == "__main__":
    main()
