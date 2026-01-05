import streamlit as st
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.team import Team
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.openai import OpenAIChat
from agno.knowledge.embedder.openai import OpenAIEmbedder
import tempfile
import os

# =========================================================
# 🔐 API HELPERS (ONLY ADDITION)
# =========================================================
def get_openai_key():
    return st.secrets.get("OPENAI_API_KEY") or st.session_state.openai_api_key

def get_qdrant_key():
    return st.secrets.get("QDRANT_API_KEY") or st.session_state.qdrant_api_key

def get_qdrant_url():
    return st.secrets.get("QDRANT_URL") or st.session_state.qdrant_url


def init_session_state():
    """Initialize session state variables"""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = None
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = None
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'legal_team' not in st.session_state:
        st.session_state.legal_team = None
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()


COLLECTION_NAME = "legal_documents"


def init_qdrant():
    """Initialize Qdrant client with configured settings."""
    if not all([get_qdrant_key(), get_qdrant_url(), get_openai_key()]):
        return None
    try:
        vector_db = Qdrant(
            collection=COLLECTION_NAME,
            url=get_qdrant_url(),
            api_key=get_qdrant_key(),
            embedder=OpenAIEmbedder(
                id="text-embedding-3-small",
                api_key=get_openai_key()
            )
        )
        return vector_db
    except Exception as e:
        st.error(f"🔴 Qdrant connection failed: {str(e)}")
        return None


def process_document(uploaded_file, vector_db: Qdrant):
    """Process document, create embeddings and store in Qdrant"""
    if not get_openai_key():
        raise ValueError("OpenAI API key not provided")

    os.environ['OPENAI_API_KEY'] = get_openai_key()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        st.info("Loading and processing document...")

        knowledge_base = Knowledge(vector_db=vector_db)

        with st.spinner('📤 Loading documents into knowledge base...'):
            knowledge_base.add_content(path=temp_file_path)
            st.success("✅ Documents stored successfully!")

        try:
            os.unlink(temp_file_path)
        except Exception:
            pass

        return knowledge_base

    except Exception as e:
        st.error(f"Document processing error: {str(e)}")
        raise Exception(f"Error processing document: {str(e)}")


def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    init_session_state()

    st.title("AI Legal Agent Team 👨‍⚖️")

    with st.sidebar:
        st.header("🔑 API Configuration")

        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key if st.session_state.openai_api_key else "",
            help="Enter your OpenAI API key"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key

        qdrant_key = st.text_input(
            "Qdrant API Key",
            type="password",
            value=st.session_state.qdrant_api_key if st.session_state.qdrant_api_key else "",
            help="Enter your Qdrant API key"
        )
        if qdrant_key:
            st.session_state.qdrant_api_key = qdrant_key

        qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url if st.session_state.qdrant_url else "",
            help="Enter your Qdrant instance URL"
        )
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url

        if all([get_qdrant_key(), get_qdrant_url()]):
            if not st.session_state.vector_db:
                st.session_state.vector_db = init_qdrant()
                if st.session_state.vector_db:
                    st.success("Successfully connected to Qdrant!")

        st.divider()

        if all([get_openai_key(), st.session_state.vector_db]):
            st.header("📄 Document Upload")
            uploaded_file = st.file_uploader("Upload Legal Document", type=['pdf'])

            if uploaded_file:
                if uploaded_file.name not in st.session_state.processed_files:
                    with st.spinner("Processing document..."):
                        knowledge_base = process_document(uploaded_file, st.session_state.vector_db)
                        st.session_state.knowledge_base = knowledge_base
                        st.session_state.processed_files.add(uploaded_file.name)

                        legal_researcher = Agent(
                            name="Legal Researcher",
                            role="Legal research specialist",
                            model=OpenAIChat(id="gpt-5"),
                            tools=[DuckDuckGoTools()],
                            knowledge=knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Find and cite relevant legal cases and precedents",
                                "Provide detailed research summaries with sources",
                                "Reference specific sections from the uploaded document",
                                "Always search the knowledge base for relevant information"
                            ],
                            debug_mode=True,
                            markdown=True
                        )

                        contract_analyst = Agent(
                            name="Contract Analyst",
                            role="Contract analysis specialist",
                            model=OpenAIChat(id="gpt-5"),
                            knowledge=knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Review contracts thoroughly",
                                "Identify key terms and potential issues",
                                "Reference specific clauses from the document"
                            ],
                            markdown=True
                        )

                        legal_strategist = Agent(
                            name="Legal Strategist",
                            role="Legal strategy specialist",
                            model=OpenAIChat(id="gpt-5"),
                            knowledge=knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Develop comprehensive legal strategies",
                                "Provide actionable recommendations",
                                "Consider both risks and opportunities"
                            ],
                            markdown=True
                        )

                        st.session_state.legal_team = Team(
                            name="Legal Team Lead",
                            model=OpenAIChat(id="gpt-5"),
                            members=[legal_researcher, contract_analyst, legal_strategist],
                            knowledge=knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Coordinate analysis between team members",
                                "Provide comprehensive responses",
                                "Ensure all recommendations are properly sourced",
                                "Reference specific parts of the uploaded document",
                                "Always search the knowledge base before delegating tasks"
                            ],
                            debug_mode=True,
                            markdown=True
                        )

                        st.success("✅ Document processed and team initialized!")

            st.divider()
            st.header("🔍 Analysis Options")
            analysis_type = st.selectbox(
                "Select Analysis Type",
                [
                    "Contract Review",
                    "Legal Research",
                    "Risk Assessment",
                    "Compliance Check",
                    "Custom Query"
                ]
            )
        else:
            st.warning("Please configure all API credentials to proceed")

    if not all([get_openai_key(), st.session_state.vector_db]):
        st.info("👈 Please configure your API credentials in the sidebar to begin")
    elif not uploaded_file:
        st.info("👈 Please upload a legal document to begin analysis")
    elif st.session_state.legal_team:
        analysis_configs = {
            "Contract Review": "Review this contract and identify key terms, obligations, and potential issues.",
            "Legal Research": "Research relevant cases and precedents related to this document.",
            "Risk Assessment": "Analyze potential legal risks and liabilities in this document.",
            "Compliance Check": "Check this document for regulatory compliance issues.",
            "Custom Query": None,
        }

        if analysis_type == "Custom Query":
            user_query = st.text_area("Enter your specific query:")
        else:
            user_query = analysis_configs[analysis_type]

        if st.button("Analyze"):
            if analysis_type == "Custom Query" and not user_query:
                st.warning("Please enter a query")
            else:
                os.environ['OPENAI_API_KEY'] = get_openai_key()
                response: RunOutput = st.session_state.legal_team.run(user_query)
                st.m
