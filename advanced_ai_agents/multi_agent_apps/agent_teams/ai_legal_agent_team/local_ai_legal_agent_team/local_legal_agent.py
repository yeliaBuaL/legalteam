import streamlit as st
from agno.agent import Agent
from agno.team import Team
from agno.run.agent import RunOutput
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.models.ollama import Ollama
from agno.knowledge.embedder.ollama import OllamaEmbedder
import tempfile
import os


# -------------------------
# INIT
# -------------------------
def get_vector_db():
    return Qdrant(
        collection="legal_knowledge",
        url="http://localhost:6333",
        embedder=OllamaEmbedder(model="openhermes"),
    )


def ingest_pdf(uploaded_file, vector_db):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    kb = Knowledge(vector_db=vector_db)
    kb.add_content(path=path)
    os.unlink(path)
    return kb


# -------------------------
# APP
# -------------------------
def main():
    st.set_page_config("Local Legal Agent", layout="wide")
    st.title("🦙 Local Legal Agent Team")

    vector_db = get_vector_db()
    st.success("✅ Connected to local Qdrant")

    uploaded_file = st.file_uploader("Upload a legal PDF", type=["pdf"])
    if not uploaded_file:
        st.info("Upload a document to start")
        return

    kb = ingest_pdf(uploaded_file, vector_db)

    # -------- AGENTS --------
    model = Ollama(id="llama3.1:8b")

    team = Team(
        name="Legal Team",
        model=model,
        knowledge=kb,
        search_knowledge=True,
        members=[
            Agent("Legal Researcher", model=model, knowledge=kb, search_knowledge=True),
            Agent("Contract Analyst", model=model, knowledge=kb, search_knowledge=True),
            Agent("Legal Strategist", model=model, knowledge=kb, search_knowledge=True),
        ],
        markdown=True,
    )

    query = st.text_area("Ask a question about the document")

    if st.button("Analyze") and query:
        with st.spinner("Analyzing…"):
            response: RunOutput = team.run(query)
            st.markdown(response.content or "")


if __name__ == "__main__":
    main()

