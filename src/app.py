import streamlit as st
import os

from pdf_ingest import load_pdf, embed_text
from scholar_agent import load_agent

st.set_page_config(page_title="Knowledge Assistant Agent", layout="wide")
st.title("📚 Knowledge Assistant Agent")

# === PDF Upload ===
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Indexing your document..."):
        # Save PDF to disk
        pdf_path = os.path.join("data", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process
        text = load_pdf(pdf_path)
        embed_text(text)
        st.session_state["pdf_indexed"] = True
        st.success("✅ PDF indexed! Ask me anything.")

# === Query Section ===
query = st.text_input("Your question:")
if query:
    if st.session_state.get("pdf_indexed", False):
        with st.spinner("Generating answer..."):
            try:
                qa_chain = load_agent()  # Load chain only when needed
                response = qa_chain.run(query)
                st.success(response)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Please upload and index a PDF first.")