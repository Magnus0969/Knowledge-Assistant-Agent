from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_agent(index_path="faiss_index"):
    llm = Ollama(model="mistral")

    vectordb = FAISS.load_local(
        index_path,
        HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"),
        allow_dangerous_deserialization=True  # ✅ explicitly allow pickle loading
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    return qa_chain

def ask(qa_chain, query):
    return qa_chain.run(query)
