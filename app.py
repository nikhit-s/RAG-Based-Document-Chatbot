import os
import tempfile
import requests
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Set page config
st.set_page_config(page_title="Multi-Document Chatbot", page_icon="🤖")

def load_documents(uploaded_files):
    """Load and process uploaded documents."""
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        if uploaded_file.name.lower().endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.name.lower().endswith(('.doc', '.docx')):
            loader = Docx2txtLoader(tmp_file_path)
        elif uploaded_file.name.lower().endswith('.txt'):
            loader = TextLoader(tmp_file_path)
        else:
            st.warning(f"Unsupported file format: {uploaded_file.name}")
            os.unlink(tmp_file_path)
            continue

        try:
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        finally:
            os.unlink(tmp_file_path)
    
    return documents

def process_documents(documents):
    """Process and split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def is_ollama_running() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1)
        return r.status_code == 200
    except Exception:
        return False


def setup_qa_chain(chunks, backend: str, openai_model: str = "gpt-3.5-turbo", ollama_model: str = "mistral", gemini_model: str = "gemini-1.5-flash"):
    """Set up the question-answering chain for the chosen backend."""
    # Select embeddings
    if backend == "OpenAI":
        embeddings = OpenAIEmbeddings()
    elif backend == "Gemini":
        # Google Generative AI embeddings require the full model path with 'models/' prefix
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    else:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Select LLM
    if backend == "OpenAI":
        llm = ChatOpenAI(temperature=0.2, model_name=openai_model)
    elif backend == "Gemini":
        llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.2)
    else:
        llm = Ollama(model=ollama_model, temperature=0.2)

    # Memory and chain
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return qa_chain

def main():
    st.title("📄 Multi-Document Chatbot")
    st.write("Upload multiple documents and ask questions about their content.")
    
    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for backend, API key and file upload
    with st.sidebar:
        st.header("Settings")
        
        backend = st.radio("Backend", options=["Local (Ollama)", "Google Gemini", "OpenAI"], index=0)

        # Defaults and API keys
        openai_api_key = None
        gemini_api_key = "AIzaSyAtIiXnRveCW-4BjFx4KQamXJ8NTS_WA5k"
        openai_model = "gpt-3.5-turbo"
        gemini_model = "gemini-1.5-flash"
        ollama_model = "llama3.1"

        if backend == "OpenAI":
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            openai_model = st.text_input("OpenAI model", value=openai_model)
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            else:
                st.info("No OpenAI API key provided. You can use 'Google Gemini' or 'Local (Ollama)' instead.")
        elif backend == "Google Gemini":
            if gemini_api_key:
                os.environ["GOOGLE_API_KEY"] = gemini_api_key
                gemini_model = st.text_input("Gemini model", value=gemini_model)
            else:
                gemini_api_key = st.text_input("Gemini API Key", type="password")
                st.warning("Provide your Gemini API key or switch to 'Local (Ollama)'. Get a key at https://ai.google.dev/")
        else:
            ollama_model = st.text_input("Ollama model", value=ollama_model)
            if not is_ollama_running():
                st.warning(
                    "Ollama doesn't seem to be running. Install from https://ollama.com, then run 'ollama serve' in a terminal and 'ollama pull %s' to download the model." % ollama_model
                )

        # File upload
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT files",
            type=["pdf", "docx", "doc", "txt"],
            accept_multiple_files=True
        )
        
        if st.button("Process Documents") and uploaded_files:
            # Validate backend requirements
            if backend == "OpenAI" and not openai_api_key:
                st.error("Please enter your OpenAI API key or switch to 'Local (Ollama)' or 'Google Gemini'.")
            elif backend == "Google Gemini" and not gemini_api_key:
                st.error("Please enter your Gemini API key or switch to 'Local (Ollama)'.")
            else:
                with st.spinner("Processing documents..."):
                    documents = load_documents(uploaded_files)
                    if documents:
                        chunks = process_documents(documents)
                        st.session_state.qa_chain = setup_qa_chain(
                            chunks,
                            backend=(
                                "OpenAI" if backend == "OpenAI" else (
                                    "Gemini" if backend == "Google Gemini" else "Local"
                                )
                            ),
                            openai_model=openai_model,
                            ollama_model=ollama_model,
                            gemini_model=gemini_model,
                        )
                        st.success(f"Processed {len(documents)} documents with {len(chunks)} chunks.")
                    else:
                        st.error("No valid documents were processed.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        if st.session_state.qa_chain:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get response from QA chain
                        result = st.session_state.qa_chain({"question": prompt, "chat_history": st.session_state.messages})
                        response = result["answer"]
                        
                        # Display response
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Show source documents
                        with st.expander("Source Documents"):
                            for i, doc in enumerate(result["source_documents"], 1):
                                st.write(f"**Source {i}**")
                                st.caption(f"Document: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'N/A')}")
                                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                st.write("---")
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        else:
            with st.chat_message("assistant"):
                st.warning("Please upload and process documents first.")
                st.session_state.messages.append({"role": "assistant", "content": "Please upload and process documents first."})

if __name__ == "__main__":
    main()
