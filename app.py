import streamlit as st
import os
import tempfile
import hashlib
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Ensure these are defined earlier in your code
MODELS = {
    "GPT-4o-Mini": "gpt-4o-mini",
    "GPT-4o": "gpt-4o",
    "GPT-4o-2024-08-06": "gpt-4o-2024-08-06"
}

# Define the system prompt
SYSTEM_PROMPT = """You are a helpful AI assistant. Your responses should be informative, 
friendly, and tailored to the user's questions. If you're unsure about something, 
it's okay to say so. When discussing document content, be specific and cite relevant parts. use emojis to make the conversation more engaging and fun."""

# Initialize session state variables
if 'file_uploaded' not in st.session_state:
    st.session_state['file_uploaded'] = False
if 'file_hash' not in st.session_state:
    st.session_state['file_hash'] = None
if 'embeddings_store' not in st.session_state:
    st.session_state['embeddings_store'] = {}
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

def process_file(file_content, file_name, api_key, selected_model):
    try:
        file_hash = get_file_hash(file_content)

        # Check if we already have embeddings for this file
        if file_hash in st.session_state['embeddings_store']:
            st.sidebar.info("Using existing embeddings for this file.")
            return st.session_state['embeddings_store'][file_hash]['chain']

        file_extension = os.path.splitext(file_name)[1].lower()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path)
        elif file_extension == '.csv':
            loader = CSVLoader(temp_file_path)
        elif file_extension in ['.docx', '.doc']:
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(temp_file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        chat = ChatOpenAI(temperature=0, model_name=MODELS[selected_model], openai_api_key=api_key)
        
        # Create a prompt template
        prompt_template = PromptTemplate(
            input_variables=["chat_history", "question", "context"],
            template=f"{SYSTEM_PROMPT}\n\nChat History: {{chat_history}}\nHuman: {{question}}\nContext: {{context}}\nAI: "
        )
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt_template}
        )

        # Store the embeddings and chain
        st.session_state['embeddings_store'][file_hash] = {
            'vectorstore': vectorstore,
            'chain': chain
        }

        return chain
    except Exception as e:
        raise Exception(f"An error occurred while processing the file: {str(e)}")
    finally:
        # Clean up the temporary file
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)

# Streamlit app
col1, col2 = st.columns([1, 12])
with col1:
    st.image("brain_boost_2.png", width=50)
with col2:
    st.markdown(
        """
        <a href="https://brain-boost-ai-experts.vercel.app" style="text-decoration: none; color: inherit;">
            <h1 style="margin-top: -24px;">Brain Boost</h1>
        </a>
        <a href="https://brain-boost-ai-experts.vercel.app" style="text-decoration: none; color: inherit;">
            <p style="font-size: 14px; margin: -20px 2px;">Click here to visit Brain Boost for AI solutions</p>
        </a>
        """,
        unsafe_allow_html=True
    )

st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
st.subheader("Chat with AI (and optionally your document)")



# Sidebar for configurations
st.sidebar.header("Configuration")

# API Key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Model selection
selected_model = st.sidebar.selectbox("Select AI Model", list(MODELS.keys()))

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "txt", "csv", "docx", "doc", "xlsx", "xls"])

# Process uploaded file
if uploaded_file is not None:
    file_content = uploaded_file.read()
    file_hash = get_file_hash(file_content)
    
    if file_hash != st.session_state['file_hash']:
        try:
            st.session_state['chain'] = process_file(file_content, uploaded_file.name, api_key, selected_model)
            st.session_state['file_uploaded'] = True
            st.session_state['file_hash'] = file_hash
            st.sidebar.success(f"{uploaded_file.name} uploaded and processed successfully!")
        except Exception as e:
            st.sidebar.error(str(e))
    else:
        st.sidebar.info("This file has already been processed.")
        st.session_state['chain'] = st.session_state['embeddings_store'][file_hash]['chain']

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state['messages'] = []
    st.session_state['chain'] = None
    st.session_state['file_uploaded'] = False
    st.session_state['file_hash'] = None
    st.rerun()

# Main chat interface
if api_key:
    # Display chat messages
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask a question about the document or chat with AI"):
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("🤔 AI is thinking..."):
                try:
                    if st.session_state['chain']:
                        response = st.session_state['chain']({"question": prompt})
                        answer = response['answer']
                    else:
                        chat = ChatOpenAI(temperature=0, model_name=MODELS[selected_model], openai_api_key=api_key)
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ]
                        answer = chat.invoke(messages).content
                    
                    message_placeholder.markdown(answer)
                    st.session_state['messages'].append({"role": "assistant", "content": answer})
                except Exception as e:
                    message_placeholder.error(f"An error occurred while generating the response: {str(e)}")

    st.info("You can start chatting now. If you want to chat about a specific document, upload it using the file uploader in the sidebar.")
else:
    st.markdown(
        """
        <div style="margin: 20px 0;">
            <div style="background-color: #FFFBE6; border-left: 5px solid #FFA500; padding: 10px; border-radius: 5px;">
                Please enter your OpenAI API key in the sidebar to start using the app.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )