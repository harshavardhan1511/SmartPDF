import streamlit as st
import tempfile
from main import create_vector_store_from_pdf, query_pdf_qa, summarize_pdf

# Page Config
st.set_page_config(page_title="📄 Smart PDF Assistant", layout="wide")
st.sidebar.title("📄 Smart PDF Assistant")

# Sidebar - Upload
with st.sidebar:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if st.button("Process") and uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.pdf_path = tmp.name

        with st.spinner("Processing..."):
            st.session_state.vectorstore, st.session_state.vector_path = create_vector_store_from_pdf(st.session_state.pdf_path)
            st.session_state.qa_history = []
            st.success("✅ PDF processed!")

    if st.button("Reset"):
        for key in ["pdf_path", "vectorstore", "vector_path", "summary", "qa_history"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

# CSS to fix the input area and avoid overlap
st.markdown("""
    <style>
        .fixed-input {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #0e1117;
            padding: 1rem 2rem;
            z-index: 100;
            border-top: 1px solid #444;
        }
        .block-container {
            padding-bottom: 150px; /* room for fixed input */
        }
        button[kind="primary"] {
            margin-top: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# If no PDF processed yet
if "pdf_path" not in st.session_state:
    st.markdown("""
    <div style="text-align: center; padding: 100px; color: #888;">
        <p>Upload and process a PDF to begin.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Summary block
    if "summary" in st.session_state:
        st.markdown("### 🔍 Summary")
        st.markdown(f"""
        <div style="
            background-color: #1a1a1a;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        ">{st.session_state.summary}</div>
        """, unsafe_allow_html=True)

    # QA History
    if "qa_history" in st.session_state:
        for q, a in st.session_state.qa_history:
            st.markdown(f"""
            <div style="
                background-color: #262730;
                color: #e1e1e1;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 10px;
            ">
                <strong>Q:</strong> {q}
            </div>
            <div style="
                background-color: #1a1a1a;
                color: white;
                padding: 15px;
                border-left: 4px solid #10a37f;
                border-radius: 6px;
                margin-bottom: 30px;
            ">
                <strong>A:</strong> {a}
            </div>
            """, unsafe_allow_html=True)

    # Summarize button (top right)
    st.markdown("---")
    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            st.session_state.summary = summarize_pdf(st.session_state.pdf_path)
            st.experimental_rerun()

# Fixed bottom input area
with st.container():
    st.markdown('<div class="fixed-input">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input("Ask a question:",  key="chat_input", placeholder="Type your question here...")
    with col2:
        if st.button("Get Answer", key="chat_button") and question:
            with st.spinner("Searching..."):
                answer, _ = query_pdf_qa(st.session_state.vector_path, question)
                if "qa_history" not in st.session_state:
                    st.session_state.qa_history = []
                st.session_state.qa_history.append((question, answer))
                st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)
