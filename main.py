import os, pickle, hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"  # Replace with your actual Groq API key

# Initialize LLMs
qa_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)
summary_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def create_vector_store_from_pdf(pdf_path):
    file_hash = get_file_hash(pdf_path)
    vector_path = f"vector_cache/faiss_{file_hash}.pkl"

    if os.path.exists(vector_path):
        with open(vector_path, "rb") as f:
            return pickle.load(f), vector_path

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embed_model)

    os.makedirs("vector_cache", exist_ok=True)
    with open(vector_path, "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore, vector_path

def query_pdf_qa(vector_path, question):
    with open(vector_path, "rb") as f:
        vectorstore = pickle.load(f)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQAWithSourcesChain.from_llm(llm=qa_llm, retriever=retriever)
    result = chain.invoke({"question": question})
    return result["answer"], result.get("sources", "")

def summarize_pdf(pdf_path):
    # Step 1: Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Step 2: Chunk into readable parts
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(docs)

    # Step 3: Define map and reduce prompts
    map_prompt = PromptTemplate.from_template(
        "Summarize the following part of the document:\n\n{text}"
    )
    reduce_prompt = PromptTemplate.from_template(
        "Combine these summaries into a cohesive, readable summary:\n\n{text}"
    )

    # Step 4: Create summarization chain
    chain = load_summarize_chain(
        summary_llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=reduce_prompt
    )

    # Step 5: Run chain and return result
    result = chain.run(docs_split)
    print("✅ Summary generated successfully")
    return result