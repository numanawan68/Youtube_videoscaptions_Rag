
import os
import subprocess
import json
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate



# ============================================================================
# STEP 1: LOAD ENVIRONMENT VARIABLES
# ============================================================================
# This loads your .env file which should contain GROQ_API_KEY
ENV = r"E:\gen_ai_projects\shared_keys\.env"
load_dotenv(ENV)

# Add your .env file path here if it's in a different location:
# load_dotenv('/path/to/your/.env')

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file. Please add it.")

print("✅ Environment variables loaded successfully!")


# ============================================================================
# STEP 2: EXTRACT YOUTUBE SUBTITLES
# ============================================================================
def extract_youtube_subtitles(youtube_url: str):
    """
    Extract subtitles from a YouTube video URL using yt-dlp.
    
    Args:
        youtube_url (str): Full YouTube URL (e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ)
    
    Returns:
        List of Document objects containing video content and metadata
    """
    print(f"\n📥 Extracting subtitles from: {youtube_url}")
    
    try:
        print("   Downloading video info and subtitles...")
        
        # Run yt-dlp command to get video info in JSON format
        cmd = [
            'yt-dlp',
            '--dump-json',
            '--no-warnings',
            youtube_url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            raise Exception(f"yt-dlp error: {result.stderr}")
        
        # Parse video info JSON
        video_info = json.loads(result.stdout)
        
        # Extract subtitle text
        # STEP 1: download actual subtitle file
        cmd = [
            "yt-dlp",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--skip-download",
            "--quiet",
            youtube_url
        ]

        subprocess.run(cmd, check=True)

        # STEP 2: read generated subtitle file
        import glob

        subtitle_files = glob.glob("*.vtt") + glob.glob("*.srt")

        if not subtitle_files:
            raise Exception("No subtitle file generated")

        with open(subtitle_files[0], "r", encoding="utf-8") as f:
            subtitle_text = f.read()
       
        
        # Create Document object with metadata
        video_title = video_info.get('title', 'Unknown')
        video_id = video_info.get('id', 'Unknown')
        video_duration = video_info.get('duration', 0)
        
        doc = Document(
            page_content=subtitle_text,
            metadata={
                'source': youtube_url,
                'title': video_title,
                'video_id': video_id,
                'duration': video_duration,
            }
        )
        
        print(f"✅ Successfully extracted subtitles from: {video_title}")
        print(f"   Video ID: {video_id} | Duration: {video_duration//60}min {video_duration%60}sec")
        print(f"   Subtitle length: {len(subtitle_text)} characters")
        print(f"   Preview: {subtitle_text[:200]}...")
        
        return [doc]
    
    except Exception as e:
        print(f"❌ Error extracting subtitles: {e}")
        print(f"⚠️  Make sure yt-dlp is installed: pip install yt-dlp")
        return None


# ============================================================================
# STEP 3: SPLIT TEXT INTO CHUNKS
# ============================================================================
def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into smaller chunks for better semantic search.
    
    Args:
        documents: List of Document objects from YouTube loader
        chunk_size (int): Number of characters per chunk (medium = 1000)
        chunk_overlap (int): Overlap between chunks to maintain context
    
    Returns:
        List of smaller Document chunks
    """
    print(f"\n📝 Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    
    # Debug: Check document content
    if documents:
        total_chars = sum(len(doc.page_content) for doc in documents)
        print(f"   Total characters in documents: {total_chars}")
        if total_chars == 0:
            print("   ⚠️  WARNING: Documents contain no text!")
            print(f"   First document preview: {str(documents[0])[:200]}")
    
    # RecursiveCharacterTextSplitter: Splits on character boundaries intelligently
    # It tries to keep sentences together (better than naive splitting)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]  # Order of preferred split points
    )
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks from documents")
    
    if len(chunks) > 0:
        print(f"   First chunk size: {len(chunks[0].page_content)} chars")
    
    return chunks


# ============================================================================
# STEP 4: CREATE EMBEDDINGS & STORE IN FAISS
# ============================================================================
def create_faiss_vectorstore(chunks, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Create embeddings for chunks and store them in FAISS vector database.
    
    Args:
        chunks: List of Document chunks
        embedding_model_name (str): HuggingFace embedding model name (free)
    
    Returns:
        FAISS vectorstore object for similarity search
    """
    # Check if chunks are empty
    if not chunks or len(chunks) == 0:
        print("\n❌ ERROR: No chunks available!")
        print("   This means subtitle extraction failed or returned empty text.")
        print("   Possible reasons:")
        print("   - Video has no subtitles")
        print("   - Video is age-restricted or private")
        print("   - yt-dlp failed to extract subtitles")
        raise ValueError("Cannot create vectorstore with empty chunks")
    
    print(f"\n🔗 Creating embeddings using: {embedding_model_name}")
    print("   (First time may take a minute to download the model...)")
    
    # HuggingFaceEmbeddings: Free embedding model from Hugging Face
    # all-MiniLM-L6-v2: Small, fast, and good quality for semantic search
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Create FAISS vector store from chunks
    # FAISS: Facebook AI Similarity Search (very fast, local, free)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"✅ FAISS vectorstore created with {len(chunks)} embeddings")
    return vectorstore


# ============================================================================
# STEP 5: INITIALIZE GROQ LLM
# ============================================================================
def initialize_groq_llm(model_name: str = "llama-3.1-8b-instant"):
    """
    Initialize Groq LLM for question answering.
    
    Args:
        model_name (str): Groq model to use (mixtral-8x7b-32768 is free and fast)
    
    Returns:
        ChatGroq LLM instance
    """
    print(f"\n🤖 Initializing Groq LLM: {model_name}")
    
    # ChatGroq: LangChain's Groq integration (free, very fast inference)
    llm = ChatGroq(
        temperature=0.7,  # Creativity level (0=deterministic, 1=random)
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        max_tokens=1024  # Limit response length
    )
    
    print(f"✅ Groq LLM initialized successfully")
    return llm

# ============================================================================
# STEP 6: CREATE RETRIEVAL QA CHAIN
# ============================================================================
def create_qa_chain(vectorstore, llm):
    """
    Create a RetrievalQA chain that combines retrieval + generation.
    
    Args:
        vectorstore: FAISS vectorstore for retrieving relevant chunks
        llm: Groq LLM for generating answers
    
    Returns:
        RetrievalQA chain object
    """
    print("\n⛓️  Creating Retrieval QA chain...")
    
    # RetrievalQA: Combines retriever (FAISS) + LLM
    # Process: 1) User asks question 2) Retrieve top-k relevant chunks 3) Pass to LLM 4) LLM answers
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = put all retrieved docs into context
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Retrieve top 3 chunks
        return_source_documents=True  # Return which chunks were used
        
        )
    
    print("✅ QA chain created successfully")
    return qa_chain


# ============================================================================
# STEP 7: MAIN RAG FUNCTION
# ============================================================================
def youtube_rag(youtube_url: str):
    """
    Main function to set up YouTube RAG system and start Q&A.
    
    Args:
        youtube_url (str): YouTube video URL
    """
    print("=" * 70)
    print("🎬 YOUTUBE RAG SYSTEM - STARTING")
    print("=" * 70)
    
    # Step 1: Extract subtitles
    documents = extract_youtube_subtitles(youtube_url)
    if not documents:
        return
    
    # Step 2: Chunk documents
    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
    
    # Step 3: Create FAISS vectorstore with embeddings
    vectorstore = create_faiss_vectorstore(chunks)
    
    # Step 4: Initialize Groq LLM
    llm = initialize_groq_llm()
    
    # Step 5: Create QA chain
    qa_chain = create_qa_chain(vectorstore, llm)
    
    print("\n" + "=" * 70)
    print("✅ RAG SYSTEM READY! Start asking questions (type 'exit' to quit)")
    print("=" * 70)
    
    # Interactive Q&A loop
    while True:
        print("\n")
        user_question = input("❓ Your question: ").strip()
        
        if user_question.lower() == 'exit':
            print("\n👋 Goodbye!")
            break
        
        if not user_question:
            print("⚠️  Please enter a valid question")
            continue
        
        try:
            print("\n🔍 Searching for relevant information...")
            
            # Run QA chain
            result = qa_chain.invoke({"query": user_question})
            
            # Extract answer and source documents
            answer = result["result"]
            source_docs = result["source_documents"]
            
            print("\n" + "-" * 70)
            print("📤 ANSWER:")
            print("-" * 70)
            print(answer)
            
            # Show which chunks were used (sources)
            print("\n📚 SOURCES USED:")
            print("-" * 70)
            for i, doc in enumerate(source_docs, 1):
                # Get chunk preview (first 200 chars)
                preview = doc.page_content[:200].replace("\n", " ")
                print(f"\n{i}. {preview}...")
            
            print("\n" + "=" * 70)
        
        except Exception as e:
            print(f"\n❌ Error processing question: {e}")
            print("Please try again.")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # REPLACE THIS WITH YOUR YOUTUBE VIDEO URL
    YOUTUBE_URL = "https://www.youtube.com/watch?v=gvnye8U30bc"
    
    # Uncomment and add your URL:
    # YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    # Start the RAG system
    youtube_rag(YOUTUBE_URL)
