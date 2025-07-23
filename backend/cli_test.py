import os
import hashlib
import asyncio
from pathlib import Path
from typing import List
import pymongo
from pymongo import MongoClient
import fitz  # PyMuPDF
from dotenv import load_dotenv
import logging
from datetime import datetime

# LangChain imports for modern OpenAI integration
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# MongoDB Atlas Vector Search and Conversational Chain imports
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFEmbeddingProcessor:
    """
    Handles PDF processing, embedding generation, and MongoDB storage
    """
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI embeddings using LangChain
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Using the latest embedding model
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Initialize text splitter for better chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize MongoDB client
        mongodb_uri = os.getenv('MONGODB_URI')
        if not mongodb_uri:
            raise ValueError("MONGODB_URI environment variable is required")
        
        self.mongo_client = MongoClient(mongodb_uri)
        db_name = os.getenv('MONGODB_DATABASE')
        if not db_name:
            raise ValueError("MONGODB_DATABASE environment variable is required")
        self.db = self.mongo_client[db_name]
        
        # Collections
        self.pdf_chunks_collection = self.db['pdf_chunks']
        self.pdf_hashes_collection = self.db['pdf_hashes']
        
        # PDF folder path
        self.pdf_folder = Path("pdf")
        
        logger.info("PDF Embedding Processor initialized successfully with LangChain OpenAI embeddings")
    
    def create_pdf_folder(self):
        """
        Create the pdf folder if it doesn't exist
        """
        self.pdf_folder.mkdir(exist_ok=True)
        logger.info(f"PDF folder ensured at: {self.pdf_folder.absolute()}")
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate MD5 hash of a file to detect changes
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def is_pdf_processed(self, pdf_path: Path) -> bool:
        """
        Check if PDF has been processed before by comparing file hash
        """
        current_hash = self.calculate_file_hash(pdf_path)
        existing_record = self.pdf_hashes_collection.find_one({
            "filename": pdf_path.name,
            "file_hash": current_hash
        })
        return existing_record is not None
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[dict]:
        """
        Extract text from PDF and split into chunks using LangChain text splitter
        """
        logger.info(f"Extracting text from: {pdf_path.name}")
        doc = fitz.open(pdf_path)
        
        # Extract all text from PDF
        full_text = ""
        page_texts = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                page_texts.append({
                    "text": text.strip(),
                    "page_number": page_num + 1
                })
                full_text += f"\n\n--- Page {page_num + 1} ---\n\n" + text.strip()
        
        doc.close()
        
        # Use LangChain text splitter for better chunking
        if full_text.strip():
            text_chunks = self.text_splitter.split_text(full_text)
            
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": i + 1,
                    "source": pdf_path.name
                })
            
            logger.info(f"Extracted and split {len(chunks)} chunks from {pdf_path.name}")
            return chunks
        else:
            logger.warning(f"No text content found in {pdf_path.name}")
            return []
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using LangChain OpenAI embeddings
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} text chunks")
            # Use LangChain's embed_documents method which handles batching efficiently
            embeddings = await asyncio.to_thread(self.embeddings.embed_documents, texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def process_pdf(self, pdf_path: Path):
        """
        Process a single PDF: extract text, generate embeddings, store in MongoDB
        """
        logger.info(f"Processing PDF: {pdf_path.name}")
        text_chunks = self.extract_text_from_pdf(pdf_path)
        
        if not text_chunks:
            logger.warning(f"No text chunks extracted from {pdf_path.name}")
            return

        # Extract texts for embedding generation
        texts = [chunk['text'] for chunk in text_chunks]
        
        # Generate embeddings for all chunks at once
        embeddings = await self.generate_embeddings(texts)
        
        # Prepare documents for insertion
        documents_to_insert = []
        for chunk, embedding in zip(text_chunks, embeddings):
            document = {
                "text": chunk['text'],
                "embedding": embedding,
                "metadata": {
                    "source": chunk['source'],
                    "chunk_id": chunk['chunk_id'],
                    "chunk_size": len(chunk['text'])
                }
            }
            documents_to_insert.append(document)
        
        if documents_to_insert:
            result = self.pdf_chunks_collection.insert_many(documents_to_insert)
            logger.info(f"Inserted {len(result.inserted_ids)} chunks into pdf_chunks collection")

            file_hash = self.calculate_file_hash(pdf_path)
            hash_record = {
                "filename": pdf_path.name,
                "file_hash": file_hash,
                "processed_at": datetime.utcnow(),
                "chunks_count": len(documents_to_insert)
            }
            self.pdf_hashes_collection.insert_one(hash_record)
            logger.info(f"Stored hash record for {pdf_path.name}")
    
    async def process_all_pdfs(self):
        """
        Check pdf folder and process any new or modified PDFs
        """
        self.create_pdf_folder()
        pdf_files = list(self.pdf_folder.glob("*.pdf"))

        if not pdf_files:
            logger.info("No PDF files found in pdf folder")
            return

        logger.info(f"Found {len(pdf_files)} PDF files")
        for pdf_path in pdf_files:
            try:
                if self.is_pdf_processed(pdf_path):
                    logger.info(f"PDF {pdf_path.name} already processed, skipping...")
                    continue
                
                await self.process_pdf(pdf_path)
                logger.info(f"Successfully processed {pdf_path.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
    
    def setup_chatbot(self):
        """
        Set up the conversational retrieval chain for chatbot functionality
        """
        try:
            # Check if there are any documents in the collection
            doc_count = self.pdf_chunks_collection.count_documents({})
            if doc_count == 0:
                print("⚠️ No documents found in database. Please add some PDFs first.")
                return None
                
            print(f"📚 Found {doc_count} document chunks in database")
            
            # Set up MongoDB Atlas Vector Search
            vector_store = MongoDBAtlasVectorSearch.from_connection_string(
                connection_string=os.getenv("MONGODB_URI"),
                embedding=self.embeddings,
                namespace=f"{os.getenv('MONGODB_DATABASE')}.pdf_chunks",
                text_key="text",
                embedding_key="embedding",
                relevance_score_fn="cosine"
            )
            
            # Create retriever with more lenient search parameters
            retriever = vector_store.as_retriever(
                search_type="similarity",  # Remove score threshold initially
                search_kwargs={
                    "k": 5,  # Number of documents to retrieve
                }
            )
            
            # Alternative: Use similarity_score_threshold with lower threshold
            # retriever = vector_store.as_retriever(
            #     search_type="similarity_score_threshold",
            #     search_kwargs={
            #         "k": 5,
            #         "score_threshold": 0.3  # Much lower threshold
            #     }
            # )
            
            # Set up ChatOpenAI LLM
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
            system_prompt = """
            You are a knowledgeable and professional virtual assistant for iQore, a deep-tech company pioneering quantum-classical hybrid compute infrastructure.
            iQore's core innovation lies in its software-native, platform-agnostic execution layers—iQD (quantum emulator) and iCD (classical compute distribution)—designed to accelerate performance and scalability of enterprise AI and simulation workloads.

            You have access to a curated set of official iQore documents and whitepapers, which you use to answer questions accurately and in detail. When responding, reference the information from these documents when relevant, but do not fabricate answers if the information is not available.

            Your tone is helpful, confident, and persuasive. You offer technical and business insights, and you’re able to support a range of user types—from curious visitors to experienced engineers and decision-makers.

            When appropriate, encourage users to:
            - Request a product demo
            - Schedule a follow-up meeting
            - Learn more about specific use cases
            - Ask deeper questions about the architecture

            Your goal is to inform, engage, and guide potential customers by showcasing the value of iQore’s solutions, while being honest if something is outside your knowledge.
            """


            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{question}")
            ])

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": prompt_template},
                verbose=False
            )
            
            logger.info("Chatbot setup completed successfully")
            print("✅ Chatbot ready! The retriever will now return documents without score filtering.")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error setting up chatbot: {e}")
            print(f"❌ Failed to set up chatbot: {e}")
            return None
    
    def close_connections(self):
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()
            logger.info("MongoDB connection closed")


async def run_chatbot(qa_chain):
    """
    Run the interactive chatbot CLI loop
    """
    print("\n" + "="*60)
    print("💬 iQore AI Assistant - Ask questions about your documents!")
    print("💡 Tips:")
    print("   - Ask specific questions about the content")
    print("   - Type 'sources' to see document sources in responses")
    print("   - Type 'exit' or 'quit' to end the conversation")
    print("="*60)
    
    chat_history = []
    show_sources = False
    
    while True:
        try:
            # Get user input
            query = input("\n🤔 You: ").strip()
            
            if not query:
                continue
                
            # Handle special commands
            if query.lower() in ("exit", "quit", "bye"):
                print("\n👋 Thanks for using iQore AI Assistant! Goodbye!")
                break
            elif query.lower() == "sources":
                show_sources = not show_sources
                status = "enabled" if show_sources else "disabled"
                print(f"📄 Source documents display {status}")
                continue
            elif query.lower() == "clear":
                chat_history = []
                print("🧹 Chat history cleared!")
                continue
            elif query.lower() == "help":
                print("\n📖 Available commands:")
                print("   - 'sources': Toggle source document display")
                print("   - 'clear': Clear chat history")
                print("   - 'help': Show this help message")
                print("   - 'exit'/'quit'/'bye': End conversation")
                continue
            
            # Process the query
            print("🤖 iQore Assistant: ", end="", flush=True)
            
            result = qa_chain({"question": query, "chat_history": chat_history})
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            print(answer)
            
            # Show source documents if enabled
            if show_sources and source_docs:
                print(f"\n📚 Sources ({len(source_docs)} documents):")
                for i, doc in enumerate(source_docs[:3], 1):  # Show top 3 sources
                    source = doc.metadata.get("source", "Unknown")
                    chunk_id = doc.metadata.get("chunk_id", "N/A")
                    content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    print(f"   {i}. {source} (Chunk {chunk_id})")
                    print(f"      Preview: {content_preview}")
            
            # Update chat history
            chat_history.append((query, answer))
            
            # Limit chat history to last 10 exchanges to manage context length
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
                
        except KeyboardInterrupt:
            print("\n\n👋 Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing your question: {e}")
            logger.error(f"Chatbot error: {e}")


async def main():
    """
    Main CLI function for PDF embedding and chat
    """
    print("🤖 iQore PDF Embedding & AI Assistant CLI (Updated with LangChain)")
    print("=" * 70)

    processor = None
    try:
        processor = PDFEmbeddingProcessor()
        
        # Process PDFs first
        await processor.process_all_pdfs()
        print("\n✅ PDF processing completed!")
        print("📄 Your PDFs have been embedded and stored in MongoDB Atlas")

        # Set up and run chatbot
        print("\n🔧 Setting up AI chatbot...")
        qa_chain = processor.setup_chatbot()
        
        if qa_chain:
            await run_chatbot(qa_chain)
        else:
            print("❌ Could not start chatbot. Please check your configuration and try again.")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")
    finally:
        if processor:
            processor.close_connections()

if __name__ == "__main__":
    asyncio.run(main())
