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
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
            print("📁 No PDF files found in the 'pdf' folder")
            print("📄 Please add PDF files to the 'pdf' folder and run the script again")
            return

        logger.info(f"Found {len(pdf_files)} PDF files")
        print(f"📚 Found {len(pdf_files)} PDF files")
        
        processed_count = 0
        skipped_count = 0
        
        for pdf_path in pdf_files:
            try:
                if self.is_pdf_processed(pdf_path):
                    logger.info(f"PDF {pdf_path.name} already processed, skipping...")
                    print(f"⏭️  Skipping {pdf_path.name} (already processed)")
                    skipped_count += 1
                    continue
                
                print(f"🔄 Processing {pdf_path.name}...")
                await self.process_pdf(pdf_path)
                logger.info(f"Successfully processed {pdf_path.name}")
                print(f"✅ Successfully processed {pdf_path.name}")
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                print(f"❌ Error processing {pdf_path.name}: {e}")
        
        # Summary
        print(f"\n📊 Processing Summary:")
        print(f"   • Processed: {processed_count} files")
        print(f"   • Skipped: {skipped_count} files")
        print(f"   • Total files: {len(pdf_files)}")
        
        # Check total documents in database
        total_chunks = self.pdf_chunks_collection.count_documents({})
        print(f"   • Total chunks in database: {total_chunks}")
    
    def close_connections(self):
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()
            logger.info("MongoDB connection closed")


async def main():
    """
    Main function for PDF embedding processing
    """
    print("📄 iQore PDF Embedding Processor")
    print("=" * 50)
    print("🔧 This script will:")
    print("   • Scan the 'pdf' folder for PDF files")
    print("   • Extract text and create embeddings") 
    print("   • Store embeddings in MongoDB Atlas")
    print("   • Skip files that have already been processed")
    print("=" * 50)

    processor = None
    try:
        processor = PDFEmbeddingProcessor()
        
        # Process all PDFs
        await processor.process_all_pdfs()
        
        print("\n✅ PDF embedding processing completed!")
        print("📄 Your PDFs have been embedded and stored in MongoDB Atlas")
        print("🤖 The chatbot backend can now use these embeddings for question answering")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")
    finally:
        if processor:
            processor.close_connections()

if __name__ == "__main__":
    asyncio.run(main())
