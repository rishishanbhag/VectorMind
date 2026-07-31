import os
import pickle
import logging
from typing import Optional, List
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_anthropic import ChatAnthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PDFChatbot:
    def __init__(self):
        """Initialize the PDF chatbot with configuration"""
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        self.claude_model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
        self.conversation_chain = None
        self.vectorstore = None
        self.vectorstore_file = 'vectorstore.pkl'
        
    def get_pdf_text(self, pdf_docs) -> str:
        """Extract text from PDF documents"""
        text = ""
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            except Exception as e:
                logger.error(f"Error reading PDF: {e}")
        return text
    
    def get_pdf_text_from_paths(self, pdf_paths: List[str]) -> str:
        """Extract text from PDF files given their paths"""
        text = ""
        for pdf_path in pdf_paths:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
            except Exception as e:
                logger.error(f"Error reading PDF {pdf_path}: {e}")
        return text
    
    def get_text_chunks(self, raw_text: str) -> List[str]:
        """Split text into chunks for processing"""
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_text(raw_text)
    
    def create_vectorstore(self, text_chunks: List[str]) -> FAISS:
        """Create FAISS vectorstore from text chunks"""
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
        self.vectorstore = vectorstore
        return vectorstore
    
    def save_vectorstore(self, vectorstore: Optional[FAISS] = None):
        """Save vectorstore to file"""
        try:
            store_to_save = vectorstore or self.vectorstore
            if store_to_save is None:
                raise ValueError("No vectorstore to save")
            
            with open(self.vectorstore_file, 'wb') as f:
                pickle.dump(store_to_save, f)
            logger.info("Vectorstore saved successfully")
        except Exception as e:
            logger.error(f"Failed to save vectorstore: {e}")
            raise
    
    def load_vectorstore(self) -> Optional[FAISS]:
        """Load vectorstore from file"""
        try:
            with open(self.vectorstore_file, 'rb') as f:
                self.vectorstore = pickle.load(f)
            logger.info("Vectorstore loaded successfully")
            return self.vectorstore
        except FileNotFoundError:
            logger.warning("No vectorstore found. Please process documents first.")
            return None
        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}")
            return None
    
    def create_conversation_chain(self, vectorstore: Optional[FAISS] = None) -> ConversationalRetrievalChain:
        """Create conversation chain with Claude"""
        store_to_use = vectorstore or self.vectorstore
        if store_to_use is None:
            raise ValueError("No vectorstore available for conversation chain")
        
        llm = ChatAnthropic(
            model=self.claude_model,
            temperature=0.2,
            anthropic_api_key=self.anthropic_api_key,
        )
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=store_to_use.as_retriever(),
            memory=memory
        )
        
        self.conversation_chain = conversation_chain
        return conversation_chain
    
    def process_documents(self, pdf_docs) -> dict:
        """Process PDF documents and create vectorstore"""
        try:
            # Extract text
            raw_text = self.get_pdf_text(pdf_docs)
            if not raw_text.strip():
                raise ValueError("No text could be extracted from the PDFs")
            
            # Create chunks
            text_chunks = self.get_text_chunks(raw_text)
            
            # Create vectorstore
            vectorstore = self.create_vectorstore(text_chunks)
            
            # Save vectorstore
            self.save_vectorstore(vectorstore)
            
            # Create conversation chain
            self.create_conversation_chain(vectorstore)
            
            logger.info(f"Successfully processed documents: {len(text_chunks)} chunks created")
            
            return {
                "status": "success",
                "chunks_created": len(text_chunks),
                "text_length": len(raw_text)
            }
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise
    
    def process_documents_from_paths(self, pdf_paths: List[str]) -> dict:
        """Process PDF documents from file paths"""
        try:
            # Extract text
            raw_text = self.get_pdf_text_from_paths(pdf_paths)
            if not raw_text.strip():
                raise ValueError("No text could be extracted from the PDFs")
            
            # Create chunks
            text_chunks = self.get_text_chunks(raw_text)
            
            # Create vectorstore
            vectorstore = self.create_vectorstore(text_chunks)
            
            # Save vectorstore
            self.save_vectorstore(vectorstore)
            
            # Create conversation chain
            self.create_conversation_chain(vectorstore)
            
            logger.info(f"Successfully processed {len(pdf_paths)} documents: {len(text_chunks)} chunks created")
            
            return {
                "status": "success",
                "files_processed": len(pdf_paths),
                "chunks_created": len(text_chunks),
                "text_length": len(raw_text)
            }
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise
    
    def initialize_from_saved_vectorstore(self):
        """Initialize conversation chain from saved vectorstore"""
        vectorstore = self.load_vectorstore()
        if vectorstore:
            self.create_conversation_chain(vectorstore)
            return True
        return False
    
    def ask_question(self, question: str, conversation_id: Optional[str] = None) -> dict:
        """Ask a question and get response"""
        try:
            if not question.strip():
                raise ValueError("Question cannot be empty")
            
            if self.conversation_chain is None:
                # Try to initialize from saved vectorstore
                if not self.initialize_from_saved_vectorstore():
                    raise ValueError("No conversation chain available. Please process documents first.")
            
            logger.info(f"Processing question: '{question}' for conversation {conversation_id}")
            
            response = self.conversation_chain({"question": question})
            answer = response['answer']
            
            logger.info(f"Generated answer: {answer[:100]}...")
            
            return {
                "answer": answer,
                "status": "success",
                "conversation_id": conversation_id,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise
    
    def get_status(self) -> dict:
        """Get current status of the chatbot"""
        return {
            "vectorstore_loaded": self.vectorstore is not None,
            "conversation_chain_initialized": self.conversation_chain is not None,
            "vectorstore_file_exists": os.path.exists(self.vectorstore_file),
            "anthropic_api_configured": bool(self.anthropic_api_key)
        }

# Global instance for easy access
chatbot = PDFChatbot()