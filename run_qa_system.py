import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.document_processor import DocumentProcessor
from src.core.embedding_service import EmbeddingService
from src.data.vector_store import VectorStoreManager
from src.core.qa_engine import QAEngine
from config.settings import settings

class InteractiveQASystem:
    """Interactive testing interface for the Document QA System."""

    def __init__(self):
        print("Initializing Interactive QA System...")
        self.processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreManager(self.embedding_service)
        self.qa_engine = QAEngine()

        self.retriever = None
        self.is_initialized = False


    def setup_documents(self, documents_path=".data/uploads"):
        """Setup documents from the specified directory."""
        documents_dir = Path(documents_path)
        if not documents_dir.exists() or not documents_dir.is_dir():
            documents_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created documents directory at {documents_path}. Please add documents and restart.")
            # self._create_sample_document(documents_path)

            print("Please add documents to the directory and restart the application.")

        try:
            documents = self.processor.process_directory(str(documents_dir))
            if not documents:
                print("❌ No documents found."
                      "Please add PDF, TXT, or DOCX files to the documents directory.")
                return False

            print("Creating Vector Store...")
            self.vector_store.create_vector_store(documents)
            self.retriever = self.vector_store.get_retriever()
            self.is_initialized = True

            print("✅ Document setup complete.")

            doc_count = self.vector_store.get_document_count()
            print(f"Total documents in vector store: {doc_count}")
            return True
        except Exception as e:
            print(f"❌ Error setting up documents: {e}")
            return False

    # def _create_sample_document(self, directory_path):
        """Create a sample document in the specified directory."""

    def ask_question(self, question: str):
        """Ask a question to the QA system."""
        if not self.is_initialized:
            print("❌ System not initialized. Please set up documents first.")
            return
        try:
            response = self.qa_engine.query(question, self.retriever)

            print("\n=== Answer ===")
            print(response["answer"])
            print("\n=== Sources ===")

            if response['sources']:
                for source in response["sources"]:
                    print(f"- Source: {source['source']}, Preview: {source['content_preview']}")

            if 'error' in response:
                print(f"\n❌ Error: {response['error']}")
        except Exception as e:
            print(f"❌ Error processing question: {e}")

    def show_system_info(self):
        """Display system information"""
        if not self.is_initialized:
            print("❌ System not initialized.")
            return

        doc_count = self.vector_store.get_document_count()

        print("\n📊 SYSTEM INFORMATION:")
        print(f"   - Embedding Model: {settings.embedding_model}")
        print(f"   - Chat Model: {settings.chat_model}")
        print(f"   - Documents in Vector Store: {doc_count}")
        print(f"   - Chunk Size: {settings.chunk_size}")
        print(f"   - Chunk Overlap: {settings.chunk_overlap}")

    def interactive_mode(self):
            """Start interactive question-answering mode"""
            if not self.is_initialized:
                print("❌ Please set up documents first using option 1.")
                return

            print("\n" + "=" * 60)
            print("🤖 DOCUMENT Q&A SYSTEM - INTERACTIVE MODE")
            print("=" * 60)
            print("Type your questions about the documents (type 'back' to return to menu)")
            print("Type 'debug' to see retrieval details")
            print("-" * 60)

            debug_mode = False

            while True:
                try:
                    question = input("\n💬 Your question: ").strip()

                    if question.lower() == 'back':
                        break
                    elif question.lower() == 'debug':
                        debug_mode = not debug_mode
                        status = "ON" if debug_mode else "OFF"
                        print(f"🔧 Debug mode {status}")
                        continue
                    elif question.lower() in ['', 'exit', 'quit']:
                        print("👋 Goodbye!")
                        return
                    elif question.lower() == 'info':
                        self.show_system_info()
                        continue

                    if debug_mode:
                        print("🔍 Retrieving relevant documents...")
                        docs = self.retriever.get_relevant_documents(question)
                        print(f"📄 Found {len(docs)} relevant documents:")
                        for i, doc in enumerate(docs, 1):
                            source = doc.metadata.get('source', 'Unknown')
                            content_preview = doc.page_content[:100] + "..." if len(
                                doc.page_content) > 100 else doc.page_content
                            print(f"   {i}. {source}")
                            print(f"      {content_preview}")
                        print("-" * 40)

                    self.ask_question(question)

                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")

def main():
    """Main function to run the interactive test"""
    print("🎯 DOCUMENT Q&A SYSTEM TEST")
    print("   Built with LangChain + Hugging Face + Google Gemini")
    print()

    # Initialize the system
    qa_system = InteractiveQASystem()

    while True:
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("=" * 50)
        print("1. 📁 Setup Documents & Initialize System")
        print("2. ❓ Ask a Single Question")
        print("3. 💬 Interactive Q&A Mode")
        print("4. 📊 System Information")
        print("5. 🚪 Exit")
        print("-" * 50)

        choice = input("Select an option (1-5): ").strip()

        if choice == '1':
            # Setup documents
            documents_path = input("Enter documents directory path [./data/uploads]: ").strip()
            if not documents_path:
                documents_path = "./data/uploads"

            success = qa_system.setup_documents(documents_path)
            if success:
                print("✅ System initialized and ready for questions!")
            else:
                print("❌ Failed to initialize system. Check the documents directory.")

        elif choice == '2':
            # Single question
            if not qa_system.is_initialized:
                print("❌ Please set up documents first (option 1).")
                continue

            question = input("Enter your question: ").strip()
            if question:
                qa_system.ask_question(question)
            else:
                print("❌ Please enter a valid question.")

        elif choice == '3':
            # Interactive mode
            qa_system.interactive_mode()

        elif choice == '4':
            # System info
            qa_system.show_system_info()

        elif choice == '5':
            print("👋 Thank you for using the Document Q&A System!")
            break

        else:
            print("❌ Invalid option. Please choose 1-5.")

if __name__ == "__main__":
    # Check if .env file exists
    if not Path(".env").exists():
        print("❌ .env file not found!")
        print("Please create a .env file with your Google API key:")
        print("GOOGLE_API_KEY=your_google_ai_key_here")
        print("\nYou can get your API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)

    # Check if Google API key is set
    if not settings.google_api_key or settings.google_api_key == "your_google_ai_key_here":
        print("❌ Please update your .env file with a valid Google API key")
        sys.exit(1)

    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Session ended by user.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check your configuration and try again.")


