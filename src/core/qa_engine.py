import logging
from typing import Dict, Any, List
from xml.dom.minidom import Document

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings

logger = logging.getLogger(__name__)


class QAEngine:
    """A simple question-answering engine."""

    def __init__(self):
        self.llm = self._initialize_llm()
        self.chain = None
        logger.info("QAEngine initialized.")

    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize the language model with proper configuration"""
        logger.info(f"Initializing LLM: {settings.chat_model}")

        return ChatGoogleGenerativeAI(
            model=settings.chat_model,
            google_api_key=settings.google_api_key,
            temperature=settings.temperature,
            max_retries=3,
            timeout=30
        )

    def _create_prompt(self) -> ChatPromptTemplate:
        """Create a chat prompt template."""
        logger.info("Creating chat prompt template.")

        return ChatPromptTemplate.from_template("""
                    You are an expert AI assistant for document analysis. Use the provided context to answer the user's question accurately and helpfully.
                    
                    CONTEXT INFORMATION:
                    {context}
                    
                    USER QUESTION:
                    {question}
                    
                    INSTRUCTIONS:
                    1. Answer based ONLY on the provided context
                    2. If the context doesn't contain the answer, clearly state that you cannot answer based on the provided documents
                    3. Do not make up information or use external knowledge
                    4. If the question is ambiguous, ask for clarification
                    5. Provide citations to the source documents when possible
                    6. Keep your answer concise but comprehensive
                    
                    ANSWER:
                """)
        # return prompt

    def create_qa_chain(self, retriever):
        """Create the Q&A chain with proper document formatting"""
        logger.info("Calling create_qa_chain.")
        def format_documents(docs: List[Document]) -> str:
            """Format retrieved documents for the prompt"""
            if not docs:
                return "No relevant documents found."

            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content.strip()
                formatted.append(f"[Document {i} - Source: {source}]\n{content}")

            return "\n\n".join(formatted)
        # print(f"retriever: {retriever}, format_documents: {format_documents}, self.llm: {self.llm}")
        self.chain = (
                {
                    "context": retriever | format_documents,
                    "question": RunnablePassthrough()
                }
                | self._create_prompt()
                | self.llm
                | StrOutputParser()
        )
        # print("QA chain created:", self.chain)
        if self.chain:
            logger.info("QA chain created successfully.")
            return self.chain
        else:
            logger.error("Failed to create QA chain.")
            raise ValueError("QA chain creation failed.")

    async def aquery(self, question: str, retriever) -> Dict[str, Any]:
        """Asynchronously query the QA chain."""
        if not self.chain:
            self.create_qa_chain(retriever)

        try:
            relevant_docs = retriever.get_relevant_documents(question)
            answer = self.chain.invoke(question)

            response = {
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "source": doc.metadata.get('source', 'unknown source'),
                        "content_preview": doc.page_content[:200] + "...",  # first 200 chars
                        "relevance_score": getattr(doc, 'score', None)
                    } for doc in relevant_docs
                ],
                "document_count": len(relevant_docs)
            }
            logger.info("QA query processed successfully.")
            return response
        except Exception as e:
            logger.error("Error processing QA query: %s", str(e))
            return {
                "question": question,
                "answer": "Error in async query.",
                "sources": [],
                "document_count": 0
            }

    def query(self, question: str, retriever) -> Dict[str, Any]:
        """Synchronously query the QA chain."""
        import asyncio

        try:
            if not self.chain:
                logger.info("The chaine self.chain is not initialized. Creating QA chain.")
                self.create_qa_chain(retriever)
            else:
                logger.info("The chaine self.chain is already initialized.")
            return asyncio.run(self.aquery(question, retriever))

        except Exception as e:
            logger.error("Error in synchronous query: %s", str(e))
            return {
                "question": question,
                "answer": "An error occurred while processing your request.",
                "sources": [],
                "document_count": 0
            }
