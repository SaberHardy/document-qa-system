import logging
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

    def _initialize_llm(self):
        """Initialize the language model."""
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

        prompt = ChatPromptTemplate.from_messages("""
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
        return prompt

    def create_qa_chain(self, retriever):
        """Create a QA chain using the vector store and LLM."""
        from langchain.chains import RetrievalQA
        from typing import List

        def format_docs(docs: List[Document]) -> str:
            if not docs:
                return "No context available."

            formatter = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "unknown source")
                content = doc.page_content.strip()
                formatter.append(f"[Document {i} - Source: {source}]\n{content}\n")

            return "\n\n".join(formatter)

        self.chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.llm.bind_chat_prompt(self._create_prompt())
            | self.llm
            | StrOutputParser()
        )
        logger.info("QA chain created.")
        return self.chain
