"""
Prompt Template Module

This module manages prompt templates and formatting for the LLM.
Prompts guide the model's behavior and provide context.

Prompt Engineering:
- System prompt: Sets AI personality and instructions
- User prompt: The actual user query
- Context prompt: Retrieved relevant documents
- RAG prompt: Combined template for generation

Variables in templates:
{context} - Retrieved document context
{query} - User question
{model} - Model name
"""

from typing import Optional, Dict, Any, List
from enum import Enum

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class PromptTemplate:
    """Base prompt template"""
    
    def __init__(self, template: str, description: str = ""):
        """
        Initialize prompt template.
        
        Args:
            template: Template string with {variable} placeholders
            description: Description of the template
        """
        self.template = template
        self.description = description
    
    def format(self, **kwargs) -> str:
        """Format template with provided variables"""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing variable in template: {e}")
            raise
    
    def get_variables(self) -> List[str]:
        """Extract variable names from template"""
        import re
        return re.findall(r'\{(\w+)\}', self.template)


class PromptTemplates:
    """Collection of prompt templates for different scenarios"""
    
    # System prompt - sets AI behavior
    SYSTEM_PROMPT = PromptTemplate(
        template="""You are an intelligent and helpful AI assistant specialized in answering questions based on provided documents.

Instructions:
1. Answer questions accurately and concisely based on the provided context
2. If the answer is not found in the context, clearly state that
3. Always cite the source document when providing information
4. Be honest about the limitations of your knowledge
5. If you're unsure, ask for clarification or more context
6. Format your response in a clear, organized manner

Context-Aware Behavior:
- Prioritize information from the provided context
- Cross-reference multiple documents if relevant
- Flag any contradictions in sources
- Provide relevant details and examples when available""",
        description="Base system prompt for all conversations"
    )
    
    # RAG prompt - combines context and query
    RAG_PROMPT = PromptTemplate(
        template="""Based on the following context from documents, answer the user's question.

CONTEXT:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer based primarily on the provided context
2. If information is not in the context, indicate this clearly
3. Cite specific documents or sections when using context
4. Provide accurate, helpful, and concise responses
5. If the question is ambiguous, ask for clarification

ANSWER:""",
        description="Template for RAG-based question answering"
    )
    
    # Context formatting
    CONTEXT_TEMPLATE = PromptTemplate(
        template="""Document: {source}
{text}

---""",
        description="Template for formatting individual context chunks"
    )
    
    # Summary prompt
    SUMMARY_PROMPT = PromptTemplate(
        template="""Summarize the following text concisely, keeping the key points:

{text}

SUMMARY:""",
        description="Template for summarizing documents"
    )
    
    # Query expansion (for better retrieval)
    QUERY_EXPANSION_PROMPT = PromptTemplate(
        template="""Given the user query: "{query}"

Generate 3 alternative phrasings or related questions that could help retrieve relevant documents:

1. 
2. 
3.""",
        description="Template for expanding queries to improve retrieval"
    )


class PromptBuilder:
    """Builder for constructing prompts from templates and context"""
    
    def __init__(self):
        """Initialize prompt builder"""
        self.logger = get_logger(__name__)
    
    def build_rag_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        include_sources: bool = True
    ) -> str:
        """
        Build complete RAG prompt with context.
        
        Args:
            query: User query
            context_chunks: List of context chunks with 'text' and 'source'
            include_sources: Whether to include source citations
        
        Returns:
            Formatted prompt ready for LLM
        """
        try:
            # Format context
            context_parts = []
            for i, chunk in enumerate(context_chunks, 1):
                text = chunk.get("text", "")
                source = chunk.get("metadata", {}).get("source", "Unknown") if include_sources else None
                similarity = chunk.get("similarity")
                
                context_text = f"[{i}] {text}"
                if source:
                    context_text += f"\n(Source: {source}"
                    if similarity:
                        context_text += f", Relevance: {similarity:.1%}"
                    context_text += ")"
                
                context_parts.append(context_text)
            
            formatted_context = "\n\n".join(context_parts)
            
            # Build prompt
            prompt = PromptTemplates.RAG_PROMPT.format(
                context=formatted_context,
                query=query
            )
            
            self.logger.debug(f"Built RAG prompt with {len(context_chunks)} context chunks")
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error building RAG prompt: {e}")
            raise
    
    def build_simple_prompt(
        self,
        query: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build simple prompt without context.
        
        Args:
            query: User query
            system_prompt: Optional custom system prompt
        
        Returns:
            Formatted prompt
        """
        if system_prompt:
            return f"{system_prompt}\n\nQuestion: {query}\n\nAnswer:"
        else:
            return f"{PromptTemplates.SYSTEM_PROMPT.template}\n\nQuestion: {query}\n\nAnswer:"
    
    def build_context_prompt(
        self,
        text: str,
        source: str
    ) -> str:
        """
        Format a single context chunk.
        
        Args:
            text: Chunk text
            source: Document source
        
        Returns:
            Formatted context
        """
        return PromptTemplates.CONTEXT_TEMPLATE.format(
            source=source,
            text=text
        )
    
    def build_summary_prompt(self, text: str) -> str:
        """Build summarization prompt"""
        return PromptTemplates.SUMMARY_PROMPT.format(text=text)
    
    def build_query_expansion_prompt(self, query: str) -> str:
        """Build query expansion prompt"""
        return PromptTemplates.QUERY_EXPANSION_PROMPT.format(query=query)


# Global prompt builder
_prompt_builder = None


def get_prompt_builder() -> PromptBuilder:
    """Get or create prompt builder instance"""
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder()
    return _prompt_builder


def build_rag_prompt(
    query: str,
    context_chunks: List[Dict[str, Any]]
) -> str:
    """Convenience function to build RAG prompt"""
    return get_prompt_builder().build_rag_prompt(query, context_chunks)


def build_simple_prompt(query: str) -> str:
    """Convenience function to build simple prompt"""
    return get_prompt_builder().build_simple_prompt(query)
