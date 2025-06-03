from __future__ import annotations as _annotations

from dataclasses import dataclass
#from dotenv import load_dotenv
import logfire
import os

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

#load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are an AI assistant trained to analyze and explain the content of Adel Messaoudi's PhD thesis.

This thesis is structured in well-defined chapters:
- Chapter 1: Introduction
- Chapter 2: Radiative transfer models in a full space
- Chapter 3: Radiative transfer models in a half-space
- Chapter 4: Radiative transfer models in a slab
- Chapter 5: Radiative transfer models in a box
- Chapter 6: Conclusion and perspectives
- Appendix and Bibliography

Your mission is to:
- Clearly answer any question related to the thesis content
- Provide in-depth explanations of the methods, models, results, and reasoning
- Guide the user to the appropriate chapters and sections
- Synthesize and connect information across different parts when needed
- Be transparent if the information is not available in the document

You have access to the full document via a RAG system and can search or retrieve relevant chapters and sections using available tools.

Always use the tools provided to support your answer with actual excerpts from the thesis.

Be clear, accurate, and helpful, especially for users who are not experts in the subject.
"""

pydantic_ai_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@pydantic_ai_agent.tool
async def retrieve_relevant_thesis_content(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant thesis chunks based on the query with RAG.

    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query about the thesis

    Returns:
        A formatted string containing the most relevant thesis chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

        # Query Supabase for relevant document chunks
        result = ctx.deps.supabase.rpc(
            'match_document_chunks',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'pdf_document'}
            }
        ).execute()

        if not result.data:
            return "No relevant content found in the thesis."

        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)

        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving thesis content: {e}")
        return f"Error retrieving thesis content: {str(e)}"

@pydantic_ai_agent.tool
async def list_thesis_chapters(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available thesis chapters.

    Returns:
        List[str]: List of unique chapter names for all thesis chapters
    """
    try:
        # Query Supabase for unique chapters where source is pdf_document
        result = ctx.deps.supabase.from_('document_chunks') \
            .select('chapter') \
            .eq('metadata->>source', 'pdf_document') \
            .execute()

        if not result.data:
            return []

        # Extract unique chapters
        chapters = sorted(set(doc['chapter'] for doc in result.data))
        return chapters

    except Exception as e:
        print(f"Error retrieving thesis chapters: {e}")
        return []

@pydantic_ai_agent.tool
async def get_chapter_content(ctx: RunContext[PydanticAIDeps], chapter_name: str) -> str:
    """
    Retrieve the full content of a specific thesis chapter by combining all its chunks.

    Args:
        ctx: The context including the Supabase client
        chapter_name: The name of the chapter to retrieve (e.g., "Chapitre 1 : Introduction")

    Returns:
        str: The complete chapter content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this chapter, ordered by chunk_number
        result = ctx.deps.supabase.from_('document_chunks') \
            .select('title, content, chunk_number') \
            .eq('chapter', chapter_name) \
            .eq('metadata->>source', 'pdf_document') \
            .order('chunk_number') \
            .execute()

        if not result.data:
            return f"No content found for chapter: {chapter_name}"

        # Format the chapter with its title and all chunks
        chapter_title = chapter_name  # Use the full chapter name as title
        formatted_content = [f"# {chapter_title}\n"]

        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])

        # Join everything together
        return "\n\n".join(formatted_content)

    except Exception as e:
        print(f"Error retrieving chapter content: {e}")
        return f"Error retrieving chapter content: {str(e)}"
