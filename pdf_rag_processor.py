import os
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

from mistralai import Mistral
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize clients
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    chapter: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def extract_pdf_with_mistral_ocr(pdf_path: str) -> str:
    """Extract text from PDF using Mistral OCR."""
    try:
        # Upload file to Mistral
        print(f"Uploading PDF to Mistral: {pdf_path}")
        uploaded_pdf = mistral_client.files.upload(
            file={
                "file_name": Path(pdf_path).name,
                "content": open(pdf_path, "rb"),
            },
            purpose="ocr"
        )

        # Get signed URL
        signed_url = mistral_client.files.get_signed_url(file_id=uploaded_pdf.id)

        # Process with OCR
        print("Processing PDF with Mistral OCR...")
        ocr_response = mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            include_image_base64=True
        )

        # Extract markdown content from all pages
        markdown_content = ""
        for page in ocr_response.pages:
            raw_markdown = page.markdown
            if '\x00' in raw_markdown or '\u0000' in raw_markdown:
                print(f"[Avertissement] Caractère \\x00 détecté dans la page {page.page_number}")
            markdown = clean_text(raw_markdown)
            markdown_content += markdown + "\n\n"

        print(f"Successfully extracted {len(markdown_content)} characters from {len(ocr_response.pages)} pages")

        # Clean up uploaded file
        try:
            mistral_client.files.delete(file_id=uploaded_pdf.id)
            print("Cleaned up uploaded file")
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up uploaded file: {cleanup_error}")

        return markdown_content

    except Exception as e:
        print(f"Error processing PDF with Mistral OCR: {e}")
        raise Exception(f"Failed to process PDF with Mistral OCR: {e}")

def clean_text(text: str) -> str:
    """Remove null bytes and other problematic characters for PostgreSQL."""
    # Remove null bytes that cause PostgreSQL errors
    text = text.replace('\u0000', '')
    text = text.replace('\x00', '')
    return text

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    # Clean the text first to remove problematic characters
    text = clean_text(text)

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, pdf_path: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"PDF: {Path(pdf_path).name}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
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

async def process_chunk(chunk: str, chunk_number: int, pdf_path: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, pdf_path)

    # Get embedding
    embedding = await get_embedding(chunk)

    # Create metadata
    metadata = {
        "source": "pdf_document",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "pdf_name": Path(pdf_path).name,
        "chapter": get_chapter_title(pdf_path)  # Get full chapter title
    }

    return ProcessedChunk(
        chapter=get_chapter_title(pdf_path),
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        # Clean all text fields before insertion
        data = {
            "chapter": chunk.chapter,
            "chunk_number": chunk.chunk_number,
            "title": clean_text(chunk.title),
            "summary": clean_text(chunk.summary),
            "content": clean_text(chunk.content),
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }

        result = supabase.table("document_chunks").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.chapter}")
        return result
    except Exception as e:
        print(f"Error inserting chunk {chunk.chunk_number} for {chunk.chapter}: {e}")
        return None

async def process_and_store_document(pdf_path: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)

    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, pdf_path)
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def process_pdf(pdf_path: str):
    """Extract text from PDF and process it as a complete document."""
    # Extract text using Mistral OCR
    markdown_content = extract_pdf_with_mistral_ocr(pdf_path)

    if not markdown_content:
        print(f"Failed to extract content from PDF: {Path(pdf_path).name}")
        return

    print(f"Successfully extracted content from {Path(pdf_path).name}")
    print(f"Processing complete document ({len(markdown_content)} characters)")

    # Process the entire document as one unit
    await process_and_store_document(pdf_path, markdown_content)

async def process_pdfs_parallel(pdf_paths: List[str], max_concurrent: int = 3):
    """Process multiple PDFs in parallel with a concurrency limit."""
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_pdf(pdf_path: str):
        async with semaphore:
            await process_pdf(pdf_path)

    # Process all PDFs in parallel with limited concurrency
    await asyncio.gather(*[process_single_pdf(pdf_path) for pdf_path in pdf_paths])

def get_chapter_title(pdf_filename: str) -> str:
    """Get the full chapter title from PDF filename."""
    chapter_titles = {
        "Chapter_1.pdf": "Chapitre 1 : Introduction",
        "Chapter_2.pdf": "Chapter 2 : Radiative transfer models in a full space",
        "Chapter_3.pdf": "Chapter 3 : Radiative transfer models in a half-space",
        "Chapter_4.pdf": "Chapter 4 : Radiative transfer models in a slab",
        "Chapter_5.pdf": "Chapter 5 : Radiative transfer models in a box",
        "Chapter_6.pdf": "Chapter 6 : Conclusion and perspectives",
        "Appendix_and_Bibliography.pdf": "Appendix and Bibliography"
    }

    filename = Path(pdf_filename).name
    return chapter_titles.get(filename, filename)  # Return filename if not found

def get_chapter_pdfs() -> List[str]:
    """Get all chapter PDF files from the current directory."""
    current_dir = Path(".")
    pdf_files = []

    # Look for chapter PDFs and other thesis-related PDFs
    for pdf_file in current_dir.glob("*.pdf"):
        pdf_files.append(str(pdf_file))

    # Sort to ensure consistent processing order
    pdf_files.sort()
    return pdf_files

async def main():
    """Main function to process multiple PDF chapters."""
    # Get all PDF files
    pdf_paths = get_chapter_pdfs()

    if not pdf_paths:
        print("No PDF files found in the current directory")
        return

    # Validate PDF paths
    valid_pdfs = []
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file not found at {pdf_path}")
            continue
        if not pdf_path.lower().endswith('.pdf'):
            print(f"Warning: Skipping non-PDF file: {pdf_path}")
            continue
        valid_pdfs.append(pdf_path)

    if not valid_pdfs:
        print("No valid PDF files found")
        return

    print(f"Found {len(valid_pdfs)} PDF files to process:")
    for pdf_path in valid_pdfs:
        print(f"  - {Path(pdf_path).name}")

    # Process all PDFs in parallel
    await process_pdfs_parallel(valid_pdfs)

if __name__ == "__main__":
    asyncio.run(main())