import os
from openai import OpenAI
import instructor
from model.schema import FinalRankedResults, Answer, Citation
from dotenv import load_dotenv

load_dotenv()

# Initialize Instructor client with OpenAI Responses API
client = instructor.from_provider(
    "openai/gpt-5-mini",
    mode=instructor.Mode.RESPONSES_TOOLS
)


def generate_answer(query: str, final_results: FinalRankedResults) -> Answer:
    """
    Generate a comprehensive answer using retrieved chunks.
    
    Args:
        query: The original user query
        final_results: Top ranked chunks after RRF
        
    Returns:
        Answer object with answer text, citations, and confidence
    """
    print(f"\nGenerating answer for query: {query}")
    print(f"Using {len(final_results.chunks)} chunks")
    
    # Prepare context from chunks
    context_parts = []
    for i, chunk in enumerate(final_results.chunks, 1):
        context_parts.append(
            f"[Chunk {chunk.chunk_id}] (Pages {chunk.page_start}-{chunk.page_end})\n{chunk.text}\n"
        )
    
    context = "\n".join(context_parts)
    
    # Create prompt
    prompt = f"""You are an expert insurance policy assistant. Answer the user's query based ONLY on the provided context from the insurance policy document.

CONTEXT:
{context}

QUERY: {query}

INSTRUCTIONS:
1. Provide a clear, brief, informative answer based on the context unless the question asks to explain in detail.
2. Use specific details from the chunks.
3. Add inline citations immediately after the claim they support using the format [Chunk <chunk_id>, p.<page_start>-<page_end>].
4. If the context doesn't fully answer the query, acknowledge the limitations. If the question is unrelated to the policy or you do not know the answer from the context, say so plainly and DO NOT include any citations.
5. Set confidence level:
   - "high": Query is fully answered with clear information
   - "medium": Query is partially answered or information is somewhat unclear
   - "low": Context doesn't adequately address the query
6. If the question is not related to the context, acknowledge that you are an insurance policy assistant and you can only answer questions related to the insurance policy document and politely decline to answer or say you don't know. In that case, do not provide citations.
Return your answer in the structured format with citations."""

    # Generate structured answer using instructor
    answer = client.responses.create(
        input=prompt,
        response_model=Answer
    )
    
    print(f"Answer generated with {len(answer.citations)} citations")
    print(f"Confidence: {answer.confidence}")
    
    return answer



