from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from ingestion_pipeline import EmbeddingFunction, load_documents, split_documents
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

load_dotenv()

# Setup
persistent_directory = "db/chroma_db"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function = EmbeddingFunction(embedding_model),
    collection_metadata={"hnsw:space": "cosine"}
)

# Load and chunk documents via the ingestion pipeline
raw_documents = load_documents("docs")
chunks = split_documents(raw_documents, chunk_size=1000, chunk_overlap=0)

# Pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]

# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

# Original query
original_query = "Tell me about the discography of Twice?"
print("\n" + "="*60)
print(f"Original Query: {original_query}\n")


# ──────────────────────────────────────────────────────────────────
# Step 1: Generate Multiple Query Variations
# ──────────────────────────────────────────────────────────────────

llm_with_tools = llm.with_structured_output(
    QueryVariations,
    method="json_schema",
)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

Original query: {original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = llm_with_tools.invoke(prompt)
query_variations = response.queries

print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "="*60)


# ──────────────────────────────────────────────────────────────────
# Step 2: Hybrid Search with Each Query Variation & Store Results
# ──────────────────────────────────────────────────────────────────

# 1. Vector Retriever
vector_retriever = db.as_retriever(search_kwargs={"k": 5})

# 2. BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(documents=chunks)
bm25_retriever.k = 5

hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

all_retrieval_results = []  # Store all results for RRF

for i, query in enumerate(query_variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
    docs = hybrid_retriever.invoke(query)
    all_retrieval_results.append(docs)  # Store for RRF calculation
    
    print(f"Retrieved {len(docs)} documents:\n")
    
    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
    
    print("-" * 50)

print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")


# all_retrieval_results = [
#     [Doc1, Doc2, Doc3, Doc4, Doc5],  ← Query 1 results
#     [Doc2, Doc1, Doc6, Doc7, Doc3],  ← Query 2 results  
#     [Doc8, Doc2, Doc9, Doc10, Doc11] ← Query 3 results
# ]
# Problem: Redundancy of chunks, difference in ranks of chunks


# ──────────────────────────────────────────────────────────────────
# Step 3: Apply Reciprocal Rank Fusion (RRF)
# ──────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")
    
    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
    all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}
    
    # For verbose output - track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1
    
    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content
            
            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            chunk_id = chunk_id_map[chunk_content]
            
            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)
            
            # Add to RRF score
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f"    Preview: {chunk_content[:80]}...")
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    
    if verbose:
        print(f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
    
    return sorted_chunks


# ──────────────────────────────────────────────────────────────────
# Step 4: Display Final Fused Results
# ──────────────────────────────────────────────────────────────────

fused_results = reciprocal_rank_fusion(all_retrieval_results, k=60, verbose=True)

print("\n" + "="*60)
print("FINAL RRF RANKING")
print("="*60)

print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):
    print(f"Rank {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:100]}...")
    print("-" * 50)


# ──────────────────────────────────────────────────────────────────
# Step 5: Apply Cohere Reranking
# ──────────────────────────────────────────────────────────────────

# Use the fused RRF results as input to the reranker
retrieved_docs = [doc for doc, _ in fused_results]

print("\n" + "-"*50)
print("After Cohere Reranking:")
print("-"*50)

# Initialize Cohere reranker
reranker = CohereRerank(model="rerank-english-v3.0", top_n=10)

# Rerank the retrieved documents
reranked_docs = reranker.compress_documents(retrieved_docs, query)

# Show reranked results
for i, doc in enumerate(reranked_docs, 1):
    print(f"Rank {i:2d}. {doc.page_content[:50]}...")


# ──────────────────────────────────────────────────────────────────
# Step 6: Final Answer Generation
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("FINAL: RAG with Reranked Context")
print("-"*40)

# Use top 5 reranked documents for final answer
top_reranked = reranked_docs[:5]

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in top_reranked])}

Please provide a clear, helpful answer using only the information from these documents."""

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

result = llm.invoke(messages)
print("Generated Response:")
print(result.content)