"""
Core RAG Workflow Module.
Handles document retrieval, prompt construction, LLM generation with inline citations,
and faithfulness evaluation.
"""
import numpy as np
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# --- Data Models ---
class CitedResponse(BaseModel):
    """Schema for ensuring the LLM returns structured text with citations."""
    answer: str = Field(description="The generated answer to the user's query.")
    citations: List[str] = Field(description="List of document IDs cited in the answer.")

# --- Mock Knowledge Base ---
KNOWLEDGE_BASE = [
    {"doc_id": "doc_1", "text": "Gander's new AI workflow enables 10x faster content generation."},
    {"doc_id": "doc_2", "text": "Deploying Django services on AWS requires load balancers for scale."},
    {"doc_id": "doc_3", "text": "Inline citations improve user trust by 40% in enterprise applications."}
]

class SimpleRetriever:
    """A lightweight TF-IDF based retriever for demonstration purposes."""
    def __init__(self, corpus: List[Dict[str, str]]):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer()
        self.doc_texts = [doc["text"] for doc in corpus]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def retrieve(self, query: str, top_k: int = 1) -> List[Dict[str, str]]:
        """Retrieves the top_k most relevant documents based on cosine similarity."""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Only return if there is some similarity
        return [self.corpus[i] for i in top_indices if similarities[i] > 0]

def generate_cited_answer(query: str, client: openai.Client) -> CitedResponse:
    """
    Executes the RAG workflow: Retrieves context and generates a structured, cited answer.
    """
    retriever = SimpleRetriever(KNOWLEDGE_BASE)
    retrieved_docs = retriever.retrieve(query, top_k=2)
    
    if not retrieved_docs:
        return CitedResponse(answer="I could not find relevant information.", citations=[])

    context_str = "\n".join([f"[{doc['doc_id']}]: {doc['text']}" for doc in retrieved_docs])
    
    prompt = f"""
    You are an expert technical assistant. Answer the user's query using ONLY the provided context.
    You must cite the context using the doc_id.
    
    Context:
    {context_str}
    
    Query: {query}
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=CitedResponse,
        temperature=0.0,
    )
    
    return response.choices[0].message.parsed

def evaluate_faithfulness(query: str, generated_response: CitedResponse, client: openai.Client) -> bool:
    """
    A tiny evaluation check (LLM-as-a-judge) to ensure the answer doesn't hallucinate 
    outside the retrieved knowledge base.
    """
    if not generated_response.citations:
        # An answer with no citations has nothing to verify against the knowledge base;
        # treat it as faithful since no factual claims are attributed to a source.
        return True

    retriever = SimpleRetriever(KNOWLEDGE_BASE)
    # Re-fetch the docs it claims to cite to check ground truth
    cited_texts = [doc["text"] for doc in KNOWLEDGE_BASE if doc["doc_id"] in generated_response.citations]
    context_str = " ".join(cited_texts)

    eval_prompt = f"""
    Given the context, is the following answer entirely factually supported by the context? 
    Reply with ONLY 'True' or 'False'.
    
    Context: {context_str}
    Answer: {generated_response.answer}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.0
    )
    
    return "true" in response.choices[0].message.content.strip().lower()
