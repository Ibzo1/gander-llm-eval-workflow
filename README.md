# Gander LLM RAG Workflow Sample

## Overview
This repository contains a self-contained, production-grade LLM workflow demonstrating Retrieval-Augmented Generation (RAG) with inline citations and automated evaluation. 

## Design Choices & Trade-offs
1. **Pydantic Structured Outputs:** I used Pydantic to enforce strict schemas on the LLM output. *Trade-off:* It slightly increases latency and token usage due to the schema prompt, but it guarantees machine-readable outputs (citations mapped to text), which is critical for downstream API consumption in a Django backend.
2. **In-Memory Vector Search:** For demonstration portability, this module uses a simple cosine similarity search via `numpy` and `scikit-learn` TF-IDF rather than spinning up a full FAISS/Pinecone instance. *Trade-off:* Perfect for small-scale testing and low overhead, but O(N) complexity means it must be swapped for an ANN index (like FAISS) for production workloads at scale.
3. **Automated Evals:** Included a lightweight `Faithfulness` eval check. It uses a secondary LLM call to verify if the generated answer is entirely supported by the retrieved context. *Trade-off:* Running an LLM-as-a-judge at runtime doubles the cost/latency, so in a real production environment, this would be an offline asynchronous task (e.g., via Celery) or used strictly during QA/CI pipelines.
