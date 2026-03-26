"""
Unit tests for the RAG workflow and evaluation metrics.
Run using: pytest test_rag_workflow.py
"""
import pytest
from unittest.mock import MagicMock
from rag_workflow import SimpleRetriever, CitedResponse, evaluate_faithfulness

# --- Mock Data ---
MOCK_CORPUS = [
    {"doc_id": "doc_A", "text": "The sky is blue due to Rayleigh scattering."},
    {"doc_id": "doc_B", "text": "Grass is green because of chlorophyll."}
]

def test_simple_retriever():
    """Test that the TF-IDF retriever correctly identifies semantic matches."""
    retriever = SimpleRetriever(MOCK_CORPUS)
    results = retriever.retrieve("Why is the sky blue?", top_k=1)
    
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc_A"

def test_evaluate_faithfulness_true():
    """Test the tiny eval check passes when the answer matches the context."""
    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "True"
    mock_client.chat.completions.create.return_value = mock_response

    # Mock a good generated response
    good_response = CitedResponse(
        answer="Gander's workflow makes generation 10x faster.", 
        citations=["doc_1"]
    )
    
    result = evaluate_faithfulness("How fast is Gander?", good_response, mock_client)
    assert result is True

def test_evaluate_faithfulness_false():
    """Test the tiny eval check fails when the answer hallucinates."""
    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "False"
    mock_client.chat.completions.create.return_value = mock_response

    # Mock a hallucinated generated response
    bad_response = CitedResponse(
        answer="Gander's workflow makes generation 1000x faster and cures diseases.", 
        citations=["doc_1"]
    )
    
    result = evaluate_faithfulness("How fast is Gander?", bad_response, mock_client)
    assert result is False
