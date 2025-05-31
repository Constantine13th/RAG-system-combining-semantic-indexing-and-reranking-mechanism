# RAG-system-combining-semantic-indexing-and-reranking-mechanism

Semantic Search and Question Answering System for Historical Texts

This system integrates semantic chunking, keyword and named entity extraction, embedding-based representation, vector retrieval with reranking, and OpenAI-powered answer generation to enable semantic search and question answering on historical book content.

In the data processing phase, chapters are used as the basic unit. Regular expressions are employed to identify chapter titles and segment the text accordingly. Each chapter is further divided into smaller chunks (paragraphs), with a BERT tokenizer ensuring that each chunk remains within a predefined token limit to preserve semantic continuity. Keywords (top 5) are extracted from each chunk using KeyBERT, while named entities—such as people, locations, dates, and organizations—are identified using SpaCy’s en_core_web_trf model. Each processed entry is saved in JSONL format, containing the chunk ID, original chapter title, content, extracted keywords, and named entities.

For vector construction, the project uses the lightweight MiniLM-L6-v2 model from the SentenceTransformers library to encode text into high-dimensional dense semantic vectors. These embeddings are indexed using FAISS with L2 distance for efficient approximate nearest neighbor (ANN) retrieval. When a user submits a query, the system retrieves relevant chunks based on a keyword-augmented semantic similarity mechanism. These results are then reranked using a CrossEncoder to improve precision.

# Dependency Installation

```
pip install regex nltk "spacy[transformers]" keybert sentence-transformers faiss-cpu numpy openai rich
python -m nltk.downloader wordnet omw-1.4
python -m spacy download en_core_web_trf

```

