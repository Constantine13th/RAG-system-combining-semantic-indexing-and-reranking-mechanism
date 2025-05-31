import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json

class VectorRetriever:
    def __init__(self, index_path='data/rag_index.faiss', model_name='all-MiniLM-L6-v2'):
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(model_name)

    def search(self, query, top_k=5):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, top_k)
        return indices[0], distances[0]
