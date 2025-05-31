import json
import re
from collections import defaultdict
from retriever import VectorRetriever
from metadata_loader import MetadataLoader
from prompt_builder import build_prompt
from gpt_wrapper import ask_gpt
from utils import print_title, print_question, print_context_snippets, print_answer, generate_paraphrases

import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

nltk.download('wordnet')
nltk.download('omw-1.4')


with open("config.json", "r") as f:
    config = json.load(f)

cross_encoder_model_name = config.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
cross_encoder = CrossEncoder(cross_encoder_model_name)

embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
embedding_model = SentenceTransformer(embedding_model_name)

def get_synonyms(word):
    
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def keywords_to_vectors(keywords):
    
    expanded_keywords = set()
    for kw in keywords:
        expanded_keywords.add(kw.lower())
        expanded_keywords.update(get_synonyms(kw.lower()))
    expanded_keywords = list(expanded_keywords)
    vectors = embedding_model.encode(expanded_keywords)
    return expanded_keywords, vectors

def compute_keyword_similarity(text, keywords, kw_vectors):
    
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0.0
    word_vectors = embedding_model.encode(words)
    sim_sum = 0.0
    for w_vec in word_vectors:
        
        sims = np.dot(kw_vectors, w_vec) / (np.linalg.norm(kw_vectors, axis=1) * np.linalg.norm(w_vec) + 1e-10)
        sim_sum += max(sims)
    return sim_sum / len(words)

def compute_weight(text, keywords, base_weight=0.3, similarity_threshold=0.3):
    
    if not keywords:
        return 0.0
    expanded_keywords, kw_vectors = keywords_to_vectors(keywords)
    sim = compute_keyword_similarity(text, expanded_keywords, kw_vectors)
    if sim >= similarity_threshold:
        return base_weight
    return 0.0

def filter_by_confidence(results, threshold):
    return [(idx, score) for idx, score in results if score >= threshold]

def retrieve_and_rerank(query, retriever, metadata, keywords, confidence_threshold, top_k=3):
    all_results = []
    paraphrases = generate_paraphrases(query, n=3)
    for para in paraphrases:
        indices, scores = retriever.search(para, top_k=5)
        all_results.extend(list(zip(indices, scores)))

    score_map = defaultdict(list)
    for idx, score in all_results:
        score_map[idx].append(score)

    unique_results = {idx: max(scores) for idx, scores in score_map.items()}
    filtered_results = filter_by_confidence(unique_results.items(), confidence_threshold)

    unique_indices = [idx for idx, _ in filtered_results]
    chunks_data = metadata.get_chunks_by_indices(unique_indices)
    idx_to_chunk = dict(zip(unique_indices, chunks_data))

    rerank_candidates = []
    for idx, base_score in filtered_results:
        text = idx_to_chunk.get(idx)
        if not text:
            continue
        weight = compute_weight(text, keywords)
        combined_score = base_score + weight
        rerank_candidates.append((idx, combined_score, text))

    rerank_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = rerank_candidates[:10]

    cross_inp = [(query, c[2]) for c in top_candidates]
    cross_scores = cross_encoder.predict(cross_inp)


    ranked = sorted(zip(top_candidates, cross_scores), key=lambda x: x[1], reverse=True)

    final_texts = [item[0][2] for item in ranked[:top_k]]

    return final_texts

def main():
    print_title()
    retriever = VectorRetriever(model_name=embedding_model_name)
    metadata = MetadataLoader()

    keywords = config.get("keywords", [])
    confidence_threshold = config.get("confidence_threshold", 0.1)
    top_k = config.get("top_k", 3)

    while True:
        query = input("\nAsk something or type 'exit': ").strip()
        if query.lower() in ['exit', 'quit']:
            print("exit")
            break

        if not query:
            print("404")
            continue

        print_question(query)

        retrieved_texts = retrieve_and_rerank(
            query=query,
            retriever=retriever,
            metadata=metadata,
            keywords=keywords,
            confidence_threshold=confidence_threshold,
            top_k=top_k
        )

        print_context_snippets(retrieved_texts)

        prompt = build_prompt(retrieved_texts, query)
        answer = ask_gpt(prompt)
        print_answer(answer)

if __name__ == '__main__':
    main()
