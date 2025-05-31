def build_prompt(retrieved_texts, user_question):
    """
    Constructs a context-constrained prompt optimized for a RAG system focused on Byzantine history.

    Parameters:
    - retrieved_texts (list of str): Relevant textual passages retrieved from historical sources.
    - user_question (str): The user's query related to Byzantine history.

    Returns:
    - str: A structured prompt that enforces evidence-based, context-aware answering with clear boundaries.
    """

    
    context = "\n\n---\n\n".join(
        [f"[{i+1}] {text}" for i, text in enumerate(retrieved_texts)]
    )

    prompt = f"""
You are a historical assistant with deep expertise in Byzantine history. Your task is to answer user questions **based strictly on the provided reference texts**.

Instructions:
- Use **only** the information contained in the reference texts below. Do **not** rely on external knowledge, prior assumptions, or general historical facts unless explicitly stated in the texts.
- You **must not hallucinate** or guess. If the answer is not clearly supported by the reference texts, respond exactly with:
  "The answer is not available in the provided sources."
- When citing evidence, refer to the corresponding numbered source in square brackets (e.g., [1], [2]).
- You may synthesize information from **multiple references** to form a comprehensive answer.
- Be accurate, concise, and maintain academic tone.

Reference Texts:
{context}

User Question:
{user_question}

Answer:
"""
    return prompt.strip()
