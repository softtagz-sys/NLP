import spacy
import pandas as pd

def compare_docs(**kwargs):
    """Compare similarity across multiple documents."""
    df = []
    for arg in kwargs.values():
        sub = []
        for arg2 in kwargs.values():
            sub.append(arg.similarity(arg2))
        df.append(sub)
    return pd.DataFrame(df, columns=list(kwargs.keys()), index=list(kwargs.keys()))


def compare_tokens(doc1, doc2, doc3, token_count=100):
    """Compare similarity between tokens in three documents up to token_count."""
    # Ensure documents are at least token_count in length
    if len(doc1) < token_count or len(doc2) < token_count or len(doc3) < token_count:
        raise ValueError("One or more documents have fewer tokens than token_count.")

    words1, words2, words3 = [], [], []
    values12, values13, values23 = [], [], []

    for i in range(token_count):
        token1, token2, token3 = doc1[i], doc2[i], doc3[i]

        # Calculate similarities if tokens have vectors
        similarity12 = token1.similarity(token2) if token1.has_vector and token2.has_vector else None
        similarity13 = token1.similarity(token3) if token1.has_vector and token3.has_vector else None
        similarity23 = token2.similarity(token3) if token2.has_vector and token3.has_vector else None

        words1.append(token1.text)
        words2.append(token2.text)
        words3.append(token3.text)
        values12.append(similarity12)
        values13.append(similarity13)
        values23.append(similarity23)

    # Create a DataFrame showing token comparisons
    return pd.DataFrame(data={
        "Token 1": words1,
        "Token 2": words2,
        "Token 3": words3,
        "Similarity 1-2": values12,
        "Similarity 1-3": values13,
        "Similarity 2-3": values23
    })
