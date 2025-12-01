from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load sentiment analysis pipeline
nlp = pipeline("sentiment-analysis")

def detect_bias(texts):
    """
    Detects potential bias in text by analyzing sentiment polarity
    and keyword frequency (e.g., gendered terms).
    """
    results = nlp(texts)
    
    # Simple keyword bias check
    gendered_terms = ["he", "she", "man", "woman", "male", "female"]
    vectorizer = CountVectorizer(vocabulary=gendered_terms, lowercase=True)
    term_counts = vectorizer.fit_transform(texts).toarray()
    
    bias_scores = np.sum(term_counts, axis=1)
    
    return [
        {"text": t, "sentiment": r, "bias_score": b}
        for t, r, b in zip(texts, results, bias_scores)
    ]

if __name__ == "__main__":
    sample_texts = [
        "The ideal candidate should be strong and assertive.",
        "We are looking for a nurturing and caring person."
    ]
    output = detect_bias(sample_texts)
    for o in output:
        print(o)
