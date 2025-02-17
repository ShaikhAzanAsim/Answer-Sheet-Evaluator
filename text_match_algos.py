from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
import torch
from Levenshtein import distance as levenshtein_distance
from nltk.util import ngrams
from collections import Counter

# Sample texts
correct_answer = "The mitochondria is the powerhouse of the cell."
student_answer = "mitochoondria is found inside a human brain."

# Preprocessing function
def preprocess(text):
    return text.lower()

correct_answer = preprocess(correct_answer)
student_answer = preprocess(student_answer)

# 1. Cosine Similarity (TF-IDF)
def cosine_similarity_tfidf(correct, student):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([correct, student])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

# 2. Jaccard Similarity
def jaccard_similarity(correct, student):
    set1, set2 = set(correct.split()), set(student.split())
    return len(set1 & set2) / len(set1 | set2)

# 3. Levenshtein Distance
def levenshtein_similarity(correct, student):
    max_len = max(len(correct), len(student))
    distance = levenshtein_distance(correct, student)
    return 1 - distance / max_len

# 4. BLEU Score
def bleu_similarity(correct, student):
    reference = [correct.split()]
    candidate = student.split()
    return sentence_bleu(reference, candidate)

# 5. ROUGE Score
def rouge_similarity(correct, student):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(correct, student)
    return scores

# 6. Deep Learning-based Semantic Similarity
def semantic_similarity(correct, student, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def encode(text):
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**tokens).last_hidden_state.mean(dim=1)
        return embeddings

    embedding1 = encode(correct)
    embedding2 = encode(student)
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2).item()
    return similarity

# Calculate similarities
print("Results:")
print(f"1. Cosine Similarity (TF-IDF): {cosine_similarity_tfidf(correct_answer, student_answer):.2f}")
print(f"2. Jaccard Similarity: {jaccard_similarity(correct_answer, student_answer):.2f}")
print(f"3. Levenshtein Similarity: {levenshtein_similarity(correct_answer, student_answer):.2f}")
print(f"4. BLEU Score: {bleu_similarity(correct_answer, student_answer):.2f}")

rouge_scores = rouge_similarity(correct_answer, student_answer)
print("5. ROUGE Scores:")
for metric, score in rouge_scores.items():
    print(f"   {metric}: Precision={score.precision:.2f}, Recall={score.recall:.2f}, F1={score.fmeasure:.2f}")

semantic_score = semantic_similarity(correct_answer, student_answer)
print(f"6. Semantic Similarity (BERT): {semantic_score:.2f}")
