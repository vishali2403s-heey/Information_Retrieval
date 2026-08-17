import os
import math
import re


# ============================================================
# INFORMATION RETRIEVAL USING TF-IDF
# ============================================================


# ============================================================
# 1. READ 10 DOCUMENTS
# ============================================================

folder = "./documents"

documents = []
document_names = []


# Natural sorting: document1, document2, ..., document10
def document_number(filename):
    match = re.search(r'(\d+)', filename)

    if match:
        return int(match.group(1))

    return 999


files = sorted(
    [file for file in os.listdir(folder) if file.endswith(".txt")],
    key=document_number
)


for filename in files:

    file_path = os.path.join(folder, filename)

    with open(file_path, "r", encoding="utf-8") as file:

        text = file.read().lower()

    # Remove punctuation and extract words
    words = re.findall(r'\b[a-zA-Z]+\b', text)

    documents.append(words)
    document_names.append(filename)


print("=" * 70)
print("          INFORMATION RETRIEVAL USING TF-IDF")
print("=" * 70)

print("\nNumber of Documents Loaded:", len(documents))


# ============================================================
# 2. MANUAL QUERY
# ============================================================

print("\n" + "=" * 70)
print("                 MANUAL QUERY")
print("=" * 70)

query = input("\nEnter your query (5-6 words): ").lower()

query_words = re.findall(r'\b[a-zA-Z]+\b', query)

print("\nEntered Query:")
print(query)


# ============================================================
# 3. TERM FREQUENCY (TF)
# ============================================================

def calculate_tf(words):

    tf = {}

    total_words = len(words)

    for word in words:

        if word in tf:
            tf[word] += 1
        else:
            tf[word] = 1

    # TF = Term Frequency / Total Number of Words

    for word in tf:

        tf[word] = tf[word] / total_words

    return tf


# Calculate TF for every document

tf_documents = []

for document in documents:

    tf = calculate_tf(document)

    tf_documents.append(tf)


# ============================================================
# DISPLAY TF
# ============================================================

print("\n" + "=" * 70)
print("                 1. TERM FREQUENCY (TF)")
print("=" * 70)

for i in range(len(tf_documents)):

    print("\n", document_names[i])

    # Display first 10 terms
    count = 0

    for term, value in tf_documents[i].items():

        print(f"{term:<20} TF = {value:.6f}")

        count += 1

        if count == 10:
            print("...")
            break


# ============================================================
# 4. INVERSE DOCUMENT FREQUENCY (IDF)
# ============================================================

N = len(documents)

# Collect all unique terms

all_terms = set()

for document in documents:

    all_terms.update(document)


idf = {}


for term in sorted(all_terms):

    document_frequency = 0

    # Count number of documents containing the term

    for document in documents:

        if term in document:

            document_frequency += 1

    # IDF = log(N / Document Frequency)

    idf[term] = math.log(N / document_frequency)


# ============================================================
# DISPLAY IDF
# ============================================================

print("\n" + "=" * 70)
print("              2. INVERSE DOCUMENT FREQUENCY (IDF)")
print("=" * 70)

count = 0

for term, value in idf.items():

    print(f"{term:<20} IDF = {value:.6f}")

    count += 1

    if count == 30:
        print("...")
        break


# ============================================================
# 5. TF-IDF
# ============================================================

tfidf_documents = []


for tf in tf_documents:

    tfidf = {}

    for term in tf:

        # TF-IDF = TF × IDF

        tfidf[term] = tf[term] * idf[term]

    tfidf_documents.append(tfidf)


# ============================================================
# DISPLAY TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("                    3. TF-IDF")
print("=" * 70)


for i in range(len(tfidf_documents)):

    print("\n", document_names[i])

    count = 0

    for term, value in tfidf_documents[i].items():

        print(f"{term:<20} TF-IDF = {value:.6f}")

        count += 1

        if count == 10:
            print("...")
            break


# ============================================================
# 6. QUERY TF-IDF
# ============================================================

query_tf = calculate_tf(query_words)

query_tfidf = {}


for term in query_tf:

    if term in idf:

        query_tfidf[term] = query_tf[term] * idf[term]

    else:

        # If query word does not occur in any document

        query_tfidf[term] = 0


# ============================================================
# DISPLAY QUERY TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("                  QUERY TF-IDF")
print("=" * 70)

for term, value in query_tfidf.items():

    print(f"{term:<20} TF-IDF = {value:.6f}")


# ============================================================
# 7. COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector1, vector2):

    # Get all terms from both vectors

    all_words = set(vector1.keys()) | set(vector2.keys())

    dot_product = 0

    magnitude1 = 0

    magnitude2 = 0


    for word in all_words:

        value1 = vector1.get(word, 0)

        value2 = vector2.get(word, 0)

        # Dot product

        dot_product += value1 * value2

        # Magnitudes

        magnitude1 += value1 ** 2

        magnitude2 += value2 ** 2


    magnitude1 = math.sqrt(magnitude1)

    magnitude2 = math.sqrt(magnitude2)


    # Avoid division by zero

    if magnitude1 == 0 or magnitude2 == 0:

        return 0


    # Cosine Similarity

    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


# ============================================================
# CALCULATE SIMILARITY FOR ALL DOCUMENTS
# ============================================================

similarities = []


for i in range(len(tfidf_documents)):

    similarity = cosine_similarity(
        query_tfidf,
        tfidf_documents[i]
    )

    similarities.append(
        (document_names[i], similarity)
    )


# ============================================================
# 8. DOCUMENT RANKING
# ============================================================

ranked_documents = sorted(
    similarities,
    key=lambda x: x[1],
    reverse=True
)


# ============================================================
# DISPLAY DOCUMENT RANKING
# ============================================================

print("\n" + "=" * 70)
print("                  4. DOCUMENT RANKING")
print("=" * 70)

print(
    f"{'RANK':<10}"
    f"{'DOCUMENT':<20}"
    f"{'COSINE SIMILARITY':<20}"
)

print("-" * 55)


for rank, (doc_name, similarity) in enumerate(
    ranked_documents,
    start=1
):

    print(
        f"{rank:<10}"
        f"{doc_name:<20}"
        f"{similarity:.6f}"
    )


# ============================================================
# 9. SIMILARITY RANGE
# ============================================================

similarity_values = [
    similarity
    for doc_name, similarity in ranked_documents
]


maximum_similarity = max(similarity_values)

minimum_similarity = min(similarity_values)

similarity_range = (
    maximum_similarity - minimum_similarity
)


print("\n" + "=" * 70)
print("                  5. SIMILARITY RANGE")
print("=" * 70)

print(
    f"Maximum Similarity : {maximum_similarity:.6f}"
)

print(
    f"Minimum Similarity : {minimum_similarity:.6f}"
)

print(
    f"Similarity Range   : {similarity_range:.6f}"
)


# ============================================================
# 10. FINAL RESULT
# ============================================================

most_relevant_document = ranked_documents[0][0]

highest_similarity = ranked_documents[0][1]


print("\n" + "=" * 70)
print("                     FINAL RESULT")
print("=" * 70)

print("\nQuery:")

print(query)


print("\nMost Relevant Document:")

print(most_relevant_document)


print("\nHighest Cosine Similarity:")

print(f"{highest_similarity:.6f}")


print("\nSimilarity Range:")

print(f"{similarity_range:.6f}")


print("\n" + "=" * 70)

print("              PROGRAM COMPLETED SUCCESSFULLY")

print("=" * 70)