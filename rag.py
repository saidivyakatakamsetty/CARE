import chromadb
from knowledge_base import DOCUMENTS

client = chromadb.Client()  # in-memory database, resets each run (fine for now)
collection = client.get_or_create_collection("clinical_kb")

# Build three parallel lists from DOCUMENTS
ids = [d["id"] for d in DOCUMENTS]
texts = [d["text"] for d in DOCUMENTS]
metadatas = [{"title": d["title"]} for d in DOCUMENTS]

collection.add(
    ids=ids,
    documents=texts,
    metadatas=metadatas,
)


def retrieve(query, k=2):
    results = collection.query(query_texts=[query], n_results=k)

    # results looks like:
    # {
    #   "ids": [["lactate", "shock-index"]],          <- outer list = one per query
    #   "documents": [["text1...", "text2..."]],
    #   "metadatas": [[{"title": "..."}, {"title": "..."}]],
    # }
    # We only sent ONE query, so we always want index [0] of each list
    # to "unwrap" down to the results for our single query.

    out = []
    for i in range(len(results["ids"][0])):
        out.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "title": results["metadatas"][0][i]["title"],
        })
    return out


if __name__ == "__main__":
    results = retrieve("lactate rising trend in a deteriorating patient", k=2)
    for r in results:
        print(r["id"], "-", r["title"])

def query_from_driving_feature(feature_name):
    """Map a feature name (e.g. 'lactate_slope') to a natural-language RAG query."""
    topic_map = {
        "lactate": "lactate rising trend in a deteriorating patient",
        "hr": "heart rate abnormal trend",
        "resp": "respiratory rate abnormal trend",
        "shock_index": "elevated shock index",
        "qsofa_approx": "qSOFA criteria met",
    }
    for key, query in topic_map.items():
        if feature_name.startswith(key):
            return query
    return feature_name  # fallback: just use the raw feature name


if __name__ == "__main__":
    # Simulate: the model said "lactate_slope" was the top driving feature
    query = query_from_driving_feature("lactate_slope")
    print(f"Generated query: '{query}'")

    results = retrieve(query, k=2)
    for r in results:
        print(" ->", r["id"], "-", r["title"])