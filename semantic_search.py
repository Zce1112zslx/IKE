from sentence_transformers import SentenceTransformer, util
import torch
import pickle

with open('embeddings.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['sentences']
    stored_embeddings = stored_data['embeddings']

corpus_embeddings = torch.tensor(stored_embeddings[2000*13:])
query_embeddings = torch.tensor(stored_embeddings[:13*2000])

corpus_embeddings = corpus_embeddings.to('cuda')
corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

query_embeddings = query_embeddings.to('cuda')
query_embeddings = util.normalize_embeddings(query_embeddings)

hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score, top_k=64)

# print(hits)

for i, hit in enumerate(hits):
    # print("Query:", stored_sentences[i])
    for k in range(len(hit)):
        print(hit[k]['corpus_id']+2000*13, end=" ")
        # print(hit[k]['score'])
    print("")