import argparse, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from evaluate_retrieval import candidates

def main():
    ap=argparse.ArgumentParser(description="Retrieve grounded evidence for FinQA questions.")
    ap.add_argument("--data",required=True)
    ap.add_argument("--question",required=True)
    ap.add_argument("--doc-index",type=int,default=0,help="Which FinQA document to search")
    ap.add_argument("--top-k",type=int,default=5)
    args=ap.parse_args()
    items=json.load(open(args.data,encoding="utf-8"))
    item=items[args.doc_index]
    cand=candidates(item); texts=[t for _,t in cand]
    vec=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)
    mat=vec.fit_transform(texts+[args.question])
    scores=(mat[:-1] @ mat[-1].T).toarray().ravel()
    for i in np.argsort(-scores)[:args.top_k]:
        print(f"{cand[i][0]}\t{scores[i]:.4f}\t{cand[i][1]}")
if __name__=="__main__":
    main()
