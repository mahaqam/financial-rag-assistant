import argparse, json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def flatten_table(table):
    header=table[0]
    out=[]
    for i,row in enumerate(table[1:], start=1):
        rowname=str(row[0]).strip().lower()
        pieces=[]
        for j in range(1,min(len(row),len(header))):
            h=str(header[j]).strip().lower()
            val=str(row[j]).strip().lower()
            if j==1 and str(header[0]).strip():
                pieces.append(f"{str(header[0]).strip().lower()} the {rowname} of {h} is {val} ;")
            else:
                pieces.append(f"the {rowname} of {h} is {val} ;")
        out.append((f"table_{i}"," ".join(pieces)))
    return out

def candidates(item):
    out=[(f"text_{i}",str(t).strip().lower()) for i,t in enumerate(item["pre_text"]+item["post_text"])]
    return out+flatten_table(item["table"])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="FinQA dev.json")
    ap.add_argument("--out", default="results/retrieval_metrics.json")
    args=ap.parse_args()
    items=json.load(open(args.data,encoding="utf-8"))
    corpus=[]
    for item in items:
        corpus.append(item["qa"]["question"].lower())
        corpus.extend(t for _,t in candidates(item))
    wv=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True).fit(corpus)
    cv=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),sublinear_tf=True,max_features=60000).fit(corpus)
    ks=[1,3,5,10]; hit={k:0 for k in ks}; recall={k:0.0 for k in ks}; mrr=0.0
    for item in items:
        q=item["qa"]["question"].lower(); cand=candidates(item); texts=[x[1] for x in cand]
        sw=(wv.transform(texts) @ wv.transform([q]).T).toarray().ravel()
        sc=(cv.transform(texts) @ cv.transform([q]).T).toarray().ravel()
        score=.7*sw+.3*sc; order=np.argsort(-score)
        ids=[cand[i][0] for i in order]; gold=set(item["qa"]["gold_inds"])
        pos=[ids.index(g)+1 for g in gold if g in ids]
        if pos: mrr+=1/min(pos)
        for k in ks:
            top=set(ids[:k]); hit[k]+=int(bool(gold&top)); recall[k]+=len(gold&top)/len(gold)
    metrics={
        "questions":len(items),"mrr":mrr/len(items),
        "hit_rate_at_k":{str(k):hit[k]/len(items) for k in ks},
        "mean_gold_evidence_recall_at_k":{str(k):recall[k]/len(items) for k in ks}
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=="__main__":
    main()
