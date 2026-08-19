# region imports, models, constants
import os
import math
import time as t
import requests
import pandas as pd
from openai import OpenAI
from pymilvus import MilvusClient

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

ZILLIZ_URI = os.environ["ZILLIZ_URI"]
ZILLIZ_TOKEN = os.environ["ZILLIZ_TOKEN"]
zilliz_client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)

COLLECTION_NAME = "MedRAG_statpearls_textbook_collection"
zilliz_client.load_collection(COLLECTION_NAME)

JINA_API_KEY = os.environ["JINA_API_KEY"]
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"

RETRIEVE_TOP_K = 20
RERANK_TOP_N = 5

INSTRUCTIONS_TEMPLATE = "Use context to select the best answer.Output your answer as a single letter ({letters}). No explanation"

errors = 0
# endregion


def softmax(logits, temperature=7.5):
    logits = [l / temperature for l in logits]
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


import tiktoken

_tpm_encoder = tiktoken.get_encoding("cl100k_base")

RERANK_TPM_LIMIT = 90_000  # leave headroom under the 100k cap
RERANK_MAX_DOC_CHARS = 3000  # trim long chunks before sending

_token_log = []  # (timestamp, tokens) within the last 60s


def _estimate_tokens(texts):
    return sum(len(_tpm_encoder.encode(txt)) for txt in texts)


def _throttle_tpm(new_tokens):
    global _token_log
    while True:
        now = t.time()
        _token_log = [(ts, tok) for ts, tok in _token_log if now - ts < 60]
        used = sum(tok for _, tok in _token_log)
        if used + new_tokens <= RERANK_TPM_LIMIT:
            _token_log.append((now, new_tokens))
            return
        sleep_for = 60 - (now - _token_log[0][0])
        t.sleep(max(sleep_for, 1))


def rerank(query, candidates, top_n=RERANK_TOP_N, max_retries=6):
    if not candidates:
        return candidates

    docs = [c["content"][:RERANK_MAX_DOC_CHARS] for c in candidates]
    est_tokens = _estimate_tokens([query] + docs)

    payload = {
        "model": JINA_RERANK_MODEL,
        "query": query,
        "top_n": top_n,
        "documents": docs,
    }
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    for attempt in range(max_retries):
        _throttle_tpm(est_tokens)
        resp = requests.post(JINA_RERANK_URL, headers=headers, json=payload, timeout=30)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else min(2 ** attempt, 30)
            print(f"Jina 429, retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
            t.sleep(wait)
            continue

        resp.raise_for_status()
        data = resp.json()
        return [{**candidates[item["index"]], "score": item["relevance_score"]} for item in data["results"]]

    raise RuntimeError("Jina rerank failed after max retries (rate limited)")


def get_context(prompt, retrieve_k=RETRIEVE_TOP_K, rerank_n=RERANK_TOP_N):
    query_embedding = openai_client.embeddings.create(
        input=[prompt], model=EMBED_MODEL
    ).data[0].embedding

    hits = zilliz_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=retrieve_k,
        output_fields=["id", "source", "content"],
        search_params={"metric_type": "COSINE", "params": {}}
    )[0]

    candidates = [
        {
            "score": r["distance"],
            "id": r["id"],
            "source": r["entity"].get("source"),
            "content": r["entity"].get("content"),
        }
        for r in hits
    ]

    results = rerank(prompt, candidates, top_n=rerank_n)
    context = "\n\n".join(r["content"] for r in results)
    return context, results
# endregion


# region Eval (baseline logic: gpt-4o-mini + logprobs, RAG-augmented)
def _eval_common(question_text, option_texts, letters, q_id):
    global errors

    option_block = "\n".join(f"    {l}) {txt}" for l, txt in zip(letters, option_texts))
    context, search_results = get_context(f"Question: {question_text}\n{option_block}")

    messages = [
        {"role": "system", "content": INSTRUCTIONS_TEMPLATE.format(letters=", ".join(letters))},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question_text}\n{option_block}\nAnswer: "}
    ]

    start = t.time()
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_completion_tokens=1,
        logprobs=True,
        top_logprobs=20,
        temperature=0
    )
    elapsed = t.time() - start

    logprob_dict = {item.token: item.logprob for item in response.choices[0].logprobs.content[0].top_logprobs}

    option_logprobs, missing = [], []
    for opt in letters:
        if opt in logprob_dict:
            option_logprobs.append(logprob_dict[opt])
        elif (" " + opt) in logprob_dict:
            option_logprobs.append(logprob_dict[" " + opt])
        else:
            option_logprobs.append(-100)
            missing.append(opt)

    if missing:
        print(f"Warning: options {missing} not in top-20 logprobs (id={q_id})")
        errors += 1

    return softmax(option_logprobs), elapsed, not missing, search_results


def eval_medmcqa(q):
    letters = ["A", "B", "C", "D"]
    options = [q["a"], q["b"], q["c"], q["d"]]
    return _eval_common(q["question"], options, letters, q["id"])


def eval_medqa(q):
    letters = ["A", "B", "C", "D", "E"]
    options = [q["a"], q["b"], q["c"], q["d"], q["e"]]
    return _eval_common(q["question"], options, letters, q["id"])
# endregion


def write(res, out_file):
    df = pd.DataFrame(res)
    file_exists = os.path.exists(out_file)
    df.to_csv(out_file, mode="a", header=not file_exists, index=False)


def run_eval(csv_path, eval_fn, out_file, prob_keys, skip):
    """Runs an eval function over a question CSV, writing results in chunks of 100."""
    print(f"Evaluating {csv_path}")
    results = []
    df = pd.read_csv(csv_path)
    for i, q in enumerate(df.to_dict("records")):
        if i < skip: continue
        probs, elapsed, ok, search_results = eval_fn(q)
        row = {"id": q["id"], "source": q.get("source")}
        row.update({key: prob for key, prob in zip(prob_keys, probs)})
        row.update({
            "answer": q["answer"],
            "time": elapsed,
            "error": not ok,
            "ctx_sources": [r["source"] for r in search_results],
            "ctx_ids": [r["id"] for r in search_results],
            "similarity": [r["score"] for r in search_results],
        })
        results.append(row)

        if (i + 1) % 100 == 0:
            print(f"Evaluated {i + 1} questions")
            write(results, out_file)
            results.clear()

    if results:
        write(results, out_file)


# Warmup
openai_client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": "Hi"}], max_completion_tokens=1)

# run_eval("question-set/medmcqa_1000.csv", eval_medmcqa, "medmcqa_results.csv", ["probA", "probB", "probC", "probD"], 0)
run_eval("question-set/medqa_1273.csv", eval_medqa, "medqa_results.csv", ["probA", "probB", "probC", "probD", "probE"], 0)
