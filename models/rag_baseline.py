# region imports, models, constants
import os
import math
import time as t
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

INSTRUCTIONS_TEMPLATE = "Output your answer as a single letter ({letters}). No explanation"

errors = 0
# endregion


def softmax(logits, temperature=7.5):
    logits = [l / temperature for l in logits]
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


# region Retrieval
def get_context(prompt, top_k=5):
    query_embedding = openai_client.embeddings.create(
        input=[prompt], model=EMBED_MODEL
    ).data[0].embedding

    hits = zilliz_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k,
        output_fields=["id", "source", "content"],
        search_params={"metric_type": "COSINE", "params": {}}
    )[0]

    results = [
        {
            "score": r["distance"],
            "id": r["id"],
            "source": r["entity"].get("source"),
            "content": r["entity"].get("content"),
        }
        for r in hits
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
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


def run_eval(csv_path, eval_fn, out_file, prob_keys):
    """Runs an eval function over a question CSV, writing results in chunks of 100."""
    print(f"Evaluating {csv_path}")
    results = []
    df = pd.read_csv(csv_path)
    for i, q in enumerate(df.to_dict("records")):
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

run_eval("question-set/medmcqa_1000.csv", eval_medmcqa, "medmcqa_results.csv", ["probA", "probB", "probC", "probD"])
run_eval("question-set/medqa_1273.csv", eval_medqa, "medqa_results.csv", ["probA", "probB", "probC", "probD", "probE"])
