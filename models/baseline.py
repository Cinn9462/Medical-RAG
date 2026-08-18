"""
gpt-4o-mini forces model to be certain, exposing softmaxed logits does not reveal confidence
Temprature of 7.5 gives more reasonable probabilities for testing later on
"""

# region imports, models, constants
import os
import math
import pandas as pd
import time as t
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model_name = "gpt-4o-mini"
errors = 0
# endregion

def softmax(logits, temperature=7.5):
    logits = [l / temperature for l in logits]
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]

# returns [[probA, probB, probC, probD], time, error flag]
def eval_medmcqa(q):
    global errors
    messages = [
        {"role": "system", "content": "Output your answer as a single letter (A, B, C, D). No explanation"},
        {"role": "user", "content": f"""Question: {q["question"]}
    A) {q["a"]}
    B) {q["b"]}
    C) {q["c"]}
    D) {q["d"]}
Answer: """}
    ]


    start = t.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=1,
        logprobs=True,
        top_logprobs=20,
        temperature=0
    )
    end = t.time()

    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    logprob_dict = {item.token: item.logprob for item in top_logprobs}

    option_logprobs = []
    missing = []
    for opt in ["A", "B", "C", "D"]:
        if opt in logprob_dict:
            option_logprobs.append(logprob_dict[opt])
        elif (" " + opt) in logprob_dict:
            option_logprobs.append(logprob_dict[(" " + opt)])
        else:
            option_logprobs.append(-100)
            missing.append(opt)

    if missing:
        print(f"Warning: options {missing} not in top-20 logprobs")
        errors += 1

    probs = softmax(option_logprobs)
    return [probs, end - start, not missing]



def eval_medqa(q):
    global errors
    messages = [
        {"role": "system", "content": "Output your answer as a single letter (A, B, C, D, E). No explanation"},
        {"role": "user", "content": f"""Question: {q["question"]}
    A) {q["a"]}
    B) {q["b"]}
    C) {q["c"]}
    D) {q["d"]}
    E) {q["e"]}
Answer: """}
    ]


    start = t.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=1,
        logprobs=True,
        top_logprobs=20,
        temperature=0
    )
    end = t.time()

    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    logprob_dict = {item.token: item.logprob for item in top_logprobs}

    option_logprobs = []
    missing = []
    for opt in ["A", "B", "C", "D", "E"]:
        if opt in logprob_dict:
            option_logprobs.append(logprob_dict[opt])
        elif (" " + opt) in logprob_dict:
            option_logprobs.append(logprob_dict[(" " + opt)])
        else:
            option_logprobs.append(-100)
            missing.append(opt)

    if missing:
        print(f"Warning: options {missing} not in top-20 logprobs")
        errors += 1

    probs = softmax(option_logprobs)
    return [probs, end - start, not missing]

def write(res, OUT_FILE):
    df = pd.DataFrame(res)
    file_exists = os.path.exists(OUT_FILE)
    df.to_csv(OUT_FILE, mode="a", header=not file_exists, index=False)

# Warmup
response = client.chat.completions.create(model = model_name, messages = [{"role" : "user", "content" : "Hi"}], max_completion_tokens = 1)

# # MEDMCQA Test
# print("Evaluating MedMCQA test")
# results = []
# df = pd.read_csv("question-set/medmcqa_20.csv")
# for i, q in enumerate(df.to_dict("records")):
#     probs, time, missing = eval_medmcqa(q)
#     results.append({
#         "id": q["id"],
#         "source" : q["source"],
#         "probA": probs[0],
#         "probB": probs[1],
#         "probC": probs[2],
#         "probD": probs[3],
#         "answer": q["answer"],
#         "time": time,
#         "error" : missing
#     })
#     print(results[-1])
#     if i % 100 == 0 and i != 0:
#         write(results, "medmcqa_results.csv")
#         results.clear()

# if results:
#     write(results, "medmcqa_results.csv")
#     results.clear()

# # MEDMCQA
# print("Evaluating MedMCQA test")
# results = []
# df = pd.read_csv("question-set/medmcqa_1000.csv")
# for i, q in enumerate(df.to_dict("records")):
#     probs, time, missing = eval_medmcqa(q)
#     results.append({
#         "id": q["id"],
#         "source" : q["source"],
#         "probA": probs[0],
#         "probB": probs[1],
#         "probC": probs[2],
#         "probD": probs[3],
#         "answer": q["answer"],
#         "time": time,
#         "error" : missing
#     })
#     if i % 100 == 0 and i != 0:
#         write(results, "medmcqa_results.csv")
#         results.clear()

# if results:
#     write(results, "medmcqa_results.csv")
#     results.clear()

# MEDQA
print("Evaluating MedQA test")
results = []
df = pd.read_csv("question-set/medqa_1273.csv")
for i, q in enumerate(df.to_dict("records")):
    probs, time, missing = eval_medqa(q)
    results.append({
        "id": q["id"],
        "source" : q["source"],
        "probA": probs[0],
        "probB": probs[1],
        "probC": probs[2],
        "probD": probs[3],
        "probE": probs[4],
        "answer": q["answer"],
        "time": time,
        "error" : missing
    })
    if i % 100 == 0 and i != 0:
        write(results, "medqa_results.csv")
        results.clear()

if results:
    write(results, "medqa_results.csv")
    results.clear()
