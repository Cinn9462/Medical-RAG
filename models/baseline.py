"""
gpt-4o-mini forces model to be certain, exposing softmaxed logits does not reveal confidence
Temprature of 7.5 gives more reasonable probabilities for testing later on
"""

# region imports, models, constants
import ast
import os
import math
import pandas as pd
import time as t
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model_name = "gpt-4o-mini"
errors = 0
OUT_FILE = "output.csv"
# endregion

def softmax(logits, temperature=7.5):
    logits = [l / temperature for l in logits]
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]

def get_option_probs(messages):
    global errors

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
    return probs + [end - start] + [not missing]

def eval_medmcqa(q):
    messages = [
        {"role": "system", "content": "Output your answer as a single letter (A, B, C, D). No explanation"},
        {"role": "user", "content": f"""Question: {q["question"]}
    A) {q["opa"]}
    B) {q["opb"]}
    C) {q["opc"]}
    D) {q["opd"]}
Answer: """}
    ]
    return get_option_probs(messages)

def eval_mmlu(q):
    options = ast.literal_eval(q["options"])
    messages = [
        {"role": "system", "content": "Output your answer as a single letter (A, B, C, D). No explanation."},
        {"role": "user", "content": f"""Question: {q["centerpiece"]}
    A) {options[0]}
    B) {options[1]}
    C) {options[2]}
    D) {options[3]}
Answer: """}
    ]
    return get_option_probs(messages)

def write(res):
    df = pd.DataFrame(res)
    file_exists = os.path.exists(OUT_FILE)
    df.to_csv(OUT_FILE, mode="a", header=not file_exists, index=False)

# Warmup
response = client.chat.completions.create(model = model_name, messages = [{"role" : "user", "content" : "Hi"}], max_completion_tokens = 1)

# MEDMCQA Test
# print("Evaluating MedMCQA test")
# results = []
# df = pd.read_csv("question-set/medmcqa_20.csv")
# for i, q in enumerate(df.to_dict("records")):
#     res = eval_medmcqa(q)
#     results.append({
#         "id": q["id"],
#         "source" : q["source"],
#         "subject": q["subject_name"],
#         "probA": res[0],
#         "probB": res[1],
#         "probC": res[2],
#         "probD": res[3],
#         "answer": q["cop"],
#         "time": res[4]
#     })
#     print(results[-1])
#     if i % 100 == 0 and i != 0:
#         write(results)
#         results.clear()

# if results:
#     write(results)
#     results.clear()

# MEDMCQA
print("Evaluating MedMCQA splits")
results = []
df = pd.read_csv("question-set/medmcqa_1000.csv")
for i, q in enumerate(df.to_dict("records")):
    res = eval_medmcqa(q)
    results.append({
        "id": q["id"],
        "source" : q["source"],
        "subject": q["subject_name"],
        "probA": res[0],
        "probB": res[1],
        "probC": res[2],
        "probD": res[3],
        "answer": q["cop"],
        "time": res[4]
    })
    if i % 100 == 0 and i != 0:
        write(results)
        results.clear()

if results:
    write(results)
    results.clear()

# MMLU
print("Evaluating MMLU splits")
results = []
df = pd.read_csv("question-set/mmlu_1000.csv")
for i, q in enumerate(df.to_dict("records")):
    res = eval_mmlu(q)
    results.append({
        "id": q["id"],
        "source" : q["source"],
        "subject": q["subject"],
        "probA": res[0],
        "probB": res[1],
        "probC": res[2],
        "probD": res[3],
        "answer": q["correct_options_idx"][1],
        "time": res[4]
    })
    if i % 100 == 0 and i != 0:
        write(results)
        results.clear()

if results: 
    write(results)
    results.clear()