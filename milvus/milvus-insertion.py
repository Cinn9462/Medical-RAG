import os
import time
import threading
from pathlib import Path
from collections import deque
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import json
import tiktoken
from openai import OpenAI, BadRequestError
from datasets import load_dataset
from pymilvus import MilvusClient, DataType

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ZILLIZ_URI = os.environ["ZILLIZ_URI"]
ZILLIZ_TOKEN = os.environ["ZILLIZ_TOKEN"]
COLLECTION_NAME = "MedRAG_statpearls_textbook_collection"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Kept well under OpenAI's real 300,000/request hard cap, so local vs. server
# token counting drift never trips the limit.
MAX_TOKENS_PER_REQUEST = 150_000
MAX_ITEMS_PER_REQUEST = 1024

# TPM budget - conservative buffer under tier-1's 1,000,000 TPM limit.
TPM_LIMIT = 800_000

# How many embed+upsert operations run concurrently.
MAX_IN_FLIGHT = 5

# Already inserted into the collection from a prior run.
TEXTBOOK_SKIP = 0
STATPEARLS_SKIP = 0

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
ENCODING = tiktoken.encoding_for_model(EMBED_MODEL)


def count_tokens(text):
    return len(ENCODING.encode(text))


# ---------------------------------------------------------------------------
# Thread-safe sliding-window token rate limiter
# ---------------------------------------------------------------------------
class TokenRateLimiter:
    def __init__(self, tpm_limit=TPM_LIMIT):
        self.tpm_limit = tpm_limit
        self.window = deque()  # (timestamp, tokens)
        self.lock = threading.Lock()

    def _prune(self, now):
        while self.window and now - self.window[0][0] > 60:
            self.window.popleft()

    def acquire(self, tokens_needed):
        while True:
            with self.lock:
                now = time.time()
                self._prune(now)
                used = sum(t for _, t in self.window)
                if used + tokens_needed <= self.tpm_limit:
                    self.window.append((now, tokens_needed))
                    return
                sleep_for = 60 - (now - self.window[0][0]) + 0.1
            print(f"[rate-limiter] sleeping {sleep_for:.1f}s (used {used}/{self.tpm_limit} TPM, need {tokens_needed})")
            time.sleep(max(sleep_for, 0.1))


rate_limiter = TokenRateLimiter()

# ---------------------------------------------------------------------------
# Embedding with retry + auto-split on hard token-per-request errors
# ---------------------------------------------------------------------------
def embed_texts(texts, retries=5):
    total_tokens = sum(count_tokens(t) for t in texts)
    rate_limiter.acquire(total_tokens)

    for attempt in range(retries):
        try:
            resp = openai_client.embeddings.create(input=texts, model=EMBED_MODEL)
            return [d.embedding for d in resp.data]
        except BadRequestError as e:
            if "max_tokens_per_request" in str(e) and len(texts) > 1:
                mid = len(texts) // 2
                print(f"[split] batch of {len(texts)} ({total_tokens} tokens) exceeded "
                      f"OpenAI's request cap — splitting into {mid} + {len(texts) - mid}")
                left = embed_texts(texts[:mid], retries=retries)
                right = embed_texts(texts[mid:], retries=retries)
                return left + right
            print(f"[fatal] non-retryable request error: {e}")
            raise
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait_s = min(2 ** attempt * 2, 60)
            print(f"[retry] embedding call failed: {type(e).__name__}: {e} — "
                  f"retrying in {wait_s}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait_s)


# ---------------------------------------------------------------------------
# Collection setup
# ---------------------------------------------------------------------------
if not client.has_collection(COLLECTION_NAME):
    schema = client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("source", DataType.VARCHAR, max_length=4)
    schema.add_field("content", DataType.VARCHAR, max_length=8192)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBED_DIM)
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

# ---------------------------------------------------------------------------
# Insert helper
# ---------------------------------------------------------------------------
def insert_chunks(chunks, source):
    if not chunks:
        return 0
    t0 = time.time()
    embeddings = embed_texts(chunks)
    t1 = time.time()
    data = [
        {"source": source, "content": chunks[i], "vector": embeddings[i]}
        for i in range(len(chunks))
    ]
    res = client.insert(collection_name=COLLECTION_NAME, data=data)
    t2 = time.time()
    print(f"Inserted {res['insert_count']} chunks (source={source}) "
          f"[embed={t1 - t0:.1f}s insert={t2 - t1:.1f}s]")
    return res["insert_count"]


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------
def make_batches(records_iter, skip=0):
    if skip:
        records_iter = islice(records_iter, skip, None)

    batch, batch_tokens = [], 0
    for text in records_iter:
        if len(text.encode("utf-8")) > 8192:
            continue
        t = count_tokens(text)
        if t > MAX_TOKENS_PER_REQUEST:
            continue
        if batch and (batch_tokens + t > MAX_TOKENS_PER_REQUEST or len(batch) >= MAX_ITEMS_PER_REQUEST):
            yield batch
            batch, batch_tokens = [], 0
        batch.append(text)
        batch_tokens += t
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Parallel processing driver
# ---------------------------------------------------------------------------
def process_stream_parallel(records_iter, source, skip=0):
    total = 0
    with ThreadPoolExecutor(max_workers=MAX_IN_FLIGHT) as executor:
        in_flight = set()
        for batch in make_batches(records_iter, skip=skip):
            if len(in_flight) >= MAX_IN_FLIGHT:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for f in done:
                    total += f.result()
            in_flight.add(executor.submit(insert_chunks, batch, source))

        for f in in_flight:
            total += f.result()

    print(f"Finished {source}: {total} chunks inserted (this run)")


# ---------------------------------------------------------------------------
# Textbooks
# ---------------------------------------------------------------------------
def textbook_stream():
    data = load_dataset("MedRAG/textbooks", split="train", streaming=True)
    for entry in data:
        yield entry["content"]

print("Loading MedRAG/textbooks...")
process_stream_parallel(textbook_stream(), "TEXT", skip=TEXTBOOK_SKIP)

# ---------------------------------------------------------------------------
# StatPearls
# ---------------------------------------------------------------------------
def statpearls_stream():
    folder = Path("milvus/statpearls/chunk")
    for file in folder.glob("*.jsonl"):
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                yield record["content"]

print("Loading StatPearls chunks...")
process_stream_parallel(statpearls_stream(), "STAT", skip=STATPEARLS_SKIP)

# ---------------------------------------------------------------------------
# Index creation
# ---------------------------------------------------------------------------
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)
client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
print("Index created.")