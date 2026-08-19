# region Imports, Cloud setup
from pymilvus import MilvusClient, DataType
from datasets import load_dataset
from openai import OpenAI
import json
import os
import time
from pathlib import Path

ZILLIZ_URI = os.environ["ZILLIZ_URI"]
ZILLIZ_TOKEN = os.environ["ZILLIZ_TOKEN"]
client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
# endregion


print(client.list_collections())
for c in client.list_collections():
    print(f"Collection: {c}")
    print(client.get_collection_stats(c))
    print(client.describe_collection(c))