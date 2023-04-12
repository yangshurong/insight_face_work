from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
import random

# connect to milvus
connections.connect("default", host="localhost", port="19530")
collection_name = "hello_milmil"
collection = Collection(name=collection_name)
collection.release()
dim = 128
nq = 10
search_vec = [[random.random() for _ in range(dim)] for _ in range(nq)]
search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
limit = 3
for i in range(2):
    results = collection.search(search_vec, "embedding", search_params, limit)
    ids = results[0].ids
    print(f"search result ids: {ids}")
    expr = f"cus_id in {ids}"
    # query to verify the ids exist
    query_res = collection.query(expr)
    print(f"query results: {query_res}")
