from pymilvus import connections, utility 
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema 
 
# connect to milvus 
connections.connect("default", host="localhost", port="19530")

# create a collection with customized primary field: id_field 
dim = 128 
id_field = FieldSchema(name="cus_id", dtype=DataType.INT64, is_primary=True) 
age_field = FieldSchema(name="age", dtype=DataType.VARCHAR, description="age") 
embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim) 
schema = CollectionSchema(fields=[id_field, age_field, embedding_field], 
                          auto_id=False, description="hello MilMil") 
collection_name = "hello_milmil" 
collection = Collection(name=collection_name, schema=schema) 
 
import random 
# insert data with customized ids 
nb = 300 
ids = [i for i in range(nb)] 
ages = ["book_" + str(i) for i in range(nb)] 
embeddings = [[random.random() for _ in range(dim)] for _ in range(nb)] 
 
entities = [ids, ages, embeddings] 
ins_res = collection.insert(entities) 
print(f"insert entities primary keys: {ins_res.primary_keys}")  
# collection.load()

nq = 10
search_vec = [[random.random() for _ in range(dim)] for _ in range(nq)]
search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
limit = 3
for i in range(2):
    results = collection.search(search_vec, embedding_field.name, search_params, limit)
    ids = results[0].ids
    print(f"search result ids: {ids}")
    expr = f"cus_id in {ids}"
    # query to verify the ids exist
    query_res = collection.query(expr)
    print(f"query results: {query_res}")

# from pymilvus import CollectionSchema, FieldSchema, DataType
# from pymilvus import connections
# connections.connect(
#   alias="default", 
#   host='localhost', 
#   port='19530'
# )
# book_id = FieldSchema(
#   name="book_id", 
#   dtype=DataType.INT64, 
#   is_primary=True, 
# )
# book_name = FieldSchema(
#   name="book_name", 
#   dtype=DataType.VARCHAR, 
#   max_length=200,
# )
# word_count = FieldSchema(
#   name="word_count", 
#   dtype=DataType.INT64,  
# )
# book_intro = FieldSchema(
#   name="book_intro", 
#   dtype=DataType.FLOAT_VECTOR, 
#   dim=2
# )
# schema = CollectionSchema(
#   fields=[book_id, book_name, word_count, book_intro], 
#   description="Test book search"
# )
# collection_name = "book"


# from pymilvus import Collection
# collection = Collection(
#     name=collection_name, 
#     schema=schema, 
#     using='default', 
#     shards_num=2
#     )

# import random
# data = [
#   [i for i in range(2000)],
#   [str(i) for i in range(2000)],
#   [i for i in range(10000, 12000)],
#   [[random.random() for _ in range(2)] for _ in range(2000)],
# ]


# collection = Collection("book")      # Get an existing collection.
# mr = collection.insert(data)