import datetime
import chromadb
import traceback
import pandas

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
csv_file = "COA_OpenData.csv"

def get_collection():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    if collection.count() == 0:
        df = pandas.read_csv(csv_file)
        required_columns = {"Name", "Type", "Address", "Tel", "City", "Town", "CreateDate", "HostWords"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file miss necessary column: {required_columns - set(df.columns)}")
        for idx, row in df.iterrows():
            metadata = {
                "file_name": csv_file,
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": datetime.datetime.strptime(row["CreateDate"], '%Y-%m-%d').timestamp()
            }
            collection.add(
                ids=[str(idx)],
                metadatas = [metadata],
                documents = [row["HostWords"]]
            )
    return collection

def get_query_results(collection, question, city, store_type, start_date, end_date):
    filters = [
        {"type": {"$in": store_type}},
        {"city": {"$in": city}}
    ]
    if start_date:
        filters.append({"date": {"$gte": int(start_date.timestamp())}}) # greater than or equal
    if end_date:
        filters.append({"date": {"$lte": int(end_date.timestamp())}}) # less than or equal

    query_results = collection.query(
        query_texts = [question],
        n_results = 10,
        include = ["metadatas", "distances"],
        where = {
            "$and": filters
        }
    )
    return query_results



def generate_hw01():
    return get_collection()
    
def generate_hw02(question, city, store_type, start_date, end_date):
    collection = get_collection()
    hw02_query_results = get_query_results(collection, question, city, store_type, start_date, end_date)
    names, distances = hw02_query_results.get("metadatas",[[]])[0], hw02_query_results.get("distances", [[]])[0]
    sorted_results = [x.get("name") for x, d in sorted(zip(names, distances), key=lambda x: x[1]) if d < 0.2]
    #print(sorted_results)
    return sorted_results

    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    collection = get_collection()
    selected = collection.get(where = {"name": store_name})
    metadatas = [{**item, "new_store_name": new_store_name} for item in selected.get("metadatas", [])]
    collection.upsert(ids = selected.get("ids", []), metadatas = metadatas, documents = selected.get("documents", []))
    hw03_query_results = get_query_results(collection, question, city, store_type, start_date = None, end_date = None)
    names, distances = hw03_query_results.get("metadatas", [[]])[0], hw03_query_results.get("distances", [[]])[0]
    sorted_results = [x.get("new_store_name", x.get("name")) for x, d in sorted(zip(names, distances), key=lambda x: x[1]) if d < 0.2]
    #print(sorted_results)
    return sorted_results

    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    return collection

# print("-----hw3-1")
# generate_hw01()
# print("\n")
# print("-----hw3-2")
# generate_hw02("我想要找有關茶餐點的店家", ["宜蘭縣", "新北市"], ["美食"], datetime.datetime(2024, 4, 1), datetime.datetime(2024, 5, 1))
# print("\n")
# print("-----hw3-3")
# generate_hw03("我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵", "耄饕客棧", "田媽媽（耄饕客棧）", ["南投縣"], ["美食"])