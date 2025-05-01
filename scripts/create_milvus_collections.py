# Connect to Milvus service on port 19530

from pymilvus import MilvusClient, DataType

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# Create a database called "vaultsense"
client.create_database(
    db_name="vaultsense"
)



client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
    db_name="vaultsense"
)

# Create the following collections in milvus
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768)

# 3.3. Prepare index parameters
index_params = client.prepare_index_params()

# 3.4. Add indexes
index_params.add_index(
    field_name="id",
    index_type="AUTOINDEX"
)

index_params.add_index(
    field_name="embedding", 
    index_type="AUTOINDEX",
    metric_type="L2"
)

# 3.5. Create a collection with the index loaded simultaneously
client.create_collection(
    collection_name="documents",
    schema=schema,
    index_params=index_params
)

res = client.get_load_state(
    collection_name="documents"
)

# Create a collection called "ltm" for storing Long-Term Memory Embeddings

client.create_collection(
    collection_name="ltm",
    schema=schema,
    index_params=index_params
)

res = client.get_load_state(
    collection_name="ltm"
)


