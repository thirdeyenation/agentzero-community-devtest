from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np

class VectorDB:
    def __init__(self, collection_name="agent_zero_memory", dim=1536):
        self.collection_name = collection_name
        self.dim = dim
        connections.connect("default", host="localhost", port="19530")
        self._create_collection()

    def _create_collection(self):
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(fields, description="Agent Zero Memory")
            self.collection = Collection(self.collection_name, schema)
            
            index_params = {
                "index_type": "HNSW",
                "metric_type": "L2",
                "params": {"M": 16, "efConstruction": 500}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)

    def insert(self, embeddings, contents, metadata_list):
        entities = [
            {"embedding": embedding.tolist(), "content": content, "metadata": metadata}
            for embedding, content, metadata in zip(embeddings, contents, metadata_list)
        ]
        self.collection.insert(entities)
        self.collection.flush()

    def search(self, query_embedding, top_k=5, filter_expr=None):
        self.collection.load()
        search_params = {"metric_type": "L2", "params": {"ef": 100}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["content", "metadata"]
        )
        return [(hit.entity.get('content'), hit.entity.get('metadata'), hit.distance) for hit in results[0]]

# Usage
vector_db = VectorDB()
embeddings = np.random.randn(10, 1536)  # 10 sample embeddings
contents = ["Sample content " + str(i) for i in range(10)]
metadata_list = [{"type": "memory", "timestamp": 12345} for _ in range(10)]
vector_db.insert(embeddings, contents, metadata_list)

query = np.random.randn(1536)
results = vector_db.search(query, top_k=3)