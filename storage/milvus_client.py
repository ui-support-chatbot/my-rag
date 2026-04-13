from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class MilvusClient:
    uri: str = "./milvus.db"
    db_name: str = "default"
    _client: Any = None

    def __post_init__(self):
        if self._client is None:
            from pymilvus import MilvusClient as PMilvusClient

            self._client = PMilvusClient(uri=self.uri, db_name=self.db_name)

    @property
    def client(self):
        return self._client

    def create_collection(self, collection_name: str, dimension: int):
        if not self._client.has_collection(collection_name):
            schema = self._client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field(name="id", dtype=17, is_primary=True)
            schema.add_field(name="doc_id", dtype=18, max_length=256)
            schema.add_field(name="text", dtype=18, max_length=65535)
            schema.add_field(name="chunk_index", dtype=9)
            schema.add_field(name="dense_embedding", dtype=101, dim=dimension)
            schema.add_field(name="sparse_embedding", dtype=104)

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="dense_embedding",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 128},
            )
            index_params.add_index(
                field_name="sparse_embedding",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )

            self._client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"Created collection: {collection_name}")

    def insert(self, collection_name: str, data: List[Dict]):
        result = self._client.insert(collection_name=collection_name, data=data)
        return result

    def query(
        self,
        collection_name: str,
        filter: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
    ):
        return self._client.query(
            collection_name=collection_name,
            filter=filter,
            output_fields=output_fields,
            limit=limit,
        )

    def hybrid_search(
        self,
        collection_name: str,
        reqs: List,
        ranker,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
    ):
        return self._client.hybrid_search(
            collection_name=collection_name,
            search_requests=reqs,
            ranker=ranker,
            limit=limit,
            output_fields=output_fields,
        )

    def has_collection(self, collection_name: str) -> bool:
        return self._client.has_collection(collection_name)

    def list_collections(self) -> List[str]:
        return self._client.list_collections()
