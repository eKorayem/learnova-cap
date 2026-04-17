from typing import List

from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
from models.db_schemas import RetrievedDocument
import logging

class QdrantDBProvider(VectorDBInterface):

    def __init__(self, url: str = None, db_path: str = None, distance_method: str = "cosine"):

        self.client = None
        self.url = url
        self.db_path = db_path
        self.distance_method = None

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.logger = logging.getLogger(__name__)

    def connect(self):
        if self.url:
            # HTTP connection for Docker/production
            self.logger.info(f"Connecting to Qdrant at {self.url}")
            self.client = QdrantClient(url=self.url)
        elif self.db_path:
            # File-based for local development
            self.logger.info(f"Connecting to Qdrant at {self.db_path}")
            self.client = QdrantClient(path=self.db_path)
        else:
            raise ValueError("Either url or db_path must be provided")

    def disconnect(self):
        if self.client:
            self.client.close()
            self.client = None

    def is_collection_exists(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)

    def list_of_all_collections(self) -> List:
        return self.client.get_collections()

    def get_collection_info(self, collection_name: str) -> dict:
        return self.client.get_collection(collection_name=collection_name)

    def delete_collection(self, collection_name: str):
        if self.is_collection_exists(collection_name=collection_name):
            self.client.delete_collection(collection_name=collection_name)


    def create_collection(self, collection_name: str,
                                embedding_size: int,
                                do_rest: bool = False):
        if do_rest:
            _ = self.delete_collection(collection_name=collection_name)

        if not self.is_collection_exists(collection_name=collection_name):
            _ = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=embedding_size, distance=self.distance_method),
            )
            return True

        return False

    def insert_one(self, collection_name: str, text: str,
                   vector: list, metadata: dict=None,
                   recored_id: str=None):
        if not self.is_collection_exists(collection_name=collection_name):
            self.logger.error(f"Can NOT insert new record to non-existed collection {collection_name}")
            return False

        try:
            _ = self.client.upload_records(
                collection_name=collection_name,
                records=[
                    models.Record(
                        id=recored_id,
                        vector=vector,
                        payload={
                            "text" : text,
                            "metadata" : metadata
                        }
                    )
                ]
            )
        except Exception as e:
            self.logger.error(f"Error while inserting batch {e}")
            return False

        return True

    def insert_many(self, collection_name: str, texts: List[str],
                    vectors: List[list], metadatas: List[dict]=None,
                    record_ids: List[str]=None, batch_size: int=50):

        if metadatas is None:
            metadatas = [None] * len(texts)

        if record_ids is None:
            record_ids = list(range(0, len(texts)))

        for i in range(0, len(texts), batch_size):
            batch_end = i+batch_size

            batch_texts = texts[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_record_ids = record_ids[i:batch_end]

            batch_records = [
                models.Record(
                    id=batch_record_ids[x],
                    vector=batch_vectors[x],
                    payload={
                        "text" : batch_texts[x],
                        "metadata" : batch_metadatas[x]
                    }
                )
                for x in range(len(batch_texts))
            ]

            try:
                _ = self.client.upload_records(
                    collection_name=collection_name,
                    records=batch_records
                )
            except Exception as e:
                self.logger.error(f"Error while inserting batch {e}")
                return False

        return True

    def search_by_vector(self, collection_name: str,
                     vector: list,
                     limit: int=5) -> List[dict]:

        results = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit
        )

        if not results or len(results) == 0:
            return None

        return [
            RetrievedDocument(**{
                "text": r.payload["text"],
                "score": r.score
                }
            )
            for r in results
        ]
