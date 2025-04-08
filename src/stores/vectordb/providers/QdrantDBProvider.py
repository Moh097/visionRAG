from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
import logging
from typing import List
from models.db_schemes import RetrievedDocument

class QdrantDBProvider(VectorDBInterface):

    def __init__(self, db_path: str, distance_method: str):

        self.client = None
        self.db_path = db_path
        self.distance_method = None

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.logger = logging.getLogger(__name__)

    def connect(self):
        self.client = QdrantClient(path=self.db_path)

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)
    
    def list_all_collections(self) -> List:
        return self.client.get_collections()
    
    def get_collection_info(self, collection_name: str) -> dict:
        return self.client.get_collection(collection_name=collection_name)
    
    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            return self.client.delete_collection(collection_name=collection_name)
        
    def create_collection(self, collection_name: str, 
                          embedding_size: int,
                          do_reset: bool = False):
        if do_reset:
            _ = self.delete_collection(collection_name=collection_name)
        
        if not self.is_collection_existed(collection_name):
            _ = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=self.distance_method
                )
            )
            return True
        
        return False
    
    def insert_one(self, collection_name: str, text: str, vector: list,
                   metadata: dict = None, record_id: str = None):
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Cannot insert to non-existent collection: {collection_name}")
            return False
        
        try:
            _ = self.client.upload_records(
                collection_name=collection_name,
                records=[
                    models.Record(
                        id=[record_id],
                        vector=vector,
                        payload={"text": text, "metadata": metadata}
                    )
                ]
            )
        except Exception as e:
            self.logger.error(f"Error while inserting single record: {e}")
            return False

        return True
    
    def insert_many(self, collection_name: str, texts: list, 
                    vectors: list, metadata: list = None, 
                    record_ids: list = None, batch_size: int = 50):
        
        if metadata is None:
            metadata = [None] * len(texts)

        if record_ids is None:
            record_ids = list(range(0, len(texts)))

        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size

            batch_texts = texts[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_metadata = metadata[i:batch_end]
            batch_record_ids = record_ids[i:batch_end]

            batch_records = []
            for x in range(len(batch_texts)):
                batch_records.append(
                    models.Record(
                        id=batch_record_ids[x],
                        vector=batch_vectors[x],
                        payload={
                            "text": batch_texts[x],
                            "metadata": batch_metadata[x]
                        }
                    )
                )

            try:
                _ = self.client.upload_records(
                    collection_name=collection_name,
                    records=batch_records,
                )
            except Exception as e:
                self.logger.error(f"Error while inserting batch: {e}")
                return False

        return True
    
    def search_by_vector(self, 
                         collection_name: str, 
                         vector: list, 
                         limit: int = 5, 
                         threshold: float = None):
        """
        Perform a semantic search in Qdrant, returning up to 'limit' docs
        whose similarity is >= threshold (0..1).
        
        Qdrant 'score' is actually the distance for COSINE or DOT.
        For COSINE, distance is in [0..2].
        We convert to similarity in [0..1]: similarity = 1 - distance/2
        
        Steps:
          1) Retrieve a large set of docs (big_limit).
          2) Convert distance->similarity, filter by threshold.
          3) Sort by similarity desc.
          4) Return top 'limit' docs.
        """

        # 1) Pull more docs than 'limit' so we can filter some out
        big_limit = max(limit, 1000)
        raw_results = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=big_limit,
        )

        if not raw_results:
            return []

        # 2) Convert distance -> similarity, filter out docs below threshold
        filtered_docs = []
        for r in raw_results:
            similarity = r.score  # Qdrant's score is a "distance"
            
            # if threshold is set, skip doc if similarity < threshold
            if threshold is not None and similarity < threshold:
                continue
            
            doc = RetrievedDocument(
                score=similarity,
                text=r.payload["text"],
            )
            filtered_docs.append(doc)

        # 3) Sort by similarity desc
        filtered_docs.sort(key=lambda d: d.score, reverse=True)

        # 4) Return top 'limit'
        return filtered_docs[:limit]
