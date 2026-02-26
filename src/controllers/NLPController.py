from .BaseController import BaseController
from models.db_schemas import Project, DataChunk
from typing import List
from stores.llm.LLMEnums import DocumentTypeEnums
import json

class NLPController(BaseController):
    
    def __init__(self, vectordb_client, generation_client, embedding_client):
        super().__init__()
        
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        
    
    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}"
    
    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)

    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)
        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        ) if collection_info else None
    
    def index_into_vector_db(self, project: Project,
                             chunks: List[DataChunk],
                             chunks_ids: List[int],
                             do_reset: bool = False):
        
        collection_name = self.create_collection_name(project_id=project.project_id)
        
        
        # Extract texts and metadatas from chunks
        texts = [chunk.chunk_text for chunk in chunks]
        metadatas = [chunk.chunk_metadata for chunk in chunks]
        
        vectors = [
            self.embedding_client.embed_text(
                text=text,
                document_type=DocumentTypeEnums.DOCUMENT.value
            )
            for text in texts
        ]
        
        # Create collection if not exists (or reset if requested)
        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_rest=do_reset
        )
            
        # Insert into vector DB
        _ = self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            vectors=vectors,
            metadatas=metadatas,
            record_ids=chunks_ids
        )
        
        return True
    
    def search_vector_db_collection(self, project: Project, text: str, limit: int = 5):
        collection_name = self.create_collection_name(project_id=project.project_id)
        
        vector = self.embedding_client.embed_text(text=text,
                                                 document_type=DocumentTypeEnums.QUERY.value)
        
        if not vector or len(vector)==0:
            return False
        
        search_results = self.vectordb_client.search_by_vector(
            collection_name,
            vector=vector,
            limit=limit
        )
        
        print(f"Search Results: {search_results}")
        return json.loads(
            json.dumps(search_results, default=lambda x: x.__dict__)
        ) if search_results else None
        