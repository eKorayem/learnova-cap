from .BaseDataModel import BaseDataModel
from .db_schemas import DataChunk
from .enums.DataBaseEnums import DataBaseEnums
from bson.objectid import ObjectId
from pymongo import InsertOne
from typing import Optional


class ChunkModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[
            DataBaseEnums.COLLECTION_CHUNK_NAME.value
        ]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnums.COLLECTION_CHUNK_NAME.value not in all_collections:
            self.collection = self.db_client[DataBaseEnums.COLLECTION_CHUNK_NAME.value]
            indexes = DataChunk.get_indexes()
            for index in indexes:
                await self.collection.create_index(
                    index["key"],
                    name=index["name"],
                    unique=index["unique"]
                )

    async def create_chunk(self, chunk: DataChunk) -> DataChunk:
        data = chunk.dict(by_alias=True, exclude={"id"})
        result = await self.collection.insert_one(data)
        chunk.id = result.inserted_id
        return chunk

    async def get_chunk(self, chunk_id: str) -> DataChunk | None:
        record = await self.collection.find_one(
            {"_id": ObjectId(chunk_id)}
        )
        if record is None:
            return None
        return DataChunk(**record)

    async def insert_many_chunks(
        self,
        chunks: list[DataChunk],
        batch_size: int = 100
    ) -> int:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            operations = [
                InsertOne(chunk.dict(by_alias=True, exclude={"id"}))
                for chunk in batch
            ]
            await self.collection.bulk_write(operations)
        return len(chunks)

    async def delete_chunks_by_project_id(
        self,
        project_id: str,
        chunk_type: Optional[str] = None   # if None, deletes ALL types
    ):
        query = {"project_id": project_id}
        if chunk_type:
            query["chunk_type"] = chunk_type

        result = await self.collection.delete_many(query)
        return result.deleted_count

    async def get_chunks_by_project_id(
        self,
        project_id: str,
        page_no: int = 1,
        page_size: int = 50,
        chunk_type: Optional[str] = None   # if None, returns ALL types
    ) -> list[DataChunk]:

        query = {"project_id": project_id}
        if chunk_type:
            query["chunk_type"] = chunk_type

        skip = (page_no - 1) * page_size
        cursor = self.collection.find(query).skip(skip).limit(page_size)

        chunks = []
        async for record in cursor:
            chunks.append(DataChunk(**record))

        return chunks

    async def get_chunks_by_project_object_id(
        self,
        chunk_project_id: ObjectId,
        page_no: int = 1,
        page_size: int = 100,
        chunk_type: Optional[str] = None
    ) -> list[DataChunk]:

        query = {"chunk_project_id": chunk_project_id}
        if chunk_type:
            query["chunk_type"] = chunk_type

        skip = (page_no - 1) * page_size
        cursor = self.collection.find(query).skip(skip).limit(page_size)

        chunks = []
        async for doc in cursor:
            chunks.append(DataChunk(**doc))

        return chunks