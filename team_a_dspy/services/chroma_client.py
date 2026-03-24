from __future__ import annotations

from chromadb import HttpClient, PersistentClient

from services.config import settings


class ChromaClient:
    """
    Client for interacting with ChromaDB vector store.
    Stores and retrieves schema interpretation passages for Elasticsearch fields.
    """

    def __init__(self, dev: bool = False) -> None:
        self.collection_name = settings.chroma_collection_name
        if dev:
            client = PersistentClient(path=settings.chroma_persistent_path)
        else:
            client = HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )

        self.collection = client.get_or_create_collection(name=self.collection_name)

    def count(self) -> int:
        return int(self.collection.count())

    def add_documents(self, interpreted_fields: list[dict] | dict) -> None:
        """
        Add one or many interpreted fields to the ChromaDB collection.

        Expected shape for each item:
        {
            "field_name": "V2Persons.V1Person.keyword",
            "field_type": "keyword",
            "interpretation": "People/entity name for exact terms aggregations ..."
        }
        """
        if isinstance(interpreted_fields, dict):
            interpreted_fields = [interpreted_fields]

        if not interpreted_fields:
            return

        ids: list[str] = []
        metadatas: list[dict] = []
        documents: list[str] = []

        for doc in interpreted_fields:
            field_name = str(doc["field_name"])
            field_type = str(doc.get("field_type", "unknown"))
            interpretation = str(doc.get("interpretation", "")).strip()
            if not interpretation:
                continue

            ids.append(field_name)
            metadatas.append(
                {
                    "field_name": field_name,
                    "field_type": field_type,
                }
            )
            documents.append(interpretation)

        if not ids:
            return

        self.collection.upsert(
            ids=ids,
            metadatas=metadatas,
            documents=documents,
        )

    def query(self, query_text: str, k: int = 6) -> dict:
        """
        Query the ChromaDB collection for the most relevant interpretations.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=max(1, int(k)),
            include=["documents", "metadatas", "distances"],
        )
        return results