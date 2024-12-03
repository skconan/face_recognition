import uuid
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Batch

VECTOR_DB_PATH = "./db/face_vectors"
COLLECTION_NAME = "face_rec"
DIM = 512

client = QdrantClient(path=VECTOR_DB_PATH)

try:
    client.get_collection(COLLECTION_NAME)
    print("Qdrant collection face_rec already exists.")
except Exception as e:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )
    print("Qdrant collection face_rec created.")
    print(e)


def registration(embedding, person_id):
    try:
        embedding = np.expand_dims(embedding, axis=0).astype("float64")
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=Batch(
                ids=[str(uuid.uuid4())],
                vectors=[embedding[0]],
                payloads=[{"person_id": person_id}],
            ),
        )
        print(f"Embedding registered for person_id: {person_id}")
        return True
    except Exception as e:
        print(f"Failed to register embedding: {e}")
        return False


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def search(query_embedding, k=5):
    query_embedding = np.expand_dims(query_embedding, axis=0).astype("float64")

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding[0].tolist(),
        limit=k,
        with_vectors=True,
    )
    distances = [res.score for res in results]
    indices = [res.payload["person_id"] for res in results]
    return np.array(distances), indices
