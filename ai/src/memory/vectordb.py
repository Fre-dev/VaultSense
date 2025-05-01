import os
from typing import Any, Dict, List
from pydantic import BaseModel
from pymilvus import MilvusClient
from src.utils.logger import setup_logger
logger = setup_logger(__name__)


class VectorStoreConnection(BaseModel):
    customer: str
    client: Any


VECTOR_STORE_CONNECTIONS: list[VectorStoreConnection] = []


class VectorStore:
    def __init__(self, customer: str):
        try:
            self.customer = customer
            self.client = MilvusClient(
                uri=f"http://{os.getenv('MILVUS_HOST')}:{os.getenv('MILVUS_PORT')}",
                db_name=customer,
                user=os.getenv("MILVUS_USER"),
                password=os.getenv("MILVUS_PASSWORD"),
            )
            VECTOR_STORE_CONNECTIONS.append(
                VectorStoreConnection(customer=customer, client=self.client)
            )
        except Exception as e:
            logger.error(f"Failed to initialize Milvus client: {e}")
            raise

    @staticmethod
    def get_vector_store_connection(customer: str) -> MilvusClient:
        existing_connection = next(
            (
                conn.client
                for conn in VECTOR_STORE_CONNECTIONS
                if conn.customer == customer
            ),
            None,
        )

        if existing_connection:
            logger.info(
                f"Using existing vectorstore connection for customer {customer}"
            )
            return existing_connection
        else:
            logger.info(f"Creating new vectorstore connection for customer {customer}")
            new_connection = VectorStore(customer=customer)
            return new_connection.client
