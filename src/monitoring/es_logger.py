import psutil
from datetime import datetime
from elasticsearch import Elasticsearch
import os

ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://elasticsearch:9200")
INDEX_NAME = "mlflow-metrics"

es = Elasticsearch(ES_HOST)

def log_to_elasticsearch(event_type: str, payload: dict):
    document = {
        "event_type": event_type,
        "timestamp": datetime.utcnow(),
        **payload,
    }
    es.index(index=INDEX_NAME, document=document)


def log_system_metrics(stage: str):
    system_metrics = {
        "stage": stage,
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
    }

    log_to_elasticsearch(
        event_type="system_metrics",
        payload=system_metrics,
    )
