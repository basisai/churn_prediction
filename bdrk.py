import os
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, List, MutableMapping, Optional
from uuid import UUID, uuid4

from google.cloud import bigquery


@dataclass(frozen=True)
class Prediction:
    entity_id: UUID
    features: List[float]
    requestBody: str
    output: float
    server_id: str
    created_at: datetime = datetime.now(tz=timezone.utc)


class PredictionStore:
    def __init__(self):
        # TODO: replace in memory store with BigQuery
        self._client = bigquery.Client()
        self._store: MutableMapping[UUID, Prediction] = {}
        self._table = self._client.get_table("span-staging.expt_prediction_store.prediction_v1")
        # Assumes gthread and uses thread local storage
        self._scope = threading.local()

    def save(self, prediction: Prediction):
        """
        Stores the prediction asynchronously to BigQuery.

        :param prediction: The completed prediction
        :type prediction: Prediction
        """
        # self._store[prediction.entity_id] = prediction
        data = asdict(prediction)
        print(data)
        data["entity_id"] = str(prediction.entity_id)
        # TODO: Supports bytes type which is not json serializable
        errors = self._client.insert_rows(self._table, [data])
        if errors == []:
            print("New rows have been added.")

    def load(self, key: UUID) -> Prediction:
        """
        Loads a prediction by its primary key.

        :param key: The primary key of the prediction.
        :type key: UUID
        :return: The past prediction.
        :rtype: Prediction
        """
        return self._store[key]

    def log(self, **kwargs):
        """
        Logs partial attributes to the currently active prediction.

        :raises RuntimeError: When no active scope is available.
        """
        if not hasattr(self._scope, "active"):
            raise RuntimeError("No active scope available")
        current = getattr(self._scope, "active")
        current.update(**kwargs)

    @contextmanager
    def activate(self) -> UUID:
        """
        Activates a prediction scope.

        :raises RuntimeError: When an active scope already exists.
        :yield: The prediction ID of the active scope
        :rtype: UUID
        """
        if hasattr(self._scope, "active"):
            raise RuntimeError("An active scope already exists")

        key = uuid4()
        setattr(self._scope, "active", {
            "server_id": os.environ["SERVER_ID"],
            "entity_id": key
        })

        try:
            yield key
        finally:
            current = getattr(self._scope, "active")
            delattr(self._scope, "active")
            self.save(Prediction(**current))


store = PredictionStore()


def track(func: Callable) -> Callable:
    """
    Middleware for automatically tracking predict function output.

    :param func: The predict function to decorate
    :type func: Callable
    :return: The decorated function
    :rtype: Callable
    """

    def wrapper(*args, **kwargs):
        with store.activate() as key:
            resp = func(*args, **kwargs)
            # resp.header["X-Prediction-ID"] = key
            store.log(output=resp)
            return resp

    return wrapper
