import os
import threading
from typing import Dict, List

import requests

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HF_API_URLS = [
    f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}",
    f"https://api-inference.huggingface.co/models/{MODEL_ID}",
]


class SentimentModel:
    def __init__(self, api_token: str | None = None, timeout: int = 30):
        token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        if not token:
            raise ValueError(
                "Missing Hugging Face token. Set HUGGINGFACE_API_TOKEN or pass api_token."
            )

        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._thread_local = threading.local()

    def _get_session(self) -> requests.Session:
        # Keep a separate session per worker thread for safe connection reuse.
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(self.headers)
            self._thread_local.session = session
        return session

    def predict(self, text: str) -> Dict[str, object]:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        payload = {"inputs": text.strip()}
        last_error = None
        data = None
        session = self._get_session()

        for api_url in HF_API_URLS:
            try:
                response = session.post(
                    api_url,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code in (404, 410):
                    last_error = RuntimeError(
                        f"{response.status_code} from {api_url}: {response.text[:200]}"
                    )
                    continue

                response.raise_for_status()
                data = response.json()
                break
            except Exception as exc:
                last_error = exc

        if data is None:
            raise RuntimeError(f"All Hugging Face endpoints failed: {last_error}")

        # HF may return either [[{label, score}, ...]] or [{label, score}, ...].
        if isinstance(data, list) and data and isinstance(data[0], list):
            predictions = data[0]
        elif isinstance(data, list):
            predictions = data
        else:
            predictions = []

        if not predictions:
            return {"model": MODEL_ID, "label": None, "score": None, "raw": data}

        predictions = [p for p in predictions if isinstance(p, dict)]
        best = max(predictions, key=lambda item: item.get("score", 0.0))

        return {
            "model": MODEL_ID,
            "label": best.get("label"),
            "score": best.get("score"),
            "raw": predictions,
        }
