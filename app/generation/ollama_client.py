"""Persistent HTTP client for local Ollama generation requests."""

import json
from collections.abc import Iterator

import httpx
from loguru import logger

from app.generation.config import GenerationConfig
from app.utils.errors import OllamaConnectionError, OllamaResponseError


class OllamaClient:
    """Submit generation requests to a locally running Ollama instance."""

    def __init__(self, config: GenerationConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        self._client = client or httpx.Client(base_url=config.base_url.rstrip("/"), timeout=config.timeout_seconds)
        self._owns_client = client is None

    def generate(self, prompt: str) -> str:
        """Generate a complete response with bounded retries and validation."""
        if self._config.streaming:
            return "".join(self.stream(prompt))
        payload = self._payload(prompt, stream=False)

        last_error: Exception | None = None

        for attempt in range(self._config.retries + 1):
            try:
                response = self._client.post("/api/generate", json=payload)
                response.raise_for_status()
                return self._response_text(response.json())

            except httpx.TimeoutException as error:
                logger.exception("Timeout contacting Ollama")
                last_error = error
                failure = OllamaConnectionError(
                    f"Ollama timed out after {self._config.timeout_seconds}s"
                )

            except httpx.RequestError as error:
                logger.exception("RequestError contacting Ollama")
                last_error = error
                failure = OllamaConnectionError(
                    f"Unable to connect to Ollama: {error}"
                )

            except httpx.HTTPStatusError as error:
                logger.error("Status code: {}", error.response.status_code)
                logger.error("Response body: {}", error.response.text)

                last_error = error
                message = self._error_message(error.response)
                failure = OllamaResponseError(
                    f"Ollama rejected model '{self._config.model_name}': {message}"
                )

            except (TypeError, ValueError, OllamaResponseError) as error:
                last_error = error
                failure = OllamaResponseError(
                    f"Invalid Ollama response: {error}"
                )

            if attempt == self._config.retries:
                raise failure from last_error

            logger.warning(
                "Ollama request failed; retrying ({}/{})",
                attempt + 1,
                self._config.retries,
            )

        raise OllamaResponseError("Ollama request unexpectedly completed without a response")

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield response fragments from Ollama's newline-delimited streaming API."""
        try:
            with self._client.stream("POST", "/api/generate", json=self._payload(prompt, stream=True)) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    payload = json.loads(line)
                    if "error" in payload:
                        raise OllamaResponseError(str(payload["error"]))
                    fragment = payload.get("response", "")
                    if not isinstance(fragment, str):
                        raise OllamaResponseError("Ollama stream response is invalid")
                    yield fragment
        except httpx.HTTPStatusError as error:
            message = self._error_message(error.response)
            raise OllamaResponseError(f"Ollama rejected model '{self._config.model_name}': {message}") from error
        except httpx.RequestError as error:
            raise OllamaConnectionError(f"Unable to stream from Ollama: {error}") from error
        except (TypeError, ValueError, OllamaResponseError) as error:
            raise OllamaResponseError(f"Invalid Ollama stream response: {error}") from error

    def close(self) -> None:
        """Close the internally owned persistent HTTP client."""
        if self._owns_client:
            self._client.close()

    def _payload(self, prompt: str, stream: bool) -> dict[str, object]:
        return {
            "model": self._config.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self._config.temperature,
                "top_p": self._config.top_p,
                "num_predict": self._config.max_tokens,
            },
        }

    @staticmethod
    def _response_text(payload: object) -> str:
        if not isinstance(payload, dict):
            raise OllamaResponseError("Ollama response must be an object")
        if "error" in payload:
            raise OllamaResponseError(str(payload["error"]))
        answer = payload.get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise OllamaResponseError("Ollama response did not contain a non-empty answer")
        return answer.strip()

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
            return str(payload.get("error", response.text)) if isinstance(payload, dict) else response.text
        except ValueError:
            return response.text
