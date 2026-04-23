import json
import logging
import asyncio
import httpx
import requests
import time

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Singleton client to communicate with Ollama REST API.
    Handles both sync (for CLI pipeline) and async (for FastAPI streaming) calls.
    Lock prevents VRAM overflow on the GPU.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OllamaClient, cls).__new__(cls)
            cls._instance.base_url = "http://localhost:11434"
            cls._instance.model = "deepseek-r1:8b" # As requested
        return cls._instance

    def generate_sync(self, prompt: str, max_retries: int = 3) -> str:
        """
        Synchronous generation for the batch CLI pipeline.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=90.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
            except Exception as e:
                logger.warning(f"Ollama call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error("All retries failed for Ollama generation.")
                    raise
                time.sleep(2.0)
        return ""

    async def generate_async(self, prompt: str, max_retries: int = 3) -> str:
        """
        Asynchronous non-streaming generation.
        Uses the lock to prevent concurrent requests from crashing Ollama.
        """
        async with self._lock:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        response = await client.post(
                            f"{self.base_url}/api/generate",
                            json=payload
                        )
                        response.raise_for_status()
                        data = response.json()
                        return data.get("response", "")
                except Exception as e:
                    logger.warning(f"Ollama async call failed (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        logger.error("All retries failed for Ollama async generation.")
                        raise
                    await asyncio.sleep(2.0)
            return ""

    async def generate_stream(self, prompt: str):
        """
        Asynchronous streaming generation for the FastAPI backend.
        Uses a lock to prevent concurrent requests from crashing Ollama.
        """
        async with self._lock:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "format": "json",
                "options": {
                    "num_ctx": 4096
                }
            }
            
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_lines():
                            if not chunk:
                                continue
                            data = json.loads(chunk)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
            except Exception as e:
                yield f'{{"error": "Failed to connect to Ollama: {str(e)}"}}'

llm_client = OllamaClient()
