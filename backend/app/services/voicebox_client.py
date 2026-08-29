from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from app.config import settings


class VoiceboxError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class VoiceboxClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.voicebox_api_url).rstrip("/")

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

    async def list_profiles(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.base_url}/profiles")
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return data
            return data.get("profiles", data.get("items", []))

    async def create_profile(
        self,
        name: str,
        engine: str = "qwen3-tts",
        language: str = "en",
        description: str = "",
    ) -> dict[str, Any]:
        payload = {
            "name": name,
            "engine": engine,
            "language": language,
            "description": description,
            "profile_type": "cloned",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{self.base_url}/profiles", json=payload)
            if response.status_code >= 400:
                raise VoiceboxError(response.text, response.status_code)
            return response.json()

    async def add_profile_sample(
        self,
        profile_id: str,
        audio_path: Path,
        reference_text: str,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with audio_path.open("rb") as audio_file:
                response = await client.post(
                    f"{self.base_url}/profiles/{profile_id}/samples",
                    files={"file": (audio_path.name, audio_file, "audio/wav")},
                    data={"reference_text": reference_text},
                )
            if response.status_code >= 400:
                raise VoiceboxError(response.text, response.status_code)
            return response.json()

    async def delete_profile(self, profile_id: str) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(f"{self.base_url}/profiles/{profile_id}")
            if response.status_code >= 400:
                raise VoiceboxError(response.text, response.status_code)

    async def transcribe(self, audio_path: Path, model: str = "whisper-turbo") -> str:
        async with httpx.AsyncClient(timeout=300.0) as client:
            with audio_path.open("rb") as audio_file:
                response = await client.post(
                    f"{self.base_url}/transcribe",
                    files={"audio": (audio_path.name, audio_file, "audio/wav")},
                    data={"model": model},
                )
            if response.status_code >= 400:
                raise VoiceboxError(response.text, response.status_code)
            data = response.json()
            return data.get("text") or data.get("transcript") or ""

    async def generate(
        self,
        text: str,
        profile_id: str,
        language: str = "en",
        engine: str | None = None,
    ) -> bytes:
        payload: dict[str, Any] = {
            "text": text,
            "profile_id": profile_id,
            "language": language,
        }
        if engine:
            payload["engine"] = engine

        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(f"{self.base_url}/generate", json=payload)
            if response.status_code >= 400:
                raise VoiceboxError(response.text, response.status_code)

            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                data = response.json()
                audio_url = data.get("audio_url") or data.get("url")
                if audio_url:
                    audio_response = await client.get(audio_url)
                    audio_response.raise_for_status()
                    return audio_response.content
                audio_path = data.get("audio_path") or data.get("path")
                if audio_path:
                    file_response = await client.get(f"{self.base_url}/audio/{audio_path}")
                    file_response.raise_for_status()
                    return file_response.content
                raise VoiceboxError(f"Unexpected generate response: {json.dumps(data)[:500]}")
            return response.content

    async def speak(self, text: str, profile: str) -> bytes:
        payload = {"text": text, "profile": profile}
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{self.base_url}/speak",
                json=payload,
                headers={"X-Voicebox-Client-Id": "ultron"},
            )
            if response.status_code >= 400:
                raise VoiceboxError(response.text, response.status_code)
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                data = response.json()
                audio_url = data.get("audio_url") or data.get("url")
                if audio_url:
                    audio_response = await client.get(audio_url)
                    audio_response.raise_for_status()
                    return audio_response.content
                raise VoiceboxError(f"Unexpected speak response: {json.dumps(data)[:500]}")
            return response.content
