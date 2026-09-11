from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from app.services.engines import normalize_engine
from app.services.voice_engine.direct import direct_generate, direct_transcribe, init_direct_engine
from app.services.voice_engine.manager import get_engine_manager

_tts_generation_lock = asyncio.Lock()


class VoiceboxError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _parse_sse_status(text: str) -> dict[str, Any] | None:
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                return json.loads(line[6:])
            except json.JSONDecodeError:
                continue
    return None


class VoiceboxClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.voicebox_api_url).rstrip("/")

    async def _ensure_engine(self) -> None:
        if settings.voice_engine_mode == "direct":
            await init_direct_engine()
            return
        if settings.voice_engine_mode == "embedded":
            await get_engine_manager().ensure_running()

    async def health(self) -> dict[str, Any]:
        if settings.voice_engine_mode == "direct":
            await init_direct_engine()
            return {"status": "ok", "mode": "direct"}
        await self._ensure_engine()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            data = response.json()
            data["mode"] = settings.voice_engine_mode
            return data

    async def list_profiles(self) -> list[dict[str, Any]]:
        await self._ensure_engine()
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
        language: str = "en",
        description: str = "",
    ) -> dict[str, Any]:
        await self._ensure_engine()
        payload = {
            "name": name,
            "language": language,
            "description": description,
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
        await self._ensure_engine()
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
        await self._ensure_engine()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(f"{self.base_url}/profiles/{profile_id}")
            if response.status_code >= 400:
                raise VoiceboxError(response.text, response.status_code)

    async def update_profile_effects(
        self,
        profile_id: str,
        effects_chain: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        await self._ensure_engine()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.put(
                f"{self.base_url}/profiles/{profile_id}/effects",
                json={"effects_chain": effects_chain},
            )
            if response.status_code >= 400:
                raise VoiceboxError(response.text, response.status_code)
            return response.json()

    async def transcribe(self, audio_path: Path, model: str = "turbo") -> str:
        if settings.voice_engine_mode == "direct":
            return await direct_transcribe(audio_path, model=model)

        await self._ensure_engine()
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                return await self._transcribe_once(audio_path, model=model)
            except VoiceboxError as exc:
                last_error = exc
                message = str(exc).lower()
                if "downloading" in message or "loading_model" in message:
                    await asyncio.sleep(5)
                    continue
                raise
        raise last_error or VoiceboxError("Transcription failed after retries")

    async def _transcribe_once(self, audio_path: Path, model: str = "turbo") -> str:
        async with httpx.AsyncClient(timeout=300.0) as client:
            with audio_path.open("rb") as audio_file:
                response = await client.post(
                    f"{self.base_url}/transcribe",
                    files={"file": (audio_path.name, audio_file, "audio/wav")},
                    data={"model": model},
                )
            if response.status_code >= 400:
                raise VoiceboxError(response.text, response.status_code)
            data = response.json()
            if isinstance(data.get("detail"), dict) and data["detail"].get("downloading"):
                raise VoiceboxError(
                    "Whisper model is still downloading in Voicebox. Wait a minute and try again.",
                    response.status_code,
                )
            return data.get("text") or data.get("transcript") or ""

    async def get_profile_samples(self, profile_id: str) -> list[dict[str, Any]]:
        await self._ensure_engine()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.base_url}/profiles/{profile_id}/samples")
            if response.status_code >= 400:
                raise VoiceboxError(response.text, response.status_code)
            data = response.json()
            if isinstance(data, list):
                return data
            return data.get("samples", [])

    async def _wait_for_generation(self, client: httpx.AsyncClient, generation_id: str) -> None:
        for _ in range(120):
            history = await client.get(f"{self.base_url}/history/{generation_id}")
            if history.status_code == 200:
                data = history.json()
                status = data.get("status", "")
                if status in {"completed", "complete", "done"}:
                    return
                if status in {"failed", "error", "cancelled"}:
                    raise VoiceboxError(data.get("error") or f"Generation failed: {status}")
                if status == "loading_model":
                    await asyncio.sleep(2)
                    continue
            status_response = await client.get(
                f"{self.base_url}/generate/{generation_id}/status",
                timeout=10.0,
            )
            if status_response.status_code == 200:
                body = status_response.text
                if body.strip().startswith("{"):
                    data = status_response.json()
                else:
                    data = _parse_sse_status(body) or {}
                status = data.get("status", "")
                if status in {"completed", "complete", "done"}:
                    return
                if status in {"failed", "error", "cancelled"}:
                    raise VoiceboxError(data.get("error") or f"Generation failed: {status}")
            await asyncio.sleep(2)
        raise VoiceboxError("Speech generation timed out. The TTS model may still be downloading in Voicebox.")

    async def generate(
        self,
        text: str,
        profile_id: str,
        language: str = "en",
        engine: str | None = None,
        instruct: str | None = None,
        effects_chain: list[dict[str, Any]] | None = None,
        normalize: bool = True,
        seed: int | None = None,
    ) -> bytes:
        async with _tts_generation_lock:
            return await self._generate_locked(
                text=text,
                profile_id=profile_id,
                language=language,
                engine=engine,
                instruct=instruct,
                effects_chain=effects_chain,
                normalize=normalize,
                seed=seed,
            )

    async def _generate_locked(
        self,
        text: str,
        profile_id: str,
        language: str = "en",
        engine: str | None = None,
        instruct: str | None = None,
        effects_chain: list[dict[str, Any]] | None = None,
        normalize: bool = True,
        seed: int | None = None,
    ) -> bytes:
        payload: dict[str, Any] = {
            "text": text,
            "profile_id": profile_id,
            "language": language,
            "model_size": "1.7B",
            "normalize": normalize,
        }
        if seed is not None:
            payload["seed"] = seed
        normalized = normalize_engine(engine)
        if normalized:
            payload["engine"] = normalized
        if instruct:
            payload["instruct"] = instruct
        if effects_chain is not None:
            payload["effects_chain"] = effects_chain

        if settings.voice_engine_mode == "direct":
            return await direct_generate(
                text=text,
                profile_id=profile_id,
                language=language,
                engine=normalized,
                instruct=instruct,
                effects_chain=effects_chain,
                normalize=normalize,
                seed=seed,
            )

        await self._ensure_engine()

        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(f"{self.base_url}/generate", json=payload)
            if response.status_code >= 400:
                raise VoiceboxError(response.text, response.status_code)

            content_type = response.headers.get("content-type", "")
            if "application/json" not in content_type:
                return response.content

            data = response.json()
            generation_id = data.get("id")
            if generation_id:
                await self._wait_for_generation(client, generation_id)
                audio_response = await client.get(f"{self.base_url}/audio/{generation_id}")
                if audio_response.status_code >= 400:
                    raise VoiceboxError(audio_response.text, audio_response.status_code)
                return audio_response.content

            audio_url = data.get("audio_url") or data.get("url")
            if audio_url:
                audio_response = await client.get(audio_url)
                audio_response.raise_for_status()
                return audio_response.content

            raise VoiceboxError(f"Unexpected generate response: {json.dumps(data)[:500]}")

    async def speak(self, text: str, profile: str, engine: str | None = None) -> bytes:
        payload: dict[str, Any] = {"text": text, "profile": profile}
        normalized = normalize_engine(engine)
        if normalized:
            payload["engine"] = normalized

        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{self.base_url}/speak",
                json=payload,
                headers={"X-Voicebox-Client-Id": "ultron"},
            )
            if response.status_code >= 400:
                raise VoiceboxError(response.text, response.status_code)

            content_type = response.headers.get("content-type", "")
            if "application/json" not in content_type:
                return response.content

            data = response.json()
            generation_id = data.get("id")
            if generation_id:
                await self._wait_for_generation(client, generation_id)
                audio_response = await client.get(f"{self.base_url}/audio/{generation_id}")
                if audio_response.status_code >= 400:
                    raise VoiceboxError(audio_response.text, audio_response.status_code)
                return audio_response.content

            raise VoiceboxError(f"Unexpected speak response: {json.dumps(data)[:500]}")
