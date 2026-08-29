from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_manager: "EmbeddedVoiceEngineManager | None" = None


class EmbeddedVoiceEngineManager:
    """Starts and supervises the vendored Voicebox Python backend."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._started_by_us = False

    @property
    def base_url(self) -> str:
        return settings.voicebox_api_url.rstrip("/")

    async def ensure_running(self) -> None:
        if await self._health_ok():
            return

        if settings.voice_engine_mode == "external":
            raise RuntimeError(
                "Voice engine is not reachable and VOICE_ENGINE_MODE=external. "
                "Start the engine manually or switch to embedded mode."
            )

        if self._process and self._process.poll() is None:
            await self._wait_for_health()
            return

        await asyncio.to_thread(self._spawn)
        await self._wait_for_health()

    async def shutdown(self) -> None:
        if not self._started_by_us or not self._process:
            return
        if self._process.poll() is not None:
            return
        logger.info("Stopping embedded voice engine (pid %s)", self._process.pid)
        self._process.send_signal(signal.SIGTERM)
        try:
            self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None
        self._started_by_us = False

    async def _health_ok(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    async def _wait_for_health(self, timeout_s: float = 120.0) -> None:
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            if await self._health_ok():
                return
            if self._process and self._process.poll() is not None:
                raise RuntimeError("Embedded voice engine exited before becoming healthy")
            await asyncio.sleep(1.0)
        raise TimeoutError("Embedded voice engine did not become healthy in time")

    def _spawn(self) -> None:
        root = Path(__file__).resolve().parents[4]
        script = root / "scripts" / "start-voicebox-engine.sh"
        if not script.exists():
            raise FileNotFoundError(f"Missing engine launcher: {script}")

        env = os.environ.copy()
        env.setdefault("ULTRON_VOICEBOX_HOST", settings.voicebox_host)
        env.setdefault("ULTRON_VOICEBOX_PORT", str(settings.voicebox_port))
        if settings.voicebox_data_dir:
            env["ULTRON_VOICEBOX_DATA_DIR"] = settings.voicebox_data_dir

        logger.info("Launching embedded voice engine via %s", script)
        self._process = subprocess.Popen(
            ["/bin/bash", str(script)],
            cwd=str(root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._started_by_us = True

        async def _drain_output() -> None:
            if not self._process or not self._process.stdout:
                return
            while True:
                line = await asyncio.to_thread(self._process.stdout.readline)
                if not line:
                    break
                logger.debug("[voice-engine] %s", line.rstrip())

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_drain_output())
        except RuntimeError:
            pass


def get_engine_manager() -> EmbeddedVoiceEngineManager:
    global _manager
    if _manager is None:
        _manager = EmbeddedVoiceEngineManager()
    return _manager
