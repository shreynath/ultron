from __future__ import annotations

import copy
from typing import Any

from app.services.dramatic_planner import PhrasePlan
from app.services.voice_presets import ultron_cinematic_effects_chain, ultron_theater_effects_chain


def _base_effects_chain(voice_style: str) -> list[dict[str, Any]]:
    if voice_style == "ultron_cinematic":
        return ultron_cinematic_effects_chain()
    return ultron_theater_effects_chain()


def effects_chain_for_phrase(voice_style: str, plan: PhrasePlan) -> list[dict[str, Any]]:
    chain = copy.deepcopy(_base_effects_chain(voice_style))

    for effect in chain:
        etype = effect.get("type")
        params = effect.setdefault("params", {})
        if etype == "reverb":
            if voice_style == "ultron_cinematic":
                wet = min(0.42, 0.18 + plan.reverb_wet * 0.35)
                params["wet_level"] = round(wet, 3)
                params["room_size"] = round(min(0.78, 0.55 + plan.intensity * 0.12), 3)
                params["dry_level"] = round(max(0.55, 0.88 - wet), 3)
            else:
                wet = min(0.18, 0.10 + plan.reverb_wet * 0.22)
                params["wet_level"] = round(wet, 3)
                params["room_size"] = round(min(0.55, 0.40 + plan.intensity * 0.10), 3)
                params["dry_level"] = round(max(0.78, 0.92 - wet), 3)
                params["damping"] = 0.68
        elif etype == "chorus":
            if voice_style == "ultron_cinematic":
                params["mix"] = round(min(0.38, max(0.12, plan.chorus_mix)), 3)
                params["depth"] = round(0.45 + plan.intensity * 0.15, 3)
                params["rate_hz"] = 0.18
            else:
                effect["enabled"] = False
        elif etype == "compressor":
            if voice_style == "ultron_cinematic":
                params["ratio"] = 2.2
                params["threshold_db"] = -20.0
            else:
                # Breathing compressor at every period is the "in and out" on long takes.
                effect["enabled"] = False

    return chain
