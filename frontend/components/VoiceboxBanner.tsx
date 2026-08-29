"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";

export function VoiceboxBanner() {
  const [voiceboxOk, setVoiceboxOk] = useState<boolean | null>(null);
  const [engineMode, setEngineMode] = useState<string>("embedded");

  useEffect(() => {
    getHealth()
      .then((health) => {
        setVoiceboxOk(health.voicebox.ok);
        setEngineMode((health.voicebox as { mode?: string }).mode || "embedded");
      })
      .catch(() => setVoiceboxOk(false));
  }, []);

  if (voiceboxOk === null || voiceboxOk) return null;

  return (
    <div className="rounded-2xl border border-amber-500/40 bg-amber-500/10 px-5 py-4 text-sm text-amber-100">
      <p className="font-medium">Voice engine is not running</p>
      <p className="mt-1 text-amber-200/80">
        {engineMode === "external"
          ? "Ultron is configured for an external voice engine. Start Voicebox or point VOICEBOX_API_URL at a running instance."
          : "Run ./scripts/setup.sh once, then ./scripts/dev.sh — Ultron starts the embedded engine automatically."}
      </p>
      {engineMode === "embedded" && (
        <p className="mt-2 text-amber-200/60">
          First time? The engine downloads ~3.5GB of MLX models on first speech generation.
        </p>
      )}
    </div>
  );
}
