"use client";

import { useEffect, useState } from "react";
import { deleteVoice, listVoices, type VoiceProfile } from "@/lib/api";

export function VoiceProfileList({ refreshKey }: { refreshKey?: number }) {
  const [profiles, setProfiles] = useState<VoiceProfile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [confirmingId, setConfirmingId] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const data = await listVoices();
      setProfiles(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load voices");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, [refreshKey]);

  async function handleDelete(profileId: string) {
    try {
      await deleteVoice(profileId);
      await load();
      setConfirmingId(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  return (
    <div className="card liquid-target">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-xl font-semibold">Saved Voices</h2>
        <button className="icon-button" type="button" onClick={load} aria-label="Refresh saved voices" title="Refresh voices">
          ↻
        </button>
      </div>

      {loading && <p className="text-sm text-[color:var(--text-secondary)]">Loading voices...</p>}
      {error && <p className="error-copy">⚠ {error}</p>}

      {!loading && profiles.length === 0 && (
        <p className="text-sm text-[color:var(--text-secondary)]">No voices yet. Clone one to get started.</p>
      )}

      <div className="space-y-3">
        {profiles.map((profile) => (
          <div
            key={profile.id}
            className="voice-row"
          >
            <div className="flex min-w-0 items-center gap-3">
              <span className="voice-avatar" aria-hidden="true">⌁</span>
              <div>
                <p className="font-medium">{profile.name}</p>
                <div className="voice-meta"><span className="badge">{profile.engine || "cloned"}</span><span className="badge">{profile.language || "en"}</span>{typeof profile.sample_count === "number" && <span className="badge">{profile.sample_count} sample{profile.sample_count === 1 ? "" : "s"}</span>}</div>
                {profile.sample_count === 0 && <p className="mt-1 text-xs text-[color:var(--crimson-300)]">No samples — delete and re-clone this voice</p>}
              </div>
            </div>
            {confirmingId === profile.id ? <div className="delete-confirm"><span>Confirm delete?</span><button className="btn-primary" type="button" onClick={() => handleDelete(profile.id)}>Delete</button><button className="btn-secondary" type="button" onClick={() => setConfirmingId(null)}>Keep</button></div> : <button className="icon-button" type="button" onClick={() => setConfirmingId(profile.id)} aria-label={`Delete ${profile.name}`} title="Delete voice">×</button>}
          </div>
        ))}
      </div>
    </div>
  );
}
