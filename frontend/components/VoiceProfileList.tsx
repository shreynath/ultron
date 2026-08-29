"use client";

import { useEffect, useState } from "react";
import { deleteVoice, listVoices, type VoiceProfile } from "@/lib/api";

export function VoiceProfileList({ refreshKey }: { refreshKey?: number }) {
  const [profiles, setProfiles] = useState<VoiceProfile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

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
    if (!confirm("Delete this voice profile?")) return;
    try {
      await deleteVoice(profileId);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  return (
    <div className="card">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-xl font-semibold">Saved Voices</h2>
        <button className="btn-secondary" type="button" onClick={load}>
          Refresh
        </button>
      </div>

      {loading && <p className="text-sm text-zinc-400">Loading voices...</p>}
      {error && <p className="text-sm text-red-400">{error}</p>}

      {!loading && profiles.length === 0 && (
        <p className="text-sm text-zinc-400">No voices yet. Clone one above to get started.</p>
      )}

      <div className="space-y-3">
        {profiles.map((profile) => (
          <div
            key={profile.id}
            className="flex items-center justify-between rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-3"
          >
            <div>
              <p className="font-medium">{profile.name}</p>
              <p className="text-xs text-zinc-500">
                {profile.engine || "cloned"} · {profile.language || "en"}
              </p>
            </div>
            <button
              className="btn-secondary text-red-300"
              type="button"
              onClick={() => handleDelete(profile.id)}
            >
              Delete
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
