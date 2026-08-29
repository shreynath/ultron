"use client";

import { useState } from "react";
import { VoiceProfileList } from "@/components/VoiceProfileList";
import { VoiceUploader } from "@/components/VoiceUploader";
import { VoiceboxBanner } from "@/components/VoiceboxBanner";

export default function VoicesPage() {
  const [refreshKey, setRefreshKey] = useState(0);

  return (
    <div className="space-y-8">
      <VoiceboxBanner />
      <section>
        <h2 className="text-3xl font-semibold">Voice Profiles</h2>
        <p className="mt-2 text-zinc-400">
          Upload a long YouTube video or audio file. Ultron extracts the best speech clips and
          builds a zero-shot cloned voice.
        </p>
      </section>

      <VoiceUploader onCreated={() => setRefreshKey((value) => value + 1)} />
      <VoiceProfileList refreshKey={refreshKey} />
    </div>
  );
}
