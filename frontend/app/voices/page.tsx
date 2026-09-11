"use client";

import { useState } from "react";
import { VoiceProfileList } from "@/components/VoiceProfileList";
import { VoiceUploader } from "@/components/VoiceUploader";
import { VoiceboxBanner } from "@/components/VoiceboxBanner";

export default function VoicesPage() {
  const [refreshKey, setRefreshKey] = useState(0);

  return (
    <div className="page-stack">
      <VoiceboxBanner />
      <section className="page-heading">
        <h1>Voice Profiles</h1>
        <p>
          Upload a long YouTube video or audio file. Ultron extracts the best speech clips and
          builds a zero-shot cloned voice.
        </p>
      </section>

      <div className="voice-page-grid">
        <VoiceUploader onCreated={() => setRefreshKey((value) => value + 1)} />
        <VoiceProfileList refreshKey={refreshKey} />
      </div>
    </div>
  );
}
