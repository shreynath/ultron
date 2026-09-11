import { DictationMic, TTSPlayer } from "@/components/StudioPanels";
import { VoiceboxBanner } from "@/components/VoiceboxBanner";

export default function StudioPage() {
  return (
    <div className="page-stack">
      <VoiceboxBanner />

      <section className="page-heading">
        <h1>Studio</h1>
        <p>
          Advanced voice tools. Everyday conversation, dictation, and automatic voice delivery live on the Conversation page.
        </p>
      </section>

      <div className="studio-grid">
        <DictationMic />
        <TTSPlayer />
      </div>
    </div>
  );
}
