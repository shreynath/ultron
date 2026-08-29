import { DictationMic, TTSPlayer } from "@/components/StudioPanels";
import { TalkMode } from "@/components/TalkMode";
import { VoiceboxBanner } from "@/components/VoiceboxBanner";

export default function StudioPage() {
  return (
    <div className="space-y-8">
      <VoiceboxBanner />

      <section>
        <h2 className="text-3xl font-semibold">Studio</h2>
        <p className="mt-2 text-zinc-400">
          Talk to your cloned voice, dictate, or generate speech.
        </p>
      </section>

      <TalkMode />

      <div className="grid gap-6 lg:grid-cols-2">
        <DictationMic />
        <TTSPlayer />
      </div>
    </div>
  );
}
