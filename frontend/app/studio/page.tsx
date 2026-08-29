import { DictationMic, TTSPlayer } from "@/components/StudioPanels";

export default function StudioPage() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-3xl font-semibold">Studio</h2>
        <p className="mt-2 text-zinc-400">
          Dictate with your microphone or generate speech with any cloned voice.
        </p>
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <DictationMic />
        <TTSPlayer />
      </div>
    </div>
  );
}
