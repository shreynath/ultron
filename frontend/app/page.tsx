import Link from "next/link";
import { getHealth } from "@/lib/api";

export default async function DashboardPage() {
  let health;
  try {
    health = await getHealth();
  } catch {
    health = null;
  }

  return (
    <div className="space-y-8">
      <section className="card">
        <p className="text-sm uppercase tracking-[0.25em] text-violet-400">Phase 1</p>
        <h2 className="mt-2 text-3xl font-semibold">Voice & Dictation</h2>
        <p className="mt-3 max-w-2xl text-zinc-400">
          Ultron is your custom voice studio. Clone any voice from YouTube or uploaded media,
          dictate with Whisper, and generate speech in voices you own.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link className="btn-primary" href="/voices">
            Clone a Voice
          </Link>
          <Link className="btn-secondary" href="/studio">
            Open Studio
          </Link>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <StatusCard
          title="Voicebox"
          ok={health?.voicebox.ok ?? false}
          detail={
            health?.voicebox.ok
              ? "Connected"
              : "Not running — download and launch Voicebox"
          }
          link={health?.voicebox.ok ? undefined : "https://voicebox.sh/download/mac-arm"}
        />
        <StatusCard
          title="Media Tools"
          ok={Boolean(health?.tools.ffmpeg && health?.tools.yt_dlp)}
          detail={
            health?.tools.ffmpeg && health?.tools.yt_dlp
              ? "ffmpeg + yt-dlp ready"
              : "Install with: brew install ffmpeg yt-dlp"
          }
        />
        <StatusCard
          title="Ollama"
          ok={health?.ollama_configured ?? false}
          detail={health?.ollama_configured ? "API key configured" : "Add OLLAMA_API_KEY to .env"}
        />
      </section>
    </div>
  );
}

function StatusCard({
  title,
  ok,
  detail,
  link,
}: {
  title: string;
  ok: boolean;
  detail: string;
  link?: string;
}) {
  return (
    <div className="card">
      <div className="flex items-center justify-between">
        <h3 className="font-medium">{title}</h3>
        <span className={ok ? "text-emerald-400" : "text-amber-400"}>
          {ok ? "OK" : "Needs setup"}
        </span>
      </div>
      <p className="mt-2 text-sm text-zinc-400">{detail}</p>
      {link && !ok && (
        <a className="mt-2 inline-block text-sm text-violet-400 underline hover:text-violet-300" href={link}>
          Download Voicebox
        </a>
      )}
    </div>
  );
}
