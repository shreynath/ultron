import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ultron Voice",
  description: "Dictation and zero-shot voice cloning powered by Voicebox",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen">
          <header className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur">
            <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-violet-400">Ultron</p>
                <h1 className="text-lg font-semibold">Voice Studio</h1>
              </div>
              <nav className="flex gap-4 text-sm text-zinc-300">
                <a className="hover:text-white" href="/">
                  Dashboard
                </a>
                <a className="hover:text-white" href="/voices">
                  Voices
                </a>
                <a className="hover:text-white" href="/studio">
                  Studio
                </a>
              </nav>
            </div>
          </header>
          <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
