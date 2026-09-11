"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const navigation = [
  { href: "/", label: "Conversation", icon: "◌" },
  { href: "/brain", label: "Brain", icon: "⌁" },
  { href: "/voices", label: "Voices", icon: "◒" },
  { href: "/studio", label: "Advanced", icon: "◇" },
];

function pageTitle(pathname: string) {
  return navigation.find((item) => item.href === pathname)?.label ?? "Conversation";
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [preferencesOpen, setPreferencesOpen] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [booting, setBooting] = useState(true);

  useEffect(() => {
    const stored = window.sessionStorage.getItem("ultron-shell-booted");
    if (stored) {
      setBooting(false);
      return;
    }
    const timer = window.setTimeout(() => {
      window.sessionStorage.setItem("ultron-shell-booted", "true");
      setBooting(false);
    }, 760);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.motion = reducedMotion ? "reduced" : "full";
  }, [reducedMotion]);

  function trackSpecular(event: React.PointerEvent<HTMLElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    event.currentTarget.style.setProperty("--x", `${event.clientX - rect.left}px`);
    event.currentTarget.style.setProperty("--y", `${event.clientY - rect.top}px`);
  }

  return (
    <div className={`app-shell ${booting ? "booting" : ""}`}>
      <div className="ambient ambient-one" aria-hidden="true" />
      <div className="ambient ambient-two" aria-hidden="true" />
      <header className="top-nav glass-panel" onPointerMove={trackSpecular}>
        <Link className="brand" href="/" aria-label="Ultron conversation home">
          <span>ULTRON</span>
          <strong>{pageTitle(pathname)}</strong>
        </Link>
        <nav className="nav-tabs" aria-label="Primary navigation">
          {navigation.map((item) => {
            const active = pathname === item.href;
            return (
              <Link key={item.href} href={item.href} className={`nav-tab ${active ? "is-active" : ""}`} aria-current={active ? "page" : undefined}>
                <span className="nav-icon" aria-hidden="true">{item.icon}</span>{item.label}
              </Link>
            );
          })}
        </nav>
      </header>

      <aside className="utility-rail glass-panel" aria-label="Workspace utilities">
        <button className="icon-button" type="button" aria-label="Open preferences" title="Preferences" onClick={() => setPreferencesOpen(true)}>⚙</button>
        <span className="rail-divider" />
        <span className="avatar-chip" aria-label="Ultron profile">U</span>
      </aside>

      <main className="app-content">{children}</main>

      {preferencesOpen && (
        <div className="preferences-backdrop" role="presentation" onMouseDown={() => setPreferencesOpen(false)}>
          <aside className="preferences-panel glass-panel" role="dialog" aria-modal="true" aria-labelledby="preferences-title" onMouseDown={(event) => event.stopPropagation()}>
            <div className="panel-heading"><div><p className="technical-label">INTERFACE</p><h2 id="preferences-title">Preferences</h2></div><button className="icon-button" type="button" onClick={() => setPreferencesOpen(false)} aria-label="Close preferences">×</button></div>
            <label className="preference-row"><span><strong>Reduce motion</strong><small>Disable ambient and decorative motion.</small></span><input type="checkbox" checked={reducedMotion} onChange={(event) => setReducedMotion(event.target.checked)} /></label>
            <p className="preferences-note">Voice, model, and delivery controls stay with the feature that uses them.</p>
          </aside>
        </div>
      )}
    </div>
  );
}
