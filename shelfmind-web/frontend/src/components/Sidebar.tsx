"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const NAV_ITEMS = [
  { href: "/", icon: "🏠", label: "Dashboard" },
  { href: "/scanner", icon: "📷", label: "Product Scanner" },
  { href: "/planogram", icon: "📋", label: "Planogram Creator" },
  { href: "/monitor", icon: "📡", label: "Live Monitor" },
  { href: "/analytics", icon: "📊", label: "Analytics" },
  { href: "/training", icon: "🧬", label: "Training Results" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* Mobile hamburger */}
      <button
        className="fixed top-4 left-4 z-[60] p-2 rounded-lg lg:hidden"
        style={{ background: "var(--surface-container)" }}
        onClick={() => setOpen(!open)}
        aria-label="Toggle menu"
      >
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          {open ? (
            <path d="M6 6l12 12M6 18L18 6" />
          ) : (
            <path d="M3 6h18M3 12h18M3 18h18" />
          )}
        </svg>
      </button>

      {/* Overlay */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`sidebar ${open ? "open" : ""}`}>
        {/* Brand */}
        <div className="sidebar-brand">
          <span>🧠</span>
          <span className="text-gradient">ShelfMind AI</span>
        </div>

        {/* Nav */}
        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => {
            const isActive =
              item.href === "/"
                ? pathname === "/"
                : pathname.startsWith(item.href);

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`sidebar-link ${isActive ? "active" : ""}`}
                onClick={() => setOpen(false)}
              >
                <span className="sidebar-link-icon">{item.icon}</span>
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Status footer */}
        <div
          style={{
            padding: "16px",
            borderTop: "1px solid var(--glass-border)",
            marginTop: "auto",
          }}
        >
          <div className="flex items-center gap-2 mb-2">
            <span className="pulse-dot pulse-dot-green" />
            <span style={{ fontSize: "0.78rem", color: "var(--primary-bright)", fontWeight: 600 }}>
              Online • {typeof window !== "undefined" && window.location.hostname === "localhost" ? "Local GPU" : "CPU"}
            </span>
          </div>
          <div style={{ fontSize: "0.72rem", color: "var(--on-surface-muted)" }}>
            {process.env.NEXT_PUBLIC_API_URL?.includes("localhost") ? "Local Backend" : "HuggingFace Spaces"}
          </div>
        </div>
      </aside>
    </>
  );
}
