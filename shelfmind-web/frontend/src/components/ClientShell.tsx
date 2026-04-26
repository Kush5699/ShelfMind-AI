"use client";

import { StoreProvider } from "@/lib/store";
import Sidebar from "@/components/Sidebar";

export default function ClientShell({ children }: { children: React.ReactNode }) {
  return (
    <StoreProvider>
      <Sidebar />
      <main className="main-content">{children}</main>
    </StoreProvider>
  );
}
