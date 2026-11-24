"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const STORAGE_KEY = "display_name";

export default function HeaderClient() {
  const [displayName, setDisplayName] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      setDisplayName(stored);
    } else {
      setModalOpen(true);
    }
  }, []);

  function openModal() {
    setInputValue(displayName ?? "");
    setModalOpen(true);
  }

  function handleSave() {
    const name = inputValue.trim();
    if (!name) return;
    localStorage.setItem(STORAGE_KEY, name);
    setDisplayName(name);
    setModalOpen(false);
  }

  return (
    <>
      <header className="bg-primary text-primary-foreground shadow-sm">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-14 items-center justify-between">
            <Link href="/" className="flex items-center gap-3">
              <span className="font-heading text-xl tracking-widest uppercase leading-none">
                Minnesota State Moorhead
              </span>
              <span className="hidden sm:block h-4 w-px bg-primary-foreground/30" />
              <span className="hidden sm:block text-primary-foreground/80 text-sm font-normal">
                Focus Group Simulator
              </span>
            </Link>
            <nav className="flex items-center gap-5 text-sm">
              {displayName && (
                <button
                  onClick={openModal}
                  className="text-primary-foreground/60 text-sm hidden sm:block hover:text-primary-foreground/90 transition-colors"
                  title="Click to edit your name"
                >
                  {displayName}
                </button>
              )}
              <Link
                href="/"
                className="text-primary-foreground/85 hover:text-primary-foreground transition-colors"
              >
                Dashboard
              </Link>
              <form method="post" action="/api/v1/auth/logout">
                <button
                  type="submit"
                  className="cursor-pointer text-primary-foreground/85 hover:text-primary-foreground transition-colors"
                >
                  Sign out
                </button>
              </form>
            </nav>
          </div>
        </div>
      </header>

      <Dialog
        open={modalOpen}
        onOpenChange={(open) => { if (!open && displayName) setModalOpen(false); }}
      >
        <DialogContent
          className="sm:max-w-sm"
          onPointerDownOutside={(e) => { if (!displayName) e.preventDefault(); }}
        >
          <DialogHeader>
            <DialogTitle>What should we call you?</DialogTitle>
          </DialogHeader>
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSave()}
            placeholder="Your name"
            autoFocus
          />
          <DialogFooter>
            <Button onClick={handleSave} disabled={!inputValue.trim()}>
              Continue
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
