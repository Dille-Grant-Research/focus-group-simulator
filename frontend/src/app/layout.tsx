import type { Metadata } from "next";
import { Bebas_Neue, Source_Sans_3 } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const bebasNeue = Bebas_Neue({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-bebas-neue",
});

const sourceSans = Source_Sans_3({
  weight: ["300", "400", "600", "700"],
  subsets: ["latin"],
  variable: "--font-source-sans",
});

export const metadata: Metadata = {
  title: "MSUM Focus Group Simulator",
  description: "Dille-Grant Research Initiative — AI Focus Group Simulator",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${bebasNeue.variable} ${sourceSans.variable} h-full`}
    >
      <body className="font-sans min-h-full flex flex-col bg-background text-foreground antialiased">
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
                <Link
                  href="/"
                  className="text-primary-foreground/85 hover:text-primary-foreground transition-colors"
                >
                  Dashboard
                </Link>
                <a
                  href="/api/v1/auth/logout"
                  className="text-primary-foreground/85 hover:text-primary-foreground transition-colors"
                >
                  Sign out
                </a>
              </nav>
            </div>
          </div>
        </header>

        <main className="flex-1">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
            {children}
          </div>
        </main>

        <footer className="border-t border-border py-4 mt-auto">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 text-center text-xs text-muted-foreground">
            Dille-Grant Research Initiative · Minnesota State University, Moorhead · Internal Use Only
          </div>
        </footer>
      </body>
    </html>
  );
}
