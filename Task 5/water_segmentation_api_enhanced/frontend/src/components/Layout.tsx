import { Link } from "react-router-dom";
import { ApiStatus } from "./ApiStatus";

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-40 border-b border-surface-800 bg-surface-950/90 backdrop-blur supports-[backdrop-filter]:bg-surface-950/70">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          <Link to="/" className="flex items-center gap-2 font-semibold text-surface-50 hover:text-brand-400 transition-colors">
            <span className="text-brand-400">◉</span>
            <span>Water Segmentation</span>
          </Link>
          <nav className="flex items-center gap-4 text-sm text-surface-400">
            <ApiStatus />
            <Link to="/" className="hover:text-surface-100 transition-colors">Gallery</Link>
          </nav>
        </div>
      </header>
      <main className="flex-1 mx-auto w-full max-w-7xl px-4 sm:px-6 lg:px-8 py-6">
        {children}
      </main>
    </div>
  );
}
