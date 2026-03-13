import { useState, useRef, useEffect, type ReactNode } from 'react';

interface Props {
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}

export default function Collapsible({ title, defaultOpen = true, children }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const contentRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | undefined>(defaultOpen ? undefined : 0);

  useEffect(() => {
    if (open) {
      const el = contentRef.current;
      if (el) {
        setHeight(el.scrollHeight);
        const timer = setTimeout(() => setHeight(undefined), 200);
        return () => clearTimeout(timer);
      }
    } else {
      const el = contentRef.current;
      if (el) {
        setHeight(el.scrollHeight);
        requestAnimationFrame(() => setHeight(0));
      }
    }
  }, [open]);

  return (
    <div className="glass-card mb-4 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3.5 text-left font-semibold text-sm hover:bg-white/5 transition-colors"
      >
        <span>{title}</span>
        <svg
          className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      <div
        ref={contentRef}
        className="transition-[height] duration-200 ease-in-out overflow-hidden"
        style={{ height: height !== undefined ? `${height}px` : 'auto' }}
      >
        <div className="px-5 pb-4">{children}</div>
      </div>
    </div>
  );
}
