interface Props {
  progress: number; // 0-1
  message?: string;
}

export default function ProgressBar({ progress, message }: Props) {
  const pct = Math.round(progress * 100);

  return (
    <div className="mb-4 glass-card p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-slate-300">{message || 'Processing...'}</span>
        <span className="text-sm font-semibold text-cyan-400">{pct}%</span>
      </div>
      <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
        <div
          className="h-2.5 rounded-full transition-all duration-300 bg-gradient-to-r from-cyan-500 to-sky-500 relative"
          style={{ width: `${pct}%` }}
        >
          <div
            className="absolute inset-0 opacity-30"
            style={{
              backgroundImage: 'linear-gradient(45deg, rgba(255,255,255,0.15) 25%, transparent 25%, transparent 50%, rgba(255,255,255,0.15) 50%, rgba(255,255,255,0.15) 75%, transparent 75%)',
              backgroundSize: '20px 20px',
              animation: 'stripes 0.6s linear infinite',
            }}
          />
        </div>
      </div>
    </div>
  );
}
