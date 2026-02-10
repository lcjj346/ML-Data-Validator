interface Props {
  progress: number; // 0-1
  message?: string;
}

export default function ProgressBar({ progress, message }: Props) {
  const pct = Math.round(progress * 100);

  return (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm text-gray-300">{message || 'Processing...'}</span>
        <span className="text-sm font-medium text-indigo-400">{pct}%</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2.5">
        <div
          className="bg-indigo-500 h-2.5 rounded-full transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
