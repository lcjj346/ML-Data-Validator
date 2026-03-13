interface Props {
  validCells: number;
  invalidCells: number;
  quality: number;
}

export default function QualityMetrics({ validCells, invalidCells, quality }: Props) {
  const qualityColor = quality >= 80 ? 'text-green-400' : quality >= 50 ? 'text-yellow-400' : 'text-red-400';
  const qualityBorder = quality >= 80 ? 'border-green-500/20' : quality >= 50 ? 'border-yellow-500/20' : 'border-red-500/20';

  return (
    <div className="grid grid-cols-3 gap-4 mb-6 animate-fadeIn">
      <div className="glass-card border-green-500/20 p-5 text-center hover:bg-white/[0.07] transition-colors">
        <div className="text-slate-400 text-xs uppercase tracking-wider mb-2">Valid Cells</div>
        <div className="text-3xl font-extrabold text-green-400">{validCells}</div>
      </div>
      <div className="glass-card border-red-500/20 p-5 text-center hover:bg-white/[0.07] transition-colors">
        <div className="text-slate-400 text-xs uppercase tracking-wider mb-2">Invalid Cells</div>
        <div className="text-3xl font-extrabold text-red-400">{invalidCells}</div>
      </div>
      <div className={`glass-card ${qualityBorder} p-5 text-center hover:bg-white/[0.07] transition-colors`}>
        <div className="text-slate-400 text-xs uppercase tracking-wider mb-2">Quality</div>
        <div className={`text-3xl font-extrabold ${qualityColor}`}>{quality.toFixed(1)}%</div>
      </div>
    </div>
  );
}
