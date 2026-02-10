interface Props {
  validCells: number;
  invalidCells: number;
  quality: number;
}

export default function QualityMetrics({ validCells, invalidCells, quality }: Props) {
  const qualityColor = quality >= 80 ? 'text-green-400' : quality >= 50 ? 'text-yellow-400' : 'text-red-400';

  return (
    <div className="grid grid-cols-3 gap-4 mb-4">
      <div className="bg-gray-800 rounded-lg p-4 text-center">
        <div className="text-gray-400 text-xs uppercase tracking-wide mb-1">Valid Cells</div>
        <div className="text-2xl font-bold text-green-400">{validCells}</div>
      </div>
      <div className="bg-gray-800 rounded-lg p-4 text-center">
        <div className="text-gray-400 text-xs uppercase tracking-wide mb-1">Invalid Cells</div>
        <div className="text-2xl font-bold text-red-400">{invalidCells}</div>
      </div>
      <div className="bg-gray-800 rounded-lg p-4 text-center">
        <div className="text-gray-400 text-xs uppercase tracking-wide mb-1">Quality</div>
        <div className={`text-2xl font-bold ${qualityColor}`}>{quality.toFixed(1)}%</div>
      </div>
    </div>
  );
}
