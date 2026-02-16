import { getExportUrl } from '../../api/client';
import Collapsible from '../Collapsible';

interface Props {
  sessionId: string;
  totalRows: number;
}

export default function ExportSection({ sessionId, totalRows }: Props) {
  return (
    <Collapsible title="Export Results" defaultOpen={true}>
      <div className="flex items-center gap-6">
        <a
          href={getExportUrl(sessionId)}
          download
          className="inline-flex items-center gap-2 px-6 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white text-sm font-medium rounded-xl transition-all duration-200 shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/30 hover:scale-[1.02] active:scale-[0.98]"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Download Validated CSV
        </a>
        <div className="glass-card px-5 py-2.5 text-center">
          <div className="text-gray-400 text-xs uppercase tracking-wide">Total Rows</div>
          <div className="text-lg font-bold">{totalRows}</div>
        </div>
      </div>
    </Collapsible>
  );
}
