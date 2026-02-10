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
          className="inline-block px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors"
        >
          Download Validated CSV
        </a>
        <div className="bg-gray-800 rounded-lg px-4 py-2 text-center">
          <div className="text-gray-400 text-xs uppercase">Total Rows</div>
          <div className="text-lg font-bold">{totalRows}</div>
        </div>
      </div>
    </Collapsible>
  );
}
