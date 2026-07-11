import { useMemo, useCallback, useRef, useState } from 'react';
import { AgGridReact } from 'ag-grid-react';
import { AllCommunityModule, ModuleRegistry, type CellValueChangedEvent, type ColDef, type CellClassParams, type CellStyle } from 'ag-grid-community';
import Collapsible from '../Collapsible';

ModuleRegistry.registerModules([AllCommunityModule]);

interface Props {
  data: Record<string, unknown>[];
  columnNames: string[];
  matchedColumns: string[];
  cellValidity: Record<string, boolean>;
  modifiedCells: string[];
  onCellEdit: (row: number, column: string, value: unknown) => void;
}

export default function ValidationGrid({
  data,
  columnNames,
  matchedColumns,
  cellValidity,
  modifiedCells,
  onCellEdit,
}: Props) {
  const gridRef = useRef<AgGridReact>(null);
  const [showInvalidOnly, setShowInvalidOnly] = useState(false);

  const modifiedSet = useMemo(() => new Set(modifiedCells), [modifiedCells]);

  // Keep each row's original index so validity keys and edits still line up
  // when the grid is filtered to invalid rows only
  const rows = useMemo(
    () => data.map((d, i) => ({ ...d, __idx: i })),
    [data],
  );

  const invalidRowCount = useMemo(
    () => rows.filter((r) => columnNames.some((col) => cellValidity[`${r.__idx}_${col}`] === false)).length,
    [rows, columnNames, cellValidity],
  );

  const displayedRows = useMemo(() => {
    if (!showInvalidOnly) return rows;
    return rows.filter((r) =>
      columnNames.some((col) => cellValidity[`${r.__idx}_${col}`] === false),
    );
  }, [rows, showInvalidOnly, columnNames, cellValidity]);

  const columnDefs = useMemo<ColDef[]>(() => {
    const cols: ColDef[] = [
      {
        headerName: '#',
        valueGetter: (p) => (p.data?.__idx ?? 0) + 1,
        width: 50,
        minWidth: 40,
        maxWidth: 60,
        editable: false,
        sortable: false,
        cellStyle: { textAlign: 'center', color: '#9ca3af' },
      },
    ];

    for (const col of columnNames) {
      const isMatched = matchedColumns.includes(col);

      cols.push({
        headerName: col,
        field: col,
        editable: true,
        sortable: false,
        cellStyle: isMatched
          ? (params: CellClassParams): CellStyle => {
              const key = `${params.data?.__idx ?? params.node.rowIndex}_${col}`;
              if (modifiedSet.has(key)) {
                return { borderLeft: '3px solid #f59e0b', backgroundColor: 'rgba(245,158,11,0.15)', color: '#fbbf24' };
              }
              const valid = cellValidity[key];
              if (valid === true) {
                return { borderLeft: '3px solid #22c55e', backgroundColor: 'rgba(34,197,94,0.1)', color: '#e5e7eb' };
              }
              if (valid === false) {
                return { borderLeft: '3px solid #ef4444', backgroundColor: 'rgba(239,68,68,0.1)', color: '#e5e7eb' };
              }
              // Matched column but no validity result yet - keep text readable
              return { color: '#cbd5e1' };
            }
          : // Unmatched (not validated) column - readable muted text instead of
            // the theme default, which is nearly invisible on the dark background
            { color: '#cbd5e1' },
      });
    }

    return cols;
  }, [columnNames, matchedColumns, cellValidity, modifiedSet]);

  const onCellValueChanged = useCallback(
    (event: CellValueChangedEvent) => {
      const rowIdx = event.data?.__idx ?? event.node.rowIndex;
      const col = event.colDef.field;
      if (rowIdx != null && col) {
        onCellEdit(rowIdx, col, event.newValue);
      }
    },
    [onCellEdit],
  );

  const defaultColDef = useMemo(
    () => ({
      flex: 1,
      minWidth: 180,
      filter: false,
      suppressHeaderMenuButton: true,
    }),
    [],
  );

  return (
    <Collapsible title="Validation Results Table" defaultOpen={true}>
      <div className="flex items-center justify-end mb-2">
        <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showInvalidOnly}
            onChange={(e) => setShowInvalidOnly(e.target.checked)}
            className="accent-red-500 w-3.5 h-3.5"
          />
          Show invalid rows only ({invalidRowCount})
        </label>
      </div>
      <div className="ag-theme-alpine-dark rounded-lg overflow-hidden" style={{ height: 400, width: '100%' }}>
        <AgGridReact
          ref={gridRef}
          rowData={displayedRows}
          columnDefs={columnDefs}
          defaultColDef={defaultColDef}
          onCellValueChanged={onCellValueChanged}
          animateRows={false}
        />
      </div>
      <div className="flex gap-5 mt-3 text-xs text-slate-400">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: '#22c55e' }} /> Valid
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: '#ef4444' }} /> Invalid
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: '#f59e0b' }} /> Manually edited
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: '#64748b' }} /> Not validated (column not in model)
        </span>
      </div>
    </Collapsible>
  );
}
