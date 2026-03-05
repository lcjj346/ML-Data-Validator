import { useMemo, useCallback, useRef } from 'react';
import { AgGridReact } from 'ag-grid-react';
import { AllCommunityModule, ModuleRegistry, type CellValueChangedEvent, type ColDef, type CellClassParams } from 'ag-grid-community';
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

  const modifiedSet = useMemo(() => new Set(modifiedCells), [modifiedCells]);

  const columnDefs = useMemo<ColDef[]>(() => {
    const cols: ColDef[] = [
      {
        headerName: '#',
        valueGetter: (p) => (p.node?.rowIndex ?? 0) + 1,
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
          ? (params: CellClassParams) => {
              const key = `${params.node.rowIndex}_${col}`;
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
              return null;
            }
          : undefined,
      });
    }

    return cols;
  }, [columnNames, matchedColumns, cellValidity, modifiedSet]);

  const onCellValueChanged = useCallback(
    (event: CellValueChangedEvent) => {
      const rowIdx = event.node.rowIndex;
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
      <div className="ag-theme-alpine-dark rounded-lg overflow-hidden" style={{ height: 400, width: '100%' }}>
        <AgGridReact
          ref={gridRef}
          rowData={data}
          columnDefs={columnDefs}
          defaultColDef={defaultColDef}
          onCellValueChanged={onCellValueChanged}
          animateRows={false}
        />
      </div>
      <div className="flex gap-5 mt-3 text-xs text-gray-400">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: '#22c55e' }} /> Valid
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: '#ef4444' }} /> Invalid
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: '#f59e0b' }} /> Manually edited
        </span>
      </div>
    </Collapsible>
  );
}
