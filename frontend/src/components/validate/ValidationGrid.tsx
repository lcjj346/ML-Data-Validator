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
    // Index column
    const cols: ColDef[] = [
      {
        headerName: '#',
        valueGetter: (p) => (p.node?.rowIndex ?? 0) + 1,
        width: 50,
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
                return { backgroundColor: '#ff8c00', color: '#ffffff' };
              }
              const valid = cellValidity[key];
              if (valid === true) {
                return { backgroundColor: '#28a745', color: '#ffffff' };
              }
              if (valid === false) {
                return { backgroundColor: '#dc3545', color: '#ffffff' };
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
      minWidth: 100,
      filter: false,
      suppressHeaderMenuButton: true,
    }),
    [],
  );

  return (
    <Collapsible title="Validation Results Table" defaultOpen={true}>
      <div className="ag-theme-alpine-dark" style={{ height: 400, width: '100%' }}>
        <AgGridReact
          ref={gridRef}
          rowData={data}
          columnDefs={columnDefs}
          defaultColDef={defaultColDef}
          onCellValueChanged={onCellValueChanged}
          animateRows={false}
        />
      </div>
      <div className="flex gap-4 mt-2 text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded" style={{ backgroundColor: '#28a745' }} /> Valid
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded" style={{ backgroundColor: '#dc3545' }} /> Invalid
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded" style={{ backgroundColor: '#ff8c00' }} /> Manually edited
        </span>
      </div>
    </Collapsible>
  );
}
