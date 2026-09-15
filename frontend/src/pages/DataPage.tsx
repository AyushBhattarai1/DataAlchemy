import React, { useEffect, useState } from 'react';
import {
  Search,
  ChevronLeft,
  ChevronRight,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Columns,
  Download,
  X,
  Layers,
} from 'lucide-react';
import { apiClient } from '../api/client';
import type { PaginatedDataResponse } from '../types/api';
import { useData } from '../context/DataContext';

export const DataPage: React.FC = () => {
  const { activeDataset } = useData();
  const [data, setData] = useState<PaginatedDataResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [page, setPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(50);
  const [search, setSearch] = useState<string>('');
  const [debouncedSearch, setDebouncedSearch] = useState<string>('');
  const [sortBy, setSortBy] = useState<string | undefined>(undefined);
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  const [visibleColumns, setVisibleColumns] = useState<string[]>([]);
  const [showColPicker, setShowColPicker] = useState<boolean>(false);
  const [selectedRow, setSelectedRow] = useState<Record<string, any> | null>(null);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedSearch(search);
      setPage(1);
    }, 300);
    return () => clearTimeout(handler);
  }, [search]);

  const fetchData = async () => {
    if (!activeDataset) return;
    setLoading(true);
    try {
      const res = await apiClient.getData({
        page,
        pageSize,
        sortBy,
        sortOrder,
        search: debouncedSearch || undefined,
      });
      setData(res);
      if (visibleColumns.length === 0 && res.columns.length > 0) {
        setVisibleColumns(res.columns);
      }
    } catch (err) {
      console.error('Failed to load table records:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [activeDataset, page, pageSize, sortBy, sortOrder, debouncedSearch]);

  const handleSort = (col: string) => {
    if (sortBy === col) {
      if (sortOrder === 'asc') {
        setSortOrder('desc');
      } else {
        setSortBy(undefined);
        setSortOrder('asc');
      }
    } else {
      setSortBy(col);
      setSortOrder('asc');
    }
    setPage(1);
  };

  const toggleColumn = (col: string) => {
    if (visibleColumns.includes(col)) {
      if (visibleColumns.length > 1) {
        setVisibleColumns(visibleColumns.filter((c) => c !== col));
      }
    } else {
      setVisibleColumns([...visibleColumns, col]);
    }
  };

  const exportCSV = () => {
    if (!data || !data.rows.length) return;
    const headers = data.columns.join(',');
    const rows = data.rows.map((r) =>
      data.columns.map((c) => JSON.stringify(r[c] ?? '')).join(',')
    );
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers, ...rows].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `${activeDataset || 'dataset'}_export.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="flex h-full overflow-hidden relative">
      {/* Main Table Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Controls Toolbar */}
        <div className="p-4 bg-slate-950/40 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-3 flex-1 max-w-md">
            <div className="relative flex-1">
              <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search values across records..."
                className="w-full bg-slate-900 border border-slate-800 text-slate-200 text-xs pl-8 pr-3 py-1.5 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500 placeholder:text-slate-500"
              />
              {search && (
                <button
                  onClick={() => setSearch('')}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200"
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>

            {/* Column Picker Dropdown */}
            <div className="relative">
              <button
                onClick={() => setShowColPicker(!showColPicker)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-slate-300 bg-slate-900 border border-slate-800 hover:border-slate-700 rounded transition cursor-pointer"
              >
                <Columns className="w-3.5 h-3.5 text-slate-400" />
                <span>Columns ({visibleColumns.length})</span>
              </button>

              {showColPicker && (
                <div className="absolute left-0 mt-1 w-56 max-h-64 overflow-y-auto bg-slate-900 border border-slate-800 rounded shadow-xl p-2 z-20 flex flex-col gap-1 text-xs">
                  <div className="font-semibold text-slate-400 px-1 py-0.5 text-[11px] uppercase tracking-wider">
                    Toggle Columns
                  </div>
                  {data?.columns.map((col) => (
                    <label
                      key={col}
                      className="flex items-center gap-2 px-1.5 py-1 hover:bg-slate-800 rounded cursor-pointer text-slate-300 select-none"
                    >
                      <input
                        type="checkbox"
                        checked={visibleColumns.includes(col)}
                        onChange={() => toggleColumn(col)}
                        className="rounded border-slate-700 text-indigo-600 focus:ring-indigo-500"
                      />
                      <span className="truncate">{col}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Pagination & Export */}
          <div className="flex items-center gap-3">
            <button
              onClick={exportCSV}
              disabled={!data || data.total_rows === 0}
              className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-slate-300 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded transition disabled:opacity-50 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5 text-slate-400" />
              <span>Export Page</span>
            </button>

            <div className="flex items-center gap-2 text-xs font-mono text-slate-400 border-l border-slate-800 pl-3">
              <select
                value={pageSize}
                onChange={(e) => {
                  setPageSize(Number(e.target.value));
                  setPage(1);
                }}
                className="bg-slate-900 border border-slate-800 text-slate-300 px-2 py-1 rounded cursor-pointer"
              >
                <option value={25}>25 / page</option>
                <option value={50}>50 / page</option>
                <option value={100}>100 / page</option>
              </select>

              <span>
                {data ? `${((page - 1) * pageSize + 1).toLocaleString()}-${Math.min(page * pageSize, data.total_rows).toLocaleString()} of ${data.total_rows.toLocaleString()}` : '0 of 0'}
              </span>

              <div className="flex items-center gap-1">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1 || loading}
                  className="p-1 rounded bg-slate-900 border border-slate-800 hover:border-slate-700 disabled:opacity-40 cursor-pointer"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setPage((p) => Math.min(data?.total_pages || 1, p + 1))}
                  disabled={!data || page >= data.total_pages || loading}
                  className="p-1 rounded bg-slate-900 border border-slate-800 hover:border-slate-700 disabled:opacity-40 cursor-pointer"
                >
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* High-density Scrollable Table */}
        <div className="flex-1 overflow-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead className="bg-slate-950/90 text-slate-400 font-mono uppercase text-[11px] sticky top-0 z-10 border-b border-slate-800 backdrop-blur-xs">
              <tr>
                <th className="py-2.5 px-3 w-12 text-slate-500 font-medium">#</th>
                {visibleColumns.map((col) => {
                  const isSorted = sortBy === col;
                  const dtype = data?.dtypes[col] || '';
                  return (
                    <th
                      key={col}
                      onClick={() => handleSort(col)}
                      className="py-2.5 px-3 font-semibold hover:bg-slate-900 cursor-pointer transition select-none group"
                    >
                      <div className="flex items-center gap-1.5 justify-between">
                        <div className="flex flex-col">
                          <span className="text-slate-200">{col}</span>
                          <span className="text-[9px] font-normal text-slate-500 lowercase">
                            {dtype}
                          </span>
                        </div>
                        <div className="text-slate-500">
                          {isSorted ? (
                            sortOrder === 'asc' ? (
                              <ArrowUp className="w-3 h-3 text-indigo-400" />
                            ) : (
                              <ArrowDown className="w-3 h-3 text-indigo-400" />
                            )
                          ) : (
                            <ArrowUpDown className="w-3 h-3 opacity-0 group-hover:opacity-100 transition" />
                          )}
                        </div>
                      </div>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-[11.5px]">
              {data?.rows.map((row, idx) => {
                const globalRowIdx = (page - 1) * pageSize + idx + 1;
                const isSelected = selectedRow === row;
                return (
                  <tr
                    key={idx}
                    onClick={() => setSelectedRow(row)}
                    className={`hover:bg-indigo-950/20 cursor-pointer transition ${
                      isSelected ? 'bg-indigo-950/40 border-l-2 border-indigo-500' : ''
                    }`}
                  >
                    <td className="py-2 px-3 text-slate-600 font-mono select-none">
                      {globalRowIdx}
                    </td>
                    {visibleColumns.map((col) => {
                      const val = row[col];
                      const isNull = val === null || val === undefined;
                      return (
                        <td key={col} className="py-2 px-3 text-slate-300 truncate max-w-xs">
                          {isNull ? (
                            <span className="text-slate-600 italic font-sans text-[10px]">
                              null
                            </span>
                          ) : (
                            String(val)
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
              {data && data.rows.length === 0 && (
                <tr>
                  <td
                    colSpan={visibleColumns.length + 1}
                    className="py-12 text-center text-slate-500 font-sans text-sm"
                  >
                    No records match the current search filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Row Detail Slide-Out Drawer */}
      {selectedRow && (
        <aside className="w-80 border-l border-slate-800 bg-slate-950/90 flex flex-col h-full shrink-0 shadow-2xl z-20">
          <div className="p-3.5 bg-slate-900 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-400" />
              <span className="font-semibold text-xs text-slate-200">Record Inspection</span>
            </div>
            <button
              onClick={() => setSelectedRow(null)}
              className="text-slate-400 hover:text-slate-200 p-1 rounded hover:bg-slate-800 cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
            {Object.entries(selectedRow).map(([col, val]) => (
              <div key={col} className="bg-slate-900/60 p-2.5 rounded border border-slate-800 flex flex-col gap-1">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="font-mono text-slate-400 font-medium">{col}</span>
                  <span className="text-[10px] text-slate-500 lowercase font-mono">
                    {data?.dtypes[col]}
                  </span>
                </div>
                <div className="text-xs font-mono text-slate-200 break-words">
                  {val === null || val === undefined ? (
                    <span className="text-slate-600 italic">null</span>
                  ) : (
                    String(val)
                  )}
                </div>
              </div>
            ))}
          </div>
        </aside>
      )}
    </div>
  );
};
