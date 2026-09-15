import React, { useRef } from 'react';
import { Database, UploadCloud, RefreshCw, ShieldCheck, HardDrive } from 'lucide-react';
import { useData } from '../../context/DataContext';

export const TopBar: React.FC = () => {
  const {
    datasets,
    activeDataset,
    selectDataset,
    uploadFile,
    rerunAnalysis,
    status,
    overview,
  } = useData();

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      uploadFile(e.target.files[0]);
    }
  };

  const getHealthBadge = (score?: number, grade?: string) => {
    if (score === undefined) return null;
    let colorClass = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
    if (score < 40) {
      colorClass = 'bg-rose-500/10 text-rose-400 border-rose-500/20';
    } else if (score < 70) {
      colorClass = 'bg-amber-500/10 text-amber-400 border-amber-500/20';
    } else if (score < 85) {
      colorClass = 'bg-blue-500/10 text-blue-400 border-blue-500/20';
    }

    return (
      <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded border text-xs font-mono font-medium ${colorClass}`}>
        <ShieldCheck className="w-3.5 h-3.5" />
        <span>Health: {score.toFixed(1)}</span>
        <span className="opacity-75 uppercase text-[10px]">({grade || 'N/A'})</span>
      </div>
    );
  };

  const formatBytes = (bytes?: number) => {
    if (!bytes) return '0 B';
    if (bytes > 1048576) return `${(bytes / 1048576).toFixed(1)} MB`;
    if (bytes > 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${bytes} B`;
  };

  return (
    <header className="h-13 bg-slate-950/80 backdrop-blur border-b border-slate-800/80 px-4 flex items-center justify-between select-none z-30 shrink-0">
      {/* Brand & Platform Indicator */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center font-black text-white text-xs tracking-tighter shadow-sm">
            DA
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="font-semibold text-slate-100 text-sm tracking-tight">DataAlchemy</span>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700/50">
              Workstation
            </span>
          </div>
        </div>

        {/* Dataset Selector */}
        <div className="flex items-center gap-2">
          <div className="relative flex items-center">
            <Database className="w-3.5 h-3.5 absolute left-2.5 text-slate-400 pointer-events-none" />
            <select
              value={activeDataset || ''}
              onChange={(e) => selectDataset(e.target.value)}
              disabled={status === 'analyzing'}
              className="bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-200 text-xs font-medium pl-8 pr-7 py-1.5 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500 cursor-pointer disabled:opacity-50"
            >
              {datasets.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name} {d.is_sample ? '(sample)' : '(uploaded)'}
                </option>
              ))}
            </select>
          </div>

          {/* Upload Button */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            className="hidden"
            accept=".csv,.xlsx,.xls,.json,.parquet"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={status === 'analyzing'}
            title="Upload CSV, JSON, Excel, or Parquet dataset"
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-slate-300 bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 rounded transition disabled:opacity-50 cursor-pointer"
          >
            <UploadCloud className="w-3.5 h-3.5 text-slate-400" />
            <span>Upload</span>
          </button>

          {/* Rerun Analysis */}
          <button
            onClick={() => rerunAnalysis()}
            disabled={status === 'analyzing' || !activeDataset}
            title="Rerun profiling, quality, statistical, ML, and AI pipelines"
            className="flex items-center gap-1 px-2 py-1.5 text-xs text-slate-400 hover:text-slate-200 bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 rounded transition disabled:opacity-50 cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${status === 'analyzing' ? 'animate-spin text-indigo-400' : ''}`} />
          </button>
        </div>
      </div>

      {/* Dataset Metrics & Health Badges */}
      <div className="flex items-center gap-3">
        {overview && (
          <div className="hidden lg:flex items-center gap-3 text-xs font-mono text-slate-400 border-r border-slate-800 pr-3">
            <span className="flex items-center gap-1">
              <span className="text-slate-200 font-semibold">{overview.dimensions.rows.toLocaleString()}</span> rows
            </span>
            <span className="text-slate-600">•</span>
            <span className="flex items-center gap-1">
              <span className="text-slate-200 font-semibold">{overview.dimensions.columns}</span> cols
            </span>
            <span className="text-slate-600">•</span>
            <span className="flex items-center gap-1">
              <HardDrive className="w-3 h-3 text-slate-500" />
              <span>{formatBytes(overview.dimensions.memory_bytes)}</span>
            </span>
          </div>
        )}

        {overview && getHealthBadge(overview.health.score, overview.health.grade)}

        {/* Engine status indicator */}
        <div className="flex items-center gap-1.5 pl-2">
          <div
            className={`w-2 h-2 rounded-full ${
              status === 'analyzing'
                ? 'bg-amber-400 animate-pulse'
                : status === 'error'
                ? 'bg-rose-500'
                : 'bg-emerald-400'
            }`}
          />
          <span className="text-[11px] font-mono text-slate-400 capitalize">
            {status === 'analyzing' ? 'Computing ML/AI...' : status === 'error' ? 'Degraded' : 'Active'}
          </span>
        </div>
      </div>
    </header>
  );
};
