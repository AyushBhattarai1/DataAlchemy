import React from 'react';
import {
  ShieldCheck,
  FileSpreadsheet,
  Layers,
  Sparkles,
  Cpu,
  ArrowRight,
  TrendingUp,
  Percent,
} from 'lucide-react';
import { useData } from '../context/DataContext';

export const OverviewPage: React.FC = () => {
  const { overview, setActiveTab, openExplainModal } = useData();

  if (!overview) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-500 font-mono text-sm">
        No dataset loaded or analyzed yet.
      </div>
    );
  }

  const { dimensions, health, missingness, duplicates, ml_summary, executive_summary, key_findings, semantic_type_counts } = overview;

  return (
    <div className="p-6 flex flex-col gap-6 max-w-7xl mx-auto w-full">
      {/* Top Header Summary */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <span>Executive Overview</span>
            <span className="text-xs font-mono font-normal text-slate-400 px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
              {overview.dataset_name}
            </span>
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Holistic data intelligence summary across Ingestion, Profiling, Quality, Statistics, ML, and AI.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveTab('data')}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded transition cursor-pointer"
          >
            <FileSpreadsheet className="w-3.5 h-3.5 text-indigo-400" />
            <span>Browse Records</span>
          </button>
          <button
            onClick={() => setActiveTab('insights')}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-indigo-600 hover:bg-indigo-500 rounded transition cursor-pointer shadow-sm"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>View All Insights</span>
          </button>
        </div>
      </div>

      {/* KPI Workstation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Dimensions */}
        <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Volume & Schema</span>
            <Layers className="w-4 h-4 text-indigo-400" />
          </div>
          <div>
            <div className="text-2xl font-bold font-mono text-slate-100">
              {dimensions.rows.toLocaleString()}
              <span className="text-xs font-sans font-normal text-slate-400 ml-1.5">rows</span>
            </div>
            <div className="text-xs text-slate-400 mt-1 flex items-center gap-2">
              <span className="text-slate-200 font-mono font-medium">{dimensions.columns}</span> columns
              <span className="text-slate-600">•</span>
              <span className="text-slate-400 font-mono">{(dimensions.memory_bytes / 1024).toFixed(1)} KB</span>
            </div>
          </div>
        </div>

        {/* Health Score */}
        <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Data Health Score</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold font-mono text-emerald-400">
                {health.score.toFixed(1)}
              </span>
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                {health.grade}
              </span>
            </div>
            <div className="text-xs text-slate-400 mt-1 flex items-center gap-2">
              <span className="text-rose-400 font-mono font-medium">{health.critical_issues}</span> critical issues
              <span className="text-slate-600">•</span>
              <span className="text-slate-300 font-mono">{health.total_issues}</span> total
            </div>
          </div>
        </div>

        {/* Completeness & Quality */}
        <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Missing & Duplicates</span>
            <Percent className="w-4 h-4 text-amber-400" />
          </div>
          <div>
            <div className="text-2xl font-bold font-mono text-slate-100">
              {missingness.missing_percentage.toFixed(1)}%
              <span className="text-xs font-sans font-normal text-slate-400 ml-1.5">missing</span>
            </div>
            <div className="text-xs text-slate-400 mt-1 flex items-center gap-2">
              <span>{duplicates.duplicate_rows} duplicate rows</span>
              <span className="text-slate-600">•</span>
              <span className="text-slate-300 font-mono">{duplicates.duplicate_percentage.toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* Machine Learning Engine */}
        <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">ML Intelligence</span>
            <Cpu className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <div className="text-2xl font-bold font-mono text-cyan-400">
              {ml_summary.anomalies_found}
              <span className="text-xs font-sans font-normal text-slate-400 ml-1.5">anomalies</span>
            </div>
            <div className="text-xs text-slate-400 mt-1 flex items-center gap-2">
              <span>{ml_summary.clusters_found} clusters</span>
              <span className="text-slate-600">•</span>
              <span className="font-mono text-slate-300">{((ml_summary.pca_variance_explained || 0) * 100).toFixed(0)}% PCA var</span>
            </div>
          </div>
        </div>
      </div>

      {/* AI Synthesized Executive Summary */}
      <div className="bg-gradient-to-r from-slate-900/90 via-slate-900/70 to-indigo-950/20 border border-indigo-900/40 rounded-lg p-5">
        <div className="flex items-center gap-2 text-indigo-400 mb-2">
          <Sparkles className="w-4 h-4" />
          <h2 className="text-xs font-semibold uppercase tracking-wider">AI Executive Synthesis</h2>
        </div>
        <p className="text-sm text-slate-200 leading-relaxed">
          {executive_summary || 'Analysis complete. Verified findings are ready for review.'}
        </p>
      </div>

      {/* Semantic Breakdown & Multi-Stage Key Findings */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Semantic Schema Column Distribution */}
        <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg p-4 flex flex-col">
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
            Semantic Column Types
          </h2>
          <div className="flex flex-col gap-2.5">
            {Object.entries(semantic_type_counts || {}).map(([type, count]) => {
              const total = dimensions.columns || 1;
              const pct = ((count / total) * 100).toFixed(0);
              return (
                <div key={type} className="flex flex-col gap-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300 capitalize">{type}</span>
                    <span className="font-mono text-slate-400">
                      {count} ({pct}%)
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div
                      className="bg-indigo-500 h-full rounded-full"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-auto pt-4 border-t border-slate-800 text-xs flex justify-between items-center text-slate-400">
            <span>Total attributes</span>
            <span className="font-mono font-semibold text-slate-200">{dimensions.columns}</span>
          </div>
        </div>

        {/* Multi-Stage Discoveries */}
        <div className="lg:col-span-2 bg-slate-900/80 border border-slate-800/80 rounded-lg p-4 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <TrendingUp className="w-3.5 h-3.5 text-cyan-400" />
              <span>Key Multi-Stage Findings</span>
            </h2>
            <button
              onClick={() => setActiveTab('statistics')}
              className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 cursor-pointer"
            >
              <span>Explore Stats</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>

          <div className="flex flex-col gap-2.5">
            {key_findings.slice(0, 5).map((f) => (
              <div
                key={f.id}
                className="bg-slate-950/50 p-3 rounded border border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
              >
                <div className="flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    <span className="px-1.5 py-0.2 rounded text-[10px] font-mono uppercase bg-slate-800 text-slate-400 border border-slate-700/50">
                      {f.stage}
                    </span>
                    <span
                      className={`px-1.5 py-0.2 rounded text-[10px] font-mono font-semibold uppercase ${
                        f.importance === 'HIGH'
                          ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                          : f.importance === 'MEDIUM'
                          ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                          : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                      }`}
                    >
                      {f.importance}
                    </span>
                    <span className="font-semibold text-slate-200">{f.title}</span>
                  </div>
                  <p className="text-slate-400 line-clamp-1">{f.description}</p>
                </div>

                <button
                  onClick={() =>
                    openExplainModal({
                      finding_id: f.id,
                      title: f.title,
                      category: f.stage,
                      context: f.description,
                    })
                  }
                  className="shrink-0 flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-indigo-400 hover:text-indigo-300 bg-indigo-950/30 hover:bg-indigo-900/40 border border-indigo-800/50 rounded transition cursor-pointer"
                >
                  <Sparkles className="w-3 h-3" />
                  <span>Explain</span>
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
