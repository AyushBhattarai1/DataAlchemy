import React, { useState } from 'react';
import {
  ShieldCheck,
  AlertOctagon,
  AlertTriangle,
  Info,
  Sparkles,
  Filter,
  CheckCircle2,
} from 'lucide-react';
import { useData } from '../context/DataContext';
import type { QualityIssue } from '../types/api';

export const QualityPage: React.FC = () => {
  const { quality, openExplainModal } = useData();
  const [severityFilter, setSeverityFilter] = useState<string>('ALL');

  if (!quality) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-500 font-mono text-sm">
        No data quality report available.
      </div>
    );
  }

  const { overall_score, grade, total_issues, critical_issues, warning_issues, info_issues, issues, column_scores } = quality;

  const filteredIssues = issues.filter((iss) => {
    if (severityFilter === 'ALL') return true;
    return iss.severity === severityFilter;
  });

  const getSeverityBadge = (severity: QualityIssue['severity']) => {
    switch (severity) {
      case 'CRITICAL':
        return (
          <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <AlertOctagon className="w-3 h-3" />
            <span>CRITICAL</span>
          </span>
        );
      case 'ERROR':
        return (
          <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-red-500/10 text-red-400 border border-red-500/20">
            <AlertTriangle className="w-3 h-3" />
            <span>ERROR</span>
          </span>
        );
      case 'WARNING':
        return (
          <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <AlertTriangle className="w-3 h-3" />
            <span>WARNING</span>
          </span>
        );
      case 'INFO':
        return (
          <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <Info className="w-3 h-3" />
            <span>INFO</span>
          </span>
        );
    }
  };

  return (
    <div className="p-6 flex flex-col gap-6 max-w-7xl mx-auto w-full overflow-y-auto">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <span>Data Health & Quality Audit</span>
            <span className="text-xs font-mono font-normal text-slate-400 px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
              {quality.dataset_name}
            </span>
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Deterministic rule-based quality evaluation across 7 audit dimensions.
          </p>
        </div>
      </div>

      {/* Scorecards */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        {/* Overall Score Card */}
        <div className="md:col-span-2 bg-slate-900/80 border border-slate-800/80 rounded-lg p-5 flex items-center justify-between">
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Overall Health Score
            </span>
            <div className="flex items-baseline gap-3 mt-1">
              <span className="text-4xl font-extrabold font-mono text-emerald-400">
                {overall_score.toFixed(1)}
              </span>
              <span className="text-slate-500 font-mono text-xs">/ 100</span>
            </div>
            <div className="flex items-center gap-2 mt-2">
              <span className="px-2.5 py-0.5 rounded text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 uppercase">
                {grade}
              </span>
              <span className="text-xs text-slate-400 font-mono">
                {total_issues === 0 ? 'No defects identified' : `${total_issues} issues detected`}
              </span>
            </div>
          </div>

          <div className="hidden sm:flex items-center justify-center w-20 h-20 rounded-full border-4 border-emerald-500/30 bg-emerald-500/5">
            <ShieldCheck className="w-10 h-10 text-emerald-400" />
          </div>
        </div>

        {/* Issue Counts */}
        <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-medium uppercase tracking-wider">Critical</span>
            <AlertOctagon className="w-4 h-4 text-rose-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-rose-400">
            {critical_issues}
          </div>
          <span className="text-[11px] text-slate-500">Requires immediate fix</span>
        </div>

        <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-medium uppercase tracking-wider">Warnings</span>
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-amber-400">
            {warning_issues}
          </div>
          <span className="text-[11px] text-slate-500">Suboptimal distributions</span>
        </div>

        <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-medium uppercase tracking-wider">Informational</span>
            <Info className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-blue-400">
            {info_issues}
          </div>
          <span className="text-[11px] text-slate-500">Identifiers & structural info</span>
        </div>
      </div>

      {/* Column Quality Scores Breakdown */}
      <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg p-4 flex flex-col gap-3">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center justify-between">
          <span>Attribute-Level Health Scores</span>
          <span className="font-mono text-[11px] text-slate-500">Scale 0 - 100</span>
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {Object.entries(column_scores).map(([col, score]) => {
            const isCritical = score < 50;
            const isWarning = score < 80;
            return (
              <div key={col} className="bg-slate-950/60 p-2.5 rounded border border-slate-800 flex flex-col gap-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-mono text-slate-200 truncate max-w-[130px]" title={col}>
                    {col}
                  </span>
                  <span
                    className={`font-mono font-semibold ${
                      isCritical
                        ? 'text-rose-400'
                        : isWarning
                        ? 'text-amber-400'
                        : 'text-emerald-400'
                    }`}
                  >
                    {score.toFixed(1)}
                  </span>
                </div>
                <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      isCritical
                        ? 'bg-rose-500'
                        : isWarning
                        ? 'bg-amber-500'
                        : 'bg-emerald-500'
                    }`}
                    style={{ width: `${Math.max(5, score)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Quality Issues List */}
      <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg overflow-hidden flex flex-col">
        {/* Table Header / Filters */}
        <div className="p-4 bg-slate-950/60 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Identified Quality Issues ({filteredIssues.length})
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            {['ALL', 'CRITICAL', 'ERROR', 'WARNING', 'INFO'].map((sev) => (
              <button
                key={sev}
                onClick={() => setSeverityFilter(sev)}
                className={`px-2.5 py-1 rounded text-xs font-mono font-medium transition cursor-pointer ${
                  severityFilter === sev
                    ? 'bg-slate-800 text-indigo-300 border border-slate-700'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
                }`}
              >
                {sev}
              </button>
            ))}
          </div>
        </div>

        {/* Issues List */}
        <div className="divide-y divide-slate-800/60">
          {filteredIssues.map((iss, i) => (
            <div
              key={i}
              className="p-4 hover:bg-slate-900/40 transition flex flex-col sm:flex-row sm:items-start justify-between gap-4 text-xs"
            >
              <div className="flex flex-col gap-1.5 flex-1">
                <div className="flex items-center gap-2.5 flex-wrap">
                  {getSeverityBadge(iss.severity)}
                  <span className="font-mono text-[10px] text-slate-400 bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">
                    {iss.rule_id}
                  </span>
                  {iss.column && (
                    <span className="font-mono font-semibold text-slate-200">
                      Column: {iss.column}
                    </span>
                  )}
                </div>

                <h3 className="font-semibold text-slate-200 text-sm">{iss.title}</h3>
                <p className="text-slate-400 leading-relaxed">{iss.description}</p>

                {iss.impact && (
                  <div className="text-slate-400 flex items-center gap-1.5 mt-0.5">
                    <span className="text-slate-400 font-medium">Impact:</span>
                    <span>{iss.impact}</span>
                  </div>
                )}
              </div>

              <div className="shrink-0 flex sm:flex-col items-end justify-between gap-2">
                <button
                  onClick={() =>
                    openExplainModal({
                      finding_id: iss.rule_id,
                      title: iss.title,
                      category: 'quality',
                      context: iss.description,
                      evidence: {
                        column: iss.column,
                        measured: iss.measured_value,
                        threshold: iss.threshold,
                      },
                    })
                  }
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-indigo-400 hover:text-indigo-300 bg-indigo-950/30 hover:bg-indigo-900/40 border border-indigo-800/50 rounded transition cursor-pointer"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Explain Defect</span>
                </button>
              </div>
            </div>
          ))}

          {filteredIssues.length === 0 && (
            <div className="py-12 flex flex-col items-center justify-center gap-2 text-slate-500">
              <CheckCircle2 className="w-6 h-6 text-emerald-500" />
              <span className="text-xs font-mono">No issues found for filter "{severityFilter}"</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
