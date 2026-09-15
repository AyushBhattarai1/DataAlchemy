import React, { useState } from 'react';
import {
  Sparkles,
  ShieldCheck,
  AlertTriangle,
  Send,
  HelpCircle,
} from 'lucide-react';
import { useData } from '../context/DataContext';

export const InsightsPage: React.FC = () => {
  const { insights, openExplainModal } = useData();
  const [customQuestion, setCustomQuestion] = useState<string>('');

  if (!insights) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-500 font-mono text-sm">
        No AI insights generated yet.
      </div>
    );
  }

  const { executive_summary, insights: insightList, recommendations, warnings } = insights;

  const handleCustomSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!customQuestion.trim()) return;
    openExplainModal({
      finding_id: 'CUSTOM_AI_QUERY',
      title: customQuestion,
      category: 'ad-hoc',
      context: `Dataset inquiry on ${insights.dataset_name}`,
    });
    setCustomQuestion('');
  };

  return (
    <div className="p-6 flex flex-col gap-6 max-w-7xl mx-auto w-full overflow-y-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <span>AI Insight Engine</span>
            <span className="text-xs font-mono font-normal text-slate-400 px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
              {insights.dataset_name}
            </span>
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Factual synthesis of statistical discoveries, ML patterns, and data quality constraints.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono text-slate-400 bg-slate-900 px-2.5 py-1 rounded border border-slate-800">
          <Sparkles className="w-3.5 h-3.5 text-violet-400" />
          <span>Local Open-Weight Model Grounded</span>
        </div>
      </div>

      {/* Executive Summary Card */}
      <div className="bg-gradient-to-br from-indigo-950/30 via-slate-900 to-slate-900 border border-indigo-500/30 rounded-lg p-5">
        <div className="flex items-center gap-2 text-indigo-400 mb-2">
          <Sparkles className="w-4 h-4" />
          <h2 className="text-xs font-semibold uppercase tracking-wider">
            Synthesized Executive Summary
          </h2>
        </div>
        <p className="text-sm text-slate-200 leading-relaxed font-sans">
          {executive_summary}
        </p>
      </div>

      {/* Warnings & Caveats (if present) */}
      {warnings && warnings.length > 0 && (
        <div className="bg-amber-950/20 border border-amber-500/30 rounded-lg p-4 flex flex-col gap-2">
          <div className="flex items-center gap-2 text-amber-400 text-xs font-semibold uppercase tracking-wider">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Operational Warnings & Caveats</span>
          </div>
          <ul className="list-disc list-inside text-xs text-slate-300 space-y-1">
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Structured Insights List */}
      <div className="flex flex-col gap-4">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center justify-between">
          <span>Structured Findings & Evidence-Backed Explanations</span>
          <span className="font-mono text-[11px] text-slate-500">
            {insightList.length} verified insights
          </span>
        </h2>

        <div className="grid grid-cols-1 gap-4">
          {insightList.map((item) => (
            <div
              key={item.insight_id}
              className="bg-slate-900/80 border border-slate-800/80 rounded-lg p-5 flex flex-col gap-3.5"
            >
              {/* Card Header */}
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-mono text-[10px] text-slate-400 uppercase bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">
                    {item.insight_id}
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                    {item.category}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold uppercase ${
                      item.importance === 'HIGH'
                        ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                        : item.importance === 'MEDIUM'
                        ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                        : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                    }`}
                  >
                    {item.importance}
                  </span>
                </div>

                <button
                  onClick={() =>
                    openExplainModal({
                      finding_id: item.insight_id,
                      title: item.title,
                      category: item.category,
                      context: item.explanation,
                      evidence: item.evidence,
                    })
                  }
                  className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-violet-400 hover:text-violet-300 bg-violet-950/30 hover:bg-violet-900/40 border border-violet-800/50 rounded transition cursor-pointer"
                >
                  <Sparkles className="w-3 h-3" />
                  <span>Deep Dive</span>
                </button>
              </div>

              {/* Title & Summary */}
              <div>
                <h3 className="text-base font-semibold text-slate-100">{item.title}</h3>
                <p className="text-xs text-slate-400 mt-1">{item.summary}</p>
              </div>

              {/* Dual Section: Evidence vs Explanation */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 text-xs">
                {/* Evidence */}
                <div className="bg-slate-950/70 p-3 rounded border border-slate-800/80 flex flex-col gap-1.5">
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                    <span>Verified Factual Evidence</span>
                  </span>
                  <div className="font-mono text-[11px] text-slate-300 flex flex-wrap gap-2 mt-1">
                    {Object.entries(item.evidence || {}).map(([k, v]) => (
                      <div key={k} className="bg-slate-900 px-2 py-1 rounded border border-slate-800">
                        <span className="text-slate-500 mr-1">{k}:</span>
                        <span className="text-indigo-400 font-medium">
                          {typeof v === 'number' ? v.toLocaleString() : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Narrative Explanation */}
                <div className="bg-slate-950/70 p-3 rounded border border-slate-800/80 flex flex-col gap-1.5">
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                    <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                    <span>Synthesized Explanation</span>
                  </span>
                  <p className="text-slate-300 text-xs leading-relaxed">
                    {item.explanation}
                  </p>
                </div>
              </div>

              {/* Caveats if any */}
              {item.caveats && (
                <div className="bg-amber-950/10 p-2.5 rounded border border-amber-900/30 text-[11.5px] text-amber-300 flex items-center gap-2">
                  <AlertTriangle className="w-3.5 h-3.5 shrink-0 text-amber-400" />
                  <span>{item.caveats}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Actionable Recommendations Section */}
      <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg p-5 flex flex-col gap-4">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>Recommended Technical Actions</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {recommendations.map((rec) => (
            <div
              key={rec.recommendation_id}
              className="bg-slate-950/60 p-4 rounded border border-slate-800 flex flex-col justify-between gap-3"
            >
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono text-slate-400 uppercase">
                    {rec.action_type}
                  </span>
                  <span
                    className={`px-1.5 py-0.2 rounded text-[10px] font-mono font-semibold uppercase ${
                      rec.priority === 'high'
                        ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                        : rec.priority === 'medium'
                        ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                        : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                    }`}
                  >
                    {rec.priority}
                  </span>
                </div>
                <h3 className="font-semibold text-slate-100 text-xs">{rec.title}</h3>
                <p className="text-xs text-slate-400 leading-relaxed">{rec.description}</p>
              </div>

              {rec.supporting_evidence && rec.supporting_evidence.length > 0 && (
                <div className="pt-2 border-t border-slate-900 flex items-center gap-1 text-[10px] font-mono text-slate-500 truncate">
                  <span>Linked:</span>
                  <span>{rec.supporting_evidence.join(', ')}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* On-Demand AI Query Box */}
      <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg p-5 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <HelpCircle className="w-4 h-4 text-indigo-400" />
          <h2 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
            Ask AI About This Dataset
          </h2>
        </div>
        <p className="text-xs text-slate-400">
          Query specific statistical or algorithmic findings with local LLM assistance.
        </p>

        <form onSubmit={handleCustomSubmit} className="flex items-center gap-2 mt-1">
          <input
            type="text"
            value={customQuestion}
            onChange={(e) => setCustomQuestion(e.target.value)}
            placeholder="e.g. Why are certain transactions flagged as high-scoring anomalies?"
            className="flex-1 bg-slate-950 border border-slate-800 text-slate-200 text-xs px-3 py-2 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500 placeholder:text-slate-500"
          />
          <button
            type="submit"
            disabled={!customQuestion.trim()}
            className="flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded transition disabled:opacity-40 cursor-pointer"
          >
            <Send className="w-3.5 h-3.5" />
            <span>Ask</span>
          </button>
        </form>
      </div>
    </div>
  );
};
