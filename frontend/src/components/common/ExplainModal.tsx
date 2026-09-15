import React, { useEffect } from 'react';
import { Sparkles, X, AlertCircle, ArrowRight, ShieldCheck } from 'lucide-react';
import { useData } from '../../context/DataContext';

export const ExplainModal: React.FC = () => {
  const { explainModal, closeExplainModal } = useData();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeExplainModal();
    };
    if (explainModal.isOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [explainModal.isOpen, closeExplainModal]);

  if (!explainModal.isOpen) return null;

  const req = explainModal.request;
  const res = explainModal.response;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-xs p-4 animate-in fade-in duration-150">
      <div className="bg-slate-900 border border-slate-700/80 rounded-lg shadow-2xl max-w-xl w-full flex flex-col overflow-hidden text-slate-100">
        {/* Header */}
        <div className="px-5 py-3.5 bg-slate-950/60 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-violet-500/20 text-violet-400 border border-violet-500/30 flex items-center justify-center">
              <Sparkles className="w-3.5 h-3.5" />
            </div>
            <h3 className="font-semibold text-sm text-slate-200">AI Deep Dive Explanation</h3>
          </div>
          <button
            onClick={closeExplainModal}
            className="text-slate-400 hover:text-slate-200 p-1 rounded hover:bg-slate-800 transition cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-5 flex flex-col gap-4 text-xs">
          {/* Finding Header */}
          <div className="bg-slate-950/40 p-3 rounded border border-slate-800/80 flex flex-col gap-1.5">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] text-slate-400 uppercase">
                {req?.finding_id || 'FINDING'}
              </span>
              {req?.category && (
                <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 uppercase">
                  {req.category}
                </span>
              )}
            </div>
            <h4 className="font-semibold text-slate-100 text-sm">{req?.title}</h4>
            {req?.context && (
              <p className="text-slate-400 text-xs italic">{req.context}</p>
            )}
          </div>

          {/* Evidence section */}
          {req?.evidence && Object.keys(req.evidence).length > 0 && (
            <div>
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-1.5">
                Verified Statistical Evidence
              </span>
              <div className="bg-slate-950/60 p-2.5 rounded border border-slate-800 font-mono text-[11px] text-slate-300 flex flex-wrap gap-2">
                {Object.entries(req.evidence).map(([key, value]) => (
                  <div key={key} className="bg-slate-900 px-2 py-1 rounded border border-slate-800 flex items-center gap-1.5">
                    <span className="text-slate-500">{key}:</span>
                    <span className="text-indigo-400 font-semibold">
                      {typeof value === 'number' ? value.toLocaleString() : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Loading state */}
          {explainModal.isLoading && (
            <div className="py-6 flex flex-col items-center justify-center gap-2 text-slate-400">
              <Sparkles className="w-5 h-5 text-violet-400 animate-spin" />
              <span className="font-mono text-[11px]">Synthesizing non-causal explanation with local LLM...</span>
            </div>
          )}

          {/* Error state */}
          {explainModal.error && (
            <div className="p-3 rounded bg-rose-500/10 border border-rose-500/20 text-rose-300 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{explainModal.error}</span>
            </div>
          )}

          {/* Response Explanation & Action */}
          {res && !explainModal.isLoading && (
            <div className="flex flex-col gap-3">
              <div>
                <span className="text-[11px] font-semibold text-violet-400 uppercase tracking-wider block mb-1">
                  Synthesized Interpretation
                </span>
                <div className="p-3 rounded bg-violet-950/20 border border-violet-800/30 text-slate-200 leading-relaxed text-xs">
                  {res.explanation}
                </div>
              </div>

              {res.recommendation && (
                <div>
                  <span className="text-[11px] font-semibold text-emerald-400 uppercase tracking-wider block mb-1 flex items-center gap-1">
                    <ShieldCheck className="w-3.5 h-3.5" />
                    <span>Technical Next Step</span>
                  </span>
                  <div className="p-2.5 rounded bg-emerald-950/20 border border-emerald-800/30 text-slate-300 text-xs flex items-start gap-2">
                    <ArrowRight className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                    <span>{res.recommendation}</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-2.5 bg-slate-950/60 border-t border-slate-800 flex justify-end">
          <button
            onClick={closeExplainModal}
            className="px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 rounded transition cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
