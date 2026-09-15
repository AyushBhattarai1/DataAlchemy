import React, { useState } from 'react';
import {
  BarChart3,
  TrendingUp,
  Sparkles,
  Layers,
  ArrowRight,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useData } from '../context/DataContext';

export const StatisticsPage: React.FC = () => {
  const { statistics, openExplainModal } = useData();
  const [selectedTab, setSelectedTab] = useState<'distributions' | 'correlations' | 'summary'>('distributions');
  const [selectedNumCol, setSelectedNumCol] = useState<string>('');
  const [selectedCatCol, setSelectedCatCol] = useState<string>('');

  if (!statistics) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-500 font-mono text-sm">
        No statistical report available.
      </div>
    );
  }

  const { correlations, distributions, column_profiles } = statistics;

  const numCols = Object.keys(column_profiles).filter(
    (c) => column_profiles[c].semantic_type === 'numerical'
  );
  const catCols = Object.keys(column_profiles).filter(
    (c) => column_profiles[c].semantic_type === 'categorical'
  );

  const activeNum = selectedNumCol || numCols[0] || '';
  const activeCat = selectedCatCol || catCols[0] || '';

  const activeProfile = activeNum ? column_profiles[activeNum] : null;
  const numStats = activeProfile?.type_specific_stats;

  // Build histogram chart data
  const histData =
    numStats?.histogram?.bins && numStats.histogram.counts
      ? numStats.histogram.counts.map((cnt, i) => {
          const binStart = numStats.histogram?.bins[i];
          const binEnd = numStats.histogram?.bins[i + 1];
          return {
            range: `${binStart !== undefined ? Number(binStart).toFixed(1) : ''} - ${binEnd !== undefined ? Number(binEnd).toFixed(1) : ''}`,
            count: cnt,
          };
        })
      : [];

  // Build categorical bar chart data
  const activeCatProfile = activeCat ? column_profiles[activeCat] : null;
  const catData =
    activeCatProfile?.type_specific_stats?.top_categories?.map((cat) => ({
      name: String(cat.value),
      count: cat.count,
      pct: cat.percentage,
    })) || [];

  return (
    <div className="p-6 flex flex-col gap-6 max-w-7xl mx-auto w-full overflow-y-auto">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <span>Statistical Intelligence</span>
            <span className="text-xs font-mono font-normal text-slate-400 px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
              {statistics.dataset_name}
            </span>
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Exploratory distributions, parametric & rank correlations, and concentration patterns.
          </p>
        </div>

        {/* Sub-tabs */}
        <div className="flex items-center gap-1 bg-slate-900 p-1 rounded border border-slate-800">
          <button
            onClick={() => setSelectedTab('distributions')}
            className={`px-3 py-1 text-xs font-medium rounded transition cursor-pointer ${
              selectedTab === 'distributions'
                ? 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Distributions
          </button>
          <button
            onClick={() => setSelectedTab('correlations')}
            className={`px-3 py-1 text-xs font-medium rounded transition cursor-pointer ${
              selectedTab === 'correlations'
                ? 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Correlations ({correlations.length})
          </button>
          <button
            onClick={() => setSelectedTab('summary')}
            className={`px-3 py-1 text-xs font-medium rounded transition cursor-pointer ${
              selectedTab === 'summary'
                ? 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Descriptive Stats
          </button>
        </div>
      </div>

      {/* Tab: Distributions */}
      {selectedTab === 'distributions' && (
        <div className="flex flex-col gap-6">
          {/* Numerical Distribution Section */}
          <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg p-5 flex flex-col gap-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-indigo-400" />
                <h2 className="text-sm font-semibold text-slate-200">
                  Numerical Histogram & Moments
                </h2>
              </div>

              {numCols.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400">Select Feature:</span>
                  <select
                    value={activeNum}
                    onChange={(e) => setSelectedNumCol(e.target.value)}
                    className="bg-slate-950 border border-slate-800 text-slate-200 text-xs px-2.5 py-1 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500 font-mono"
                  >
                    {numCols.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            {/* Metrics pills */}
            {numStats && (
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 font-mono text-xs">
                <div className="bg-slate-950/60 p-2 rounded border border-slate-800 flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase">Mean</span>
                  <span className="text-slate-200 font-semibold">{numStats.mean !== undefined ? numStats.mean.toFixed(2) : 'N/A'}</span>
                </div>
                <div className="bg-slate-950/60 p-2 rounded border border-slate-800 flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase">Median</span>
                  <span className="text-slate-200 font-semibold">{numStats.median !== undefined ? numStats.median.toFixed(2) : 'N/A'}</span>
                </div>
                <div className="bg-slate-950/60 p-2 rounded border border-slate-800 flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase">Std Dev</span>
                  <span className="text-slate-200 font-semibold">{numStats.std !== undefined ? numStats.std.toFixed(2) : 'N/A'}</span>
                </div>
                <div className="bg-slate-950/60 p-2 rounded border border-slate-800 flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase">Min</span>
                  <span className="text-slate-200 font-semibold">{numStats.min !== undefined ? numStats.min.toFixed(2) : 'N/A'}</span>
                </div>
                <div className="bg-slate-950/60 p-2 rounded border border-slate-800 flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase">Max</span>
                  <span className="text-slate-200 font-semibold">{numStats.max !== undefined ? numStats.max.toFixed(2) : 'N/A'}</span>
                </div>
                <div className="bg-slate-950/60 p-2 rounded border border-slate-800 flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase">Skewness</span>
                  <span className="text-indigo-400 font-semibold">{numStats.skewness !== undefined ? numStats.skewness.toFixed(2) : 'N/A'}</span>
                </div>
                <div className="bg-slate-950/60 p-2 rounded border border-slate-800 flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase">Nulls</span>
                  <span className="text-slate-300">{activeProfile?.null_percentage.toFixed(1)}%</span>
                </div>
              </div>
            )}

            {/* Recharts Bar Chart Histogram */}
            <div className="h-64 w-full pt-2">
              {histData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={histData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                    <XAxis
                      dataKey="range"
                      stroke="#64748b"
                      fontSize={10}
                      tickLine={false}
                      angle={-20}
                      textAnchor="end"
                    />
                    <YAxis stroke="#64748b" fontSize={10} tickLine={false} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#090d16',
                        borderColor: '#334155',
                        borderRadius: '6px',
                        fontSize: '11px',
                        color: '#f8fafc',
                      }}
                    />
                    <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-full text-slate-500 text-xs font-mono">
                  No histogram bins available for {activeNum}.
                </div>
              )}
            </div>
          </div>

          {/* Categorical Distribution Section */}
          {catCols.length > 0 && (
            <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg p-5 flex flex-col gap-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  <h2 className="text-sm font-semibold text-slate-200">
                    Categorical Cardinality & Frequencies
                  </h2>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400">Select Category:</span>
                  <select
                    value={activeCat}
                    onChange={(e) => setSelectedCatCol(e.target.value)}
                    className="bg-slate-950 border border-slate-800 text-slate-200 text-xs px-2.5 py-1 rounded focus:outline-none focus:ring-1 focus:ring-cyan-500 font-mono"
                  >
                    {catCols.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Categorical Frequency Bar Chart */}
              <div className="h-56 w-full">
                {catData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={catData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                      <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} />
                      <YAxis stroke="#64748b" fontSize={10} tickLine={false} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#090d16',
                          borderColor: '#334155',
                          borderRadius: '6px',
                          fontSize: '11px',
                          color: '#f8fafc',
                        }}
                      />
                      <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-slate-500 text-xs font-mono">
                    No categorical data available.
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab: Correlations */}
      {selectedTab === 'correlations' && (
        <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg overflow-hidden flex flex-col">
          <div className="p-4 bg-slate-950/60 border-b border-slate-800 flex items-center justify-between">
            <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <TrendingUp className="w-3.5 h-3.5 text-indigo-400" />
              <span>Discovered Correlation Pairs (Filtered |r| ≥ 0.3)</span>
            </h2>
            <span className="text-xs text-slate-500 font-mono">
              Deterministic Pearson & Spearman
            </span>
          </div>

          <div className="divide-y divide-slate-800/60">
            {correlations.map((corr, i) => (
              <div
                key={i}
                className="p-4 hover:bg-slate-900/40 transition flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
              >
                <div className="flex items-center gap-3">
                  <div className="flex flex-col">
                    <div className="flex items-center gap-2 font-mono font-semibold text-slate-200">
                      <span>{corr.column_x}</span>
                      <ArrowRight className="w-3 h-3 text-slate-600" />
                      <span>{corr.column_y}</span>
                    </div>
                    <span className="text-[11px] text-slate-400 mt-0.5">
                      {corr.direction} statistical association
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-3 font-mono">
                    <div className="bg-slate-950 px-2 py-1 rounded border border-slate-800 flex items-center gap-1.5">
                      <span className="text-slate-500 text-[10px]">Pearson:</span>
                      <span
                        className={`font-semibold ${
                          corr.pearson_r > 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {corr.pearson_r.toFixed(3)}
                      </span>
                    </div>

                    <div className="bg-slate-950 px-2 py-1 rounded border border-slate-800 flex items-center gap-1.5">
                      <span className="text-slate-500 text-[10px]">Spearman:</span>
                      <span className="text-slate-300 font-medium">
                        {corr.spearman_rho.toFixed(3)}
                      </span>
                    </div>

                    <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                      {corr.strength}
                    </span>
                  </div>

                  <button
                    onClick={() =>
                      openExplainModal({
                        finding_id: `CORR_${corr.column_x}_${corr.column_y}`,
                        title: `Correlation between ${corr.column_x} and ${corr.column_y}`,
                        category: 'correlation',
                        evidence: {
                          pearson_r: corr.pearson_r,
                          spearman_rho: corr.spearman_rho,
                          strength: corr.strength,
                          direction: corr.direction,
                        },
                      })
                    }
                    className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-indigo-400 hover:text-indigo-300 bg-indigo-950/30 hover:bg-indigo-900/40 border border-indigo-800/50 rounded transition cursor-pointer"
                  >
                    <Sparkles className="w-3 h-3" />
                    <span>Explain</span>
                  </button>
                </div>
              </div>
            ))}

            {correlations.length === 0 && (
              <div className="py-12 text-center text-slate-500 text-xs font-mono">
                No linear correlations above threshold |r| ≥ 0.3 identified.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab: Descriptive Summary Table */}
      {selectedTab === 'summary' && (
        <div className="bg-slate-900/80 border border-slate-800/80 rounded-lg overflow-hidden flex flex-col">
          <div className="p-4 bg-slate-950/60 border-b border-slate-800">
            <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Descriptive Statistics Summary (All Numerical Features)
            </h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead className="bg-slate-950/90 text-slate-400 font-mono uppercase text-[10px] border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Column</th>
                  <th className="py-2.5 px-3">Count</th>
                  <th className="py-2.5 px-3">Mean</th>
                  <th className="py-2.5 px-3">Std</th>
                  <th className="py-2.5 px-3">Min</th>
                  <th className="py-2.5 px-3">Median</th>
                  <th className="py-2.5 px-3">Max</th>
                  <th className="py-2.5 px-3">Skewness</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-[11.5px]">
                {distributions.map((d) => (
                  <tr key={d.column} className="hover:bg-slate-800/30">
                    <td className="py-2 px-3 font-semibold text-slate-200">{d.column}</td>
                    <td className="py-2 px-3 text-slate-400">
                      {column_profiles[d.column]?.total_count || '-'}
                    </td>
                    <td className="py-2 px-3 text-slate-300">{d.mean.toFixed(2)}</td>
                    <td className="py-2 px-3 text-slate-400">{d.std.toFixed(2)}</td>
                    <td className="py-2 px-3 text-slate-400">
                      {column_profiles[d.column]?.type_specific_stats?.min?.toFixed(2) ?? '-'}
                    </td>
                    <td className="py-2 px-3 text-slate-300">{d.median.toFixed(2)}</td>
                    <td className="py-2 px-3 text-slate-400">
                      {column_profiles[d.column]?.type_specific_stats?.max?.toFixed(2) ?? '-'}
                    </td>
                    <td className="py-2 px-3 text-indigo-400 font-medium">
                      {d.skewness.toFixed(2)} ({d.skewness_type})
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
