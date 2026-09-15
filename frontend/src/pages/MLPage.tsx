import React, { useState } from 'react';
import {
  AlertOctagon,
  Boxes,
  Network,
  Sparkles,
} from 'lucide-react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import { useData } from '../context/DataContext';

export const MLPage: React.FC = () => {
  const { ml, openExplainModal } = useData();
  const [mlTab, setMlTab] = useState<'anomalies' | 'clusters' | 'pca'>('anomalies');

  if (!ml) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-500 font-mono text-sm">
        No machine learning analysis report available.
      </div>
    );
  }

  const { anomaly, clustering, dimensionality, visualization_data } = ml;

  // Prepare PCA scatter data
  const pcaScatter = visualization_data?.pca_scatter_2d || [];
  const normalPoints = pcaScatter.filter((p) => !p.is_anomaly);
  const anomalyPoints = pcaScatter.filter((p) => p.is_anomaly);

  // Cluster colors
  const clusterColors = ['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6'];

  // Scree plot data
  const screeData =
    dimensionality.explained_variance_ratio.map((ratio, i) => ({
      component: `PC${i + 1}`,
      ratio: Number((ratio * 100).toFixed(1)),
      cumulative: Number(
        ((dimensionality.cumulative_explained_variance[i] || ratio) * 100).toFixed(1)
      ),
    })) || [];

  // Feature contribution data for anomalies
  const featContribData = Object.entries(anomaly.feature_contributions || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([feat, score]) => ({
      feature: feat,
      score: Number(score.toFixed(3)),
    }));

  return (
    <div className="p-6 flex flex-col gap-6 max-w-7xl mx-auto w-full overflow-y-auto">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <span>Machine Learning Intelligence</span>
            <span className="text-xs font-mono font-normal text-slate-400 px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
              {ml.dataset_name}
            </span>
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Unsupervised anomaly detection, cluster discovery, and principal component analysis.
          </p>
        </div>

        {/* Navigation Sub-tabs */}
        <div className="flex items-center gap-1 bg-slate-900 p-1 rounded border border-slate-800">
          <button
            onClick={() => setMlTab('anomalies')}
            className={`flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded transition cursor-pointer ${
              mlTab === 'anomalies'
                ? 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <AlertOctagon className="w-3.5 h-3.5" />
            <span>Anomalies ({anomaly.n_anomalies})</span>
          </button>
          <button
            onClick={() => setMlTab('clusters')}
            className={`flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded transition cursor-pointer ${
              mlTab === 'clusters'
                ? 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Boxes className="w-3.5 h-3.5" />
            <span>Clustering ({clustering.n_clusters})</span>
          </button>
          <button
            onClick={() => setMlTab('pca')}
            className={`flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded transition cursor-pointer ${
              mlTab === 'pca'
                ? 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Network className="w-3.5 h-3.5" />
            <span>PCA & Projections</span>
          </button>
        </div>
      </div>

      {/* Sub-view: Anomalies */}
      {mlTab === 'anomalies' && (
        <div className="flex flex-col gap-6">
          {/* Top Metric Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-4 flex flex-col">
              <span className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                Outliers Detected
              </span>
              <span className="text-2xl font-bold font-mono text-rose-400 mt-1">
                {anomaly.n_anomalies.toLocaleString()}
              </span>
              <span className="text-xs text-slate-500 mt-0.5">
                {anomaly.anomaly_percentage.toFixed(1)}% of dataset
              </span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-4 flex flex-col">
              <span className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                Algorithm
              </span>
              <span className="text-lg font-bold font-mono text-slate-200 mt-1 uppercase">
                {anomaly.algorithm}
              </span>
              <span className="text-xs text-slate-500 mt-0.5">Unsupervised contamination</span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-4 flex flex-col">
              <span className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                Mean Outlier Score
              </span>
              <span className="text-2xl font-bold font-mono text-indigo-400 mt-1">
                {anomaly.score_summary?.mean?.toFixed(3) ?? '0.000'}
              </span>
              <span className="text-xs text-slate-500 mt-0.5">Decision boundary threshold</span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-4 flex flex-col justify-center">
              <button
                onClick={() =>
                  openExplainModal({
                    finding_id: 'ML_ANOMALIES_SUMMARY',
                    title: `Statistical Anomaly Detection (${anomaly.n_anomalies} outliers identified)`,
                    category: 'anomaly',
                    evidence: {
                      n_anomalies: anomaly.n_anomalies,
                      percentage: `${anomaly.anomaly_percentage.toFixed(1)}%`,
                      algorithm: anomaly.algorithm,
                      mean_score: anomaly.score_summary?.mean,
                    },
                  })
                }
                className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded transition cursor-pointer"
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Explain Outliers with AI</span>
              </button>
            </div>
          </div>

          {/* 2D PCA Anomaly Scatter & Top Contributing Features */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 2D Projection Scatter */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-5 flex flex-col">
              <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 flex items-center justify-between">
                <span>PCA 2D Projection (Normal vs Outlier)</span>
                <div className="flex items-center gap-3 text-[11px] font-mono">
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-slate-500" /> Normal
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-rose-500" /> Anomaly
                  </span>
                </div>
              </h2>

              <div className="h-64 w-full">
                {pcaScatter.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 10, left: -20, bottom: 10 }}>
                      <XAxis
                        dataKey="pc1"
                        name="PC1"
                        stroke="#64748b"
                        fontSize={10}
                        tickLine={false}
                      />
                      <YAxis
                        dataKey="pc2"
                        name="PC2"
                        stroke="#64748b"
                        fontSize={10}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#090d16',
                          borderColor: '#334155',
                          borderRadius: '6px',
                          fontSize: '11px',
                          color: '#f8fafc',
                        }}
                      />
                      <Scatter name="Normal" data={normalPoints} fill="#475569" opacity={0.6} />
                      <Scatter name="Anomaly" data={anomalyPoints} fill="#f43f5e" />
                    </ScatterChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-slate-500 text-xs font-mono">
                    No projection points available.
                  </div>
                )}
              </div>
            </div>

            {/* Feature Contributions */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-5 flex flex-col">
              <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                Top Features Driving Anomaly Scores
              </h2>

              <div className="h-64 w-full">
                {featContribData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={featContribData}
                      layout="vertical"
                      margin={{ top: 10, right: 20, left: 40, bottom: 10 }}
                    >
                      <XAxis type="number" stroke="#64748b" fontSize={10} tickLine={false} />
                      <YAxis
                        type="category"
                        dataKey="feature"
                        stroke="#64748b"
                        fontSize={10}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#090d16',
                          borderColor: '#334155',
                          borderRadius: '6px',
                          fontSize: '11px',
                          color: '#f8fafc',
                        }}
                      />
                      <Bar dataKey="score" fill="#ec4899" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-slate-500 text-xs font-mono">
                    No feature contributions calculated.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Sub-view: Clustering */}
      {mlTab === 'clusters' && (
        <div className="flex flex-col gap-6">
          {/* Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-4 flex flex-col">
              <span className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                Clusters Discovered
              </span>
              <span className="text-2xl font-bold font-mono text-cyan-400 mt-1">
                {clustering.n_clusters}
              </span>
              <span className="text-xs text-slate-500 mt-0.5">Distinct partitions</span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-4 flex flex-col">
              <span className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                Silhouette Score
              </span>
              <span className="text-2xl font-bold font-mono text-emerald-400 mt-1">
                {clustering.silhouette_score !== null ? clustering.silhouette_score.toFixed(3) : 'N/A'}
              </span>
              <span className="text-xs text-slate-500 mt-0.5">Partition quality metric</span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-4 flex flex-col">
              <span className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                Algorithm
              </span>
              <span className="text-lg font-bold font-mono text-slate-200 mt-1 uppercase">
                {clustering.algorithm}
              </span>
              <span className="text-xs text-slate-500 mt-0.5">Unsupervised space partitioning</span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-4 flex flex-col justify-center">
              <button
                onClick={() =>
                  openExplainModal({
                    finding_id: 'ML_CLUSTERING_SUMMARY',
                    title: `Cluster Segmentation Analysis (${clustering.n_clusters} clusters)`,
                    category: 'segment',
                    evidence: {
                      n_clusters: clustering.n_clusters,
                      silhouette_score: clustering.silhouette_score,
                      algorithm: clustering.algorithm,
                    },
                  })
                }
                className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded transition cursor-pointer"
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Explain Clusters with AI</span>
              </button>
            </div>
          </div>

          {/* 2D Projection by Cluster */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-5 flex flex-col">
            <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 flex items-center justify-between">
              <span>PCA 2D Cluster Visualization</span>
              <div className="flex items-center gap-2 text-[11px] font-mono">
                {clustering.cluster_summaries?.map((c, i) => (
                  <span key={c.cluster_id} className="flex items-center gap-1">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: clusterColors[i % clusterColors.length] }}
                    />
                    C{c.cluster_id} ({c.size})
                  </span>
                ))}
              </div>
            </h2>

            <div className="h-64 w-full">
              {pcaScatter.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 10, left: -20, bottom: 10 }}>
                    <XAxis dataKey="pc1" stroke="#64748b" fontSize={10} tickLine={false} />
                    <YAxis dataKey="pc2" stroke="#64748b" fontSize={10} tickLine={false} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#090d16',
                        borderColor: '#334155',
                        borderRadius: '6px',
                        fontSize: '11px',
                        color: '#f8fafc',
                      }}
                    />
                    {clustering.cluster_summaries?.map((c, i) => {
                      const cPoints = pcaScatter.filter((p) => p.cluster === c.cluster_id);
                      return (
                        <Scatter
                          key={c.cluster_id}
                          name={`Cluster ${c.cluster_id}`}
                          data={cPoints}
                          fill={clusterColors[i % clusterColors.length]}
                        />
                      );
                    })}
                  </ScatterChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-full text-slate-500 text-xs font-mono">
                  No cluster scatter points available.
                </div>
              )}
            </div>
          </div>

          {/* Distinguishing Features Table */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-lg overflow-hidden flex flex-col">
            <div className="p-4 bg-slate-950/60 border-b border-slate-800">
              <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Distinguishing Cluster Characteristics
              </h2>
            </div>
            <div className="divide-y divide-slate-800/60 p-4 flex flex-col gap-3">
              {clustering.distinguishing_features?.map((df, i) => (
                <div key={i} className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-semibold text-slate-200">{df.feature}</span>
                    {df.f_stat && (
                      <span className="text-[10px] font-mono text-slate-500">
                        (F-stat: {df.f_stat.toFixed(1)})
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 font-mono">
                    {df.cluster_means &&
                      Object.entries(df.cluster_means).map(([cid, meanVal]) => (
                        <div key={cid} className="bg-slate-950 px-2 py-1 rounded border border-slate-800 text-[11px]">
                          <span className="text-slate-500 mr-1">C{cid}:</span>
                          <span className="text-indigo-300">{Number(meanVal).toFixed(2)}</span>
                        </div>
                      ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Sub-view: PCA */}
      {mlTab === 'pca' && (
        <div className="flex flex-col gap-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Scree Plot */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-5 flex flex-col">
              <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 flex items-center justify-between">
                <span>PCA Scree Plot (Variance Explained)</span>
                <span className="font-mono text-indigo-400 text-xs">
                  {(screeData[screeData.length - 1]?.cumulative || 0)}% Total Var
                </span>
              </h2>

              <div className="h-64 w-full">
                {screeData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={screeData} margin={{ top: 10, right: 10, left: -20, bottom: 10 }}>
                      <XAxis dataKey="component" stroke="#64748b" fontSize={10} tickLine={false} />
                      <YAxis stroke="#64748b" fontSize={10} tickLine={false} unit="%" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#090d16',
                          borderColor: '#334155',
                          borderRadius: '6px',
                          fontSize: '11px',
                          color: '#f8fafc',
                        }}
                      />
                      <Bar dataKey="ratio" fill="#818cf8" radius={[4, 4, 0, 0]} name="Variance Ratio" />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-slate-500 text-xs font-mono">
                    No PCA scree data available.
                  </div>
                )}
              </div>
            </div>

            {/* Loadings Matrix */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-5 flex flex-col">
              <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                Principal Component Loadings (PC1 & PC2)
              </h2>

              <div className="overflow-y-auto max-h-64">
                <table className="w-full text-left text-xs border-collapse font-mono">
                  <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] border-b border-slate-800 sticky top-0">
                    <tr>
                      <th className="py-2 px-3">Feature</th>
                      <th className="py-2 px-3">PC1 Loading</th>
                      <th className="py-2 px-3">PC2 Loading</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-[11px]">
                    {Object.entries(dimensionality.loadings?.PC1 || {}).map(([feat, pc1Val]) => {
                      const pc2Val = dimensionality.loadings?.PC2?.[feat] ?? 0;
                      return (
                        <tr key={feat} className="hover:bg-slate-800/30">
                          <td className="py-1.5 px-3 font-medium text-slate-300 truncate max-w-[140px]">
                            {feat}
                          </td>
                          <td
                            className={`py-1.5 px-3 ${
                              pc1Val > 0 ? 'text-indigo-400' : 'text-rose-400'
                            }`}
                          >
                            {pc1Val.toFixed(3)}
                          </td>
                          <td
                            className={`py-1.5 px-3 ${
                              pc2Val > 0 ? 'text-cyan-400' : 'text-amber-400'
                            }`}
                          >
                            {pc2Val.toFixed(3)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
