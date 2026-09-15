import React from 'react';
import { useData } from '../../context/DataContext';
import { TopBar } from './TopBar';
import { Sidebar } from './Sidebar';
import { ExplainModal } from '../common/ExplainModal';
import { OverviewPage } from '../../pages/OverviewPage';
import { DataPage } from '../../pages/DataPage';
import { QualityPage } from '../../pages/QualityPage';
import { StatisticsPage } from '../../pages/StatisticsPage';
import { MLPage } from '../../pages/MLPage';
import { InsightsPage } from '../../pages/InsightsPage';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export const Shell: React.FC = () => {
  const { activeTab, status, errorMessage, rerunAnalysis } = useData();

  const renderActivePage = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewPage />;
      case 'data':
        return <DataPage />;
      case 'quality':
        return <QualityPage />;
      case 'statistics':
        return <StatisticsPage />;
      case 'ml':
        return <MLPage />;
      case 'insights':
        return <InsightsPage />;
      default:
        return <OverviewPage />;
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#090d16] text-slate-100 font-sans">
      {/* Top Header */}
      <TopBar />

      {/* Main Workspace Layout (Sidebar + Center Content) */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar Nav */}
        <Sidebar />

        {/* Dynamic Center Workstation */}
        <main className="flex-1 flex flex-col overflow-hidden bg-slate-950/20 relative">
          {/* Analysis Loading Overlay */}
          {status === 'analyzing' && (
            <div className="absolute inset-0 z-30 bg-slate-950/70 backdrop-blur-xs flex flex-col items-center justify-center gap-3">
              <RefreshCw className="w-7 h-7 text-indigo-400 animate-spin" />
              <div className="flex flex-col items-center">
                <span className="text-sm font-semibold text-slate-200">
                  Autonomous Engine Running
                </span>
                <span className="text-xs font-mono text-slate-400 mt-1">
                  Profiling • Quality Audit • Statistical Discovery • ML Intelligence • AI Synthesis
                </span>
              </div>
            </div>
          )}

          {/* Error Banner */}
          {errorMessage && status === 'error' && (
            <div className="bg-rose-950/40 border-b border-rose-800/60 p-3 px-6 flex items-center justify-between text-xs text-rose-300">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{errorMessage}</span>
              </div>
              <button
                onClick={() => rerunAnalysis()}
                className="px-2.5 py-1 rounded bg-rose-900/60 hover:bg-rose-800/80 text-rose-100 font-medium cursor-pointer"
              >
                Retry
              </button>
            </div>
          )}

          {/* Page Body */}
          <div className="flex-1 overflow-hidden flex flex-col">
            {renderActivePage()}
          </div>
        </main>
      </div>

      {/* On-Demand AI Explain Modal */}
      <ExplainModal />
    </div>
  );
};
