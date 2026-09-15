import React from 'react';
import {
  LayoutDashboard,
  Table,
  ShieldCheck,
  BarChart3,
  Cpu,
  Sparkles,
  Layers,
} from 'lucide-react';
import { useData, type TabType } from '../../context/DataContext';

interface NavItem {
  id: TabType;
  label: string;
  icon: React.ElementType;
  badge?: string | number | null;
  badgeColor?: string;
}

export const Sidebar: React.FC = () => {
  const { activeTab, setActiveTab, quality, ml, insights } = useData();

  const navItems: NavItem[] = [
    {
      id: 'overview',
      label: 'Overview',
      icon: LayoutDashboard,
    },
    {
      id: 'data',
      label: 'Data Explorer',
      icon: Table,
    },
    {
      id: 'quality',
      label: 'Quality Health',
      icon: ShieldCheck,
      badge: quality?.total_issues ? quality.total_issues : null,
      badgeColor: quality?.critical_issues
        ? 'bg-rose-500/20 text-rose-300 border-rose-500/30'
        : 'bg-slate-800 text-slate-300 border-slate-700/50',
    },
    {
      id: 'statistics',
      label: 'Statistical Engine',
      icon: BarChart3,
    },
    {
      id: 'ml',
      label: 'Machine Learning',
      icon: Cpu,
      badge: ml?.anomaly?.n_anomalies ? `${ml.anomaly.n_anomalies} outl` : null,
      badgeColor: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
    },
    {
      id: 'insights',
      label: 'AI Insights',
      icon: Sparkles,
      badge: insights?.insights?.length ? `${insights.insights.length} ai` : null,
      badgeColor: 'bg-violet-500/20 text-violet-300 border-violet-500/30',
    },
  ];

  return (
    <aside className="w-56 bg-slate-950/60 border-r border-slate-800/80 flex flex-col justify-between select-none shrink-0">
      <div className="py-3 px-2 flex flex-col gap-1">
        <div className="px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
          <Layers className="w-3 h-3 text-slate-500" />
          <span>Workstation Modules</span>
        </div>

        <nav className="flex flex-col gap-0.5 mt-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center justify-between px-3 py-2 rounded text-xs font-medium transition cursor-pointer text-left ${
                  isActive
                    ? 'bg-slate-800/90 text-indigo-300 border border-slate-700/80 shadow-xs'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60 border border-transparent'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Icon
                    className={`w-4 h-4 transition ${
                      isActive ? 'text-indigo-400' : 'text-slate-500 group-hover:text-slate-300'
                    }`}
                  />
                  <span>{item.label}</span>
                </div>

                {item.badge !== undefined && item.badge !== null && (
                  <span
                    className={`px-1.5 py-0.2 rounded text-[10px] font-mono border ${
                      item.badgeColor || 'bg-slate-800 text-slate-400 border-slate-700'
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* System Engine Version Footer */}
      <div className="p-3 border-t border-slate-800/80 bg-slate-950/40">
        <div className="flex flex-col gap-1 text-[11px] text-slate-400">
          <div className="flex justify-between items-center">
            <span className="text-slate-400">Pipeline Pipeline</span>
            <span className="font-mono text-emerald-400">Steps 1–8</span>
          </div>
          <div className="flex justify-between items-center text-[10px] text-slate-400">
            <span>Deterministic ML + AI</span>
            <span className="font-mono">Local HF</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
