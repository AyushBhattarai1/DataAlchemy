import React, { createContext, useContext, useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import type {
  AIInsightReport,
  AnalyzeRequest,
  DataQualityReport,
  DatasetItem,
  ExplainRequest,
  ExplainResponse,
  MLReport,
  OverviewResponse,
  StatisticsReport,
} from '../types/api';

export type TabType = 'overview' | 'data' | 'quality' | 'statistics' | 'ml' | 'insights';

interface ExplainModalState {
  isOpen: boolean;
  request: ExplainRequest | null;
  response: ExplainResponse | null;
  isLoading: boolean;
  error: string | null;
}

interface DataContextType {
  datasets: DatasetItem[];
  activeDataset: string | null;
  status: 'idle' | 'loading' | 'analyzing' | 'ready' | 'error';
  errorMessage: string | null;
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;

  // Analysis Reports
  overview: OverviewResponse | null;
  quality: DataQualityReport | null;
  statistics: StatisticsReport | null;
  ml: MLReport | null;
  insights: AIInsightReport | null;

  // Actions
  refreshDatasets: () => Promise<void>;
  selectDataset: (datasetName: string) => Promise<void>;
  uploadFile: (file: File) => Promise<void>;
  rerunAnalysis: (params?: Partial<AnalyzeRequest>) => Promise<void>;

  // AI Modal
  explainModal: ExplainModalState;
  openExplainModal: (request: ExplainRequest) => Promise<void>;
  closeExplainModal: () => void;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

export const DataProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [activeDataset, setActiveDataset] = useState<string | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'analyzing' | 'ready' | 'error'>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('overview');

  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [quality, setQuality] = useState<DataQualityReport | null>(null);
  const [statistics, setStatistics] = useState<StatisticsReport | null>(null);
  const [ml, setML] = useState<MLReport | null>(null);
  const [insights, setInsights] = useState<AIInsightReport | null>(null);

  const [explainModal, setExplainModal] = useState<ExplainModalState>({
    isOpen: false,
    request: null,
    response: null,
    isLoading: false,
    error: null,
  });

  const loadAllReports = async () => {
    try {
      const [ov, ql, st, mlRes, ins] = await Promise.all([
        apiClient.getOverview(),
        apiClient.getQuality(),
        apiClient.getStatistics(),
        apiClient.getML(),
        apiClient.getInsights(),
      ]);
      setOverview(ov);
      setQuality(ql);
      setStatistics(st);
      setML(mlRes);
      setInsights(ins);
      setStatus('ready');
      setErrorMessage(null);
    } catch (err: any) {
      console.error('Failed to load reports:', err);
      setStatus('error');
      setErrorMessage(err.message || 'Failed to fetch analytical reports');
    }
  };

  const selectDataset = async (datasetName: string) => {
    setStatus('analyzing');
    setErrorMessage(null);
    try {
      await apiClient.runAnalysis({ dataset_name: datasetName });
      setActiveDataset(datasetName);
      await loadAllReports();
    } catch (err: any) {
      console.error('Analysis failed:', err);
      setStatus('error');
      setErrorMessage(err.message || `Failed to analyze dataset ${datasetName}`);
    }
  };

  const refreshDatasets = async () => {
    try {
      const res = await apiClient.getDatasets();
      setDatasets(res.datasets);
      if (res.active_dataset) {
        setActiveDataset(res.active_dataset);
      } else if (!activeDataset && res.datasets.length > 0) {
        // Auto-select first sample (prefer customers.csv if present)
        const defaultDs = res.datasets.find((d) => d.name === 'customers.csv') || res.datasets[0];
        await selectDataset(defaultDs.name);
      }
    } catch (err: any) {
      console.error('Failed to fetch dataset list:', err);
      setStatus('error');
      setErrorMessage(err.message || 'Failed to communicate with DataAlchemy backend.');
    }
  };

  const uploadFile = async (file: File) => {
    setStatus('analyzing');
    setErrorMessage(null);
    try {
      const item = await apiClient.uploadDataset(file);
      await refreshDatasets();
      await selectDataset(item.name);
    } catch (err: any) {
      setStatus('error');
      setErrorMessage(err.message || 'File upload and analysis failed');
    }
  };

  const rerunAnalysis = async (params?: Partial<AnalyzeRequest>) => {
    if (!activeDataset) return;
    setStatus('analyzing');
    setErrorMessage(null);
    try {
      await apiClient.runAnalysis({
        dataset_name: activeDataset,
        ...params,
      });
      await loadAllReports();
    } catch (err: any) {
      setStatus('error');
      setErrorMessage(err.message || 'Failed to rerun analysis');
    }
  };

  const openExplainModal = async (request: ExplainRequest) => {
    setExplainModal({
      isOpen: true,
      request,
      response: null,
      isLoading: true,
      error: null,
    });
    try {
      const res = await apiClient.explain(request);
      setExplainModal((prev) => ({
        ...prev,
        response: res,
        isLoading: false,
      }));
    } catch (err: any) {
      setExplainModal((prev) => ({
        ...prev,
        isLoading: false,
        error: err.message || 'Failed to generate AI explanation',
      }));
    }
  };

  const closeExplainModal = () => {
    setExplainModal({
      isOpen: false,
      request: null,
      response: null,
      isLoading: false,
      error: null,
    });
  };

  useEffect(() => {
    refreshDatasets();
  }, []);

  return (
    <DataContext.Provider
      value={{
        datasets,
        activeDataset,
        status,
        errorMessage,
        activeTab,
        setActiveTab,
        overview,
        quality,
        statistics,
        ml,
        insights,
        refreshDatasets,
        selectDataset,
        uploadFile,
        rerunAnalysis,
        explainModal,
        openExplainModal,
        closeExplainModal,
      }}
    >
      {children}
    </DataContext.Provider>
  );
};

export const useData = () => {
  const context = useContext(DataContext);
  if (!context) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
};
