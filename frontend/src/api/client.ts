/**
 * DataAlchemy API Client
 */

import type {
  AIInsightReport,
  AnalyzeRequest,
  AnalyzeResponse,
  DataQualityReport,
  DatasetListResponse,
  ExplainRequest,
  ExplainResponse,
  MLReport,
  OverviewResponse,
  PaginatedDataResponse,
  StatisticsReport,
} from '../types/api';

const API_BASE = '/api';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = `Request failed with status ${res.status}`;
    try {
      const errData = await res.json();
      if (errData && errData.detail) {
        errorDetail = errData.detail;
      }
    } catch {
      // ignore json parse error
    }
    throw new Error(errorDetail);
  }
  return res.json();
}

export const apiClient = {
  async getDatasets(): Promise<DatasetListResponse> {
    const res = await fetch(`${API_BASE}/datasets`);
    return handleResponse<DatasetListResponse>(res);
  },

  async uploadDataset(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/datasets/upload`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(res);
  },

  async runAnalysis(req: AnalyzeRequest): Promise<AnalyzeResponse> {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    return handleResponse<AnalyzeResponse>(res);
  },

  async getData(params: {
    page?: number;
    pageSize?: number;
    sortBy?: string;
    sortOrder?: 'asc' | 'desc';
    search?: string;
  }): Promise<PaginatedDataResponse> {
    const query = new URLSearchParams();
    if (params.page) query.set('page', params.page.toString());
    if (params.pageSize) query.set('page_size', params.pageSize.toString());
    if (params.sortBy) query.set('sort_by', params.sortBy);
    if (params.sortOrder) query.set('sort_order', params.sortOrder);
    if (params.search) query.set('search', params.search);

    const res = await fetch(`${API_BASE}/data?${query.toString()}`);
    return handleResponse<PaginatedDataResponse>(res);
  },

  async getOverview(): Promise<OverviewResponse> {
    const res = await fetch(`${API_BASE}/overview`);
    return handleResponse<OverviewResponse>(res);
  },

  async getQuality(): Promise<DataQualityReport> {
    const res = await fetch(`${API_BASE}/quality`);
    return handleResponse<DataQualityReport>(res);
  },

  async getStatistics(): Promise<StatisticsReport> {
    const res = await fetch(`${API_BASE}/statistics`);
    return handleResponse<StatisticsReport>(res);
  },

  async getML(): Promise<MLReport> {
    const res = await fetch(`${API_BASE}/ml`);
    return handleResponse<MLReport>(res);
  },

  async getInsights(): Promise<AIInsightReport> {
    const res = await fetch(`${API_BASE}/insights`);
    return handleResponse<AIInsightReport>(res);
  },

  async explain(req: ExplainRequest): Promise<ExplainResponse> {
    const res = await fetch(`${API_BASE}/insights/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    return handleResponse<ExplainResponse>(res);
  },
};
