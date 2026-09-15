/**
 * DataAlchemy API Data Contracts & Types
 */

export interface DatasetItem {
  name: string;
  path: string;
  file_type: string;
  is_sample: boolean;
}

export interface DatasetListResponse {
  datasets: DatasetItem[];
  active_dataset: string | null;
}

export interface AnalyzeRequest {
  dataset_name: string;
  anomaly_algorithm?: string;
  clustering_algorithm?: string;
  n_clusters?: number;
}

export interface AnalyzeResponse {
  status: string;
  dataset_name: string;
  row_count: number;
  column_count: number;
  health_score: number;
  health_grade: string;
}

export interface PaginatedDataResponse {
  columns: string[];
  dtypes: Record<string, string>;
  rows: Record<string, any>[];
  total_rows: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface OverviewFinding {
  id: string;
  stage: 'quality' | 'statistics' | 'ml';
  type: string;
  title: string;
  description: string;
  importance: string;
}

export interface OverviewResponse {
  dataset_name: string;
  dimensions: {
    rows: number;
    columns: number;
    memory_bytes: number;
  };
  health: {
    score: number;
    grade: string;
    critical_issues: number;
    error_issues: number;
    warning_issues: number;
    total_issues: number;
  };
  missingness: {
    missing_cells: number;
    missing_percentage: number;
  };
  duplicates: {
    duplicate_rows: number;
    duplicate_percentage: number;
  };
  ml_summary: {
    anomalies_found: number;
    anomaly_percentage: number;
    clusters_found: number;
    pca_variance_explained: number;
  };
  executive_summary: string;
  key_findings: OverviewFinding[];
  semantic_type_counts: Record<string, number>;
}

export interface QualityIssue {
  rule_id: string;
  severity: 'CRITICAL' | 'ERROR' | 'WARNING' | 'INFO';
  title: string;
  description: string;
  column: string | null;
  measured_value: any;
  threshold: any;
  impact: string;
  metadata?: Record<string, any>;
}

export interface DataQualityReport {
  dataset_name: string;
  overall_score: number;
  grade: string;
  total_issues: number;
  critical_issues: number;
  error_issues: number;
  warning_issues: number;
  info_issues: number;
  issues: QualityIssue[];
  column_scores: Record<string, number>;
  score_breakdown: Record<string, number>;
}

export interface CorrelationResult {
  column_x: string;
  column_y: string;
  pearson_r: number;
  spearman_rho: number;
  abs_pearson: number;
  strength: string;
  direction: string;
}

export interface DistributionResult {
  column: string;
  skewness: number;
  skewness_type: string;
  is_skewed: boolean;
  kurtosis: number;
  kurtosis_type: string;
  mean: number;
  std: number;
  median: number;
  iqr: number;
}

export interface ColumnProfile {
  column_name: string;
  semantic_type: string;
  total_count: number;
  null_count: number;
  null_percentage: number;
  distinct_count: number;
  unique_percentage: number;
  common_stats: Record<string, any>;
  type_specific_stats: {
    mean?: number;
    std?: number;
    min?: number;
    max?: number;
    median?: number;
    skewness?: number;
    quantiles?: Record<string, number>;
    histogram?: {
      bins: number[];
      counts: number[];
    };
    top_categories?: Array<{ value: any; count: number; percentage: number }>;
    cardinality_ratio?: number;
    min_date?: string;
    max_date?: string;
  };
}

export interface StatisticsReport {
  dataset_name: string;
  row_count: number;
  column_count: number;
  findings: Array<{
    finding_id: string;
    finding_type: string;
    title: string;
    description: string;
    importance: string;
    measured_values: Record<string, any>;
    affected_columns: string[];
  }>;
  correlations: CorrelationResult[];
  distributions: DistributionResult[];
  column_profiles: Record<string, ColumnProfile>;
}

export interface AnomalyResult {
  algorithm: string;
  n_anomalies: number;
  anomaly_percentage: number;
  anomaly_indices: number[];
  feature_contributions: Record<string, number>;
  score_summary: Record<string, number>;
  raw_scores?: number[];
  status: string;
}

export interface ClusterSummary {
  cluster_id: number;
  size: number;
  proportion: number;
  feature_means: Record<string, number>;
}

export interface ClusterResult {
  algorithm: string;
  n_clusters: number;
  cluster_labels: number[];
  cluster_sizes: Record<string, number>;
  cluster_proportions: Record<string, number>;
  cluster_summaries: ClusterSummary[];
  distinguishing_features: Array<{
    feature: string;
    f_stat?: number;
    cluster_means?: Record<string, number>;
  }>;
  silhouette_score: number | null;
  status: string;
}

export interface PCAResult {
  n_components: number;
  explained_variance_ratio: number[];
  cumulative_explained_variance: number[];
  loadings: Record<string, Record<string, number>>;
  status: string;
}

export interface MLReport {
  dataset_name: string;
  row_count: number;
  feature_count: number;
  anomaly: AnomalyResult;
  clustering: ClusterResult;
  dimensionality: PCAResult;
  findings: Array<{
    finding_id: string;
    finding_type: string;
    title: string;
    description: string;
    importance: string;
    measured_values: Record<string, any>;
    affected_columns: string[];
    affected_rows?: number[];
  }>;
  visualization_data?: {
    pca_scatter_2d?: Array<{
      row_idx: number;
      pc1: number;
      pc2: number;
      cluster: number;
      is_anomaly: boolean;
      score: number;
    }>;
    scree_plot?: Array<{
      component: string;
      variance_ratio: number;
      cumulative_variance: number;
    }>;
    loadings_heatmap?: Array<{
      feature: string;
      pc1: number;
      pc2: number;
    }>;
  };
}

export interface AIInsight {
  insight_id: string;
  title: string;
  summary: string;
  explanation: string;
  importance: 'HIGH' | 'MEDIUM' | 'LOW' | 'INFO';
  category: string;
  evidence: Record<string, any>;
  affected_columns: string[];
  caveats: string | null;
}

export interface AIRecommendation {
  recommendation_id: string;
  title: string;
  description: string;
  action_type: string;
  supporting_evidence: string[];
  priority: 'high' | 'medium' | 'low';
}

export interface AIInsightReport {
  dataset_name: string;
  executive_summary: string;
  insights: AIInsight[];
  recommendations: AIRecommendation[];
  warnings: string[];
}

export interface ExplainRequest {
  finding_id: string;
  title: string;
  evidence?: Record<string, any>;
  category?: string;
  context?: string;
}

export interface ExplainResponse {
  finding_id: string;
  explanation: string;
  recommendation?: string;
}
