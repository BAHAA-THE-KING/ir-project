// API service for centralized API calls
import { API_CONFIG, buildApiUrl } from '../config/api';

export interface SearchRequest {
  query: string;
  model: 'TF-IDF' | 'EMBEDDING' | 'BM25' | 'HYBRID';
  dataset_name: 'antique' | 'quora';
  top_k: number;
  use_inverted_index: boolean;
  use_vector_store: boolean;
}

export interface SearchResponse {
  results: Array<{
    doc_id: string;
    score: number;
    snippet: string;
  }>;
  totalResults: number;
  searchTime: number;
}

export interface Document {
  id: string;
  title: string;
  content: string;
  url: string;
  metadata?: Record<string, unknown>;
}

export interface SuggestRequest {
  query: string;
  dataset: string;
  top_k: number;
}

export interface Suggestion {
  suggestion: string;
  snippet: string;
}

export interface SuggestResponse {
  suggestions: Suggestion[];
}

// Search API
export const searchDocuments = async (searchRequest: SearchRequest): Promise<SearchResponse> => {
  const response = await fetch(buildApiUrl(API_CONFIG.ENDPOINTS.SEARCH), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(searchRequest),
  });

  if (!response.ok) {
    throw new Error(`Search failed: ${response.statusText}`);
  }

  return response.json();
};

// Document API
export const getDocumentById = async (id: string): Promise<Document> => {
  const response = await fetch(buildApiUrl(`${API_CONFIG.ENDPOINTS.DOCUMENTS}/${id}`));

  if (!response.ok) {
    throw new Error(`Failed to fetch document: ${response.statusText}`);
  }

  return response.json();
};

// Health check API
export const checkApiHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(buildApiUrl(API_CONFIG.ENDPOINTS.HEALTH));
    return response.ok;
  } catch (error) {
    console.error('API health check failed:', error);
    return false;
  }
};

// Dataset info API
export const getDatasetInfo = async (): Promise<Record<string, unknown>> => {
  const response = await fetch(buildApiUrl(API_CONFIG.ENDPOINTS.DATASETS));

  if (!response.ok) {
    throw new Error(`Failed to fetch dataset info: ${response.statusText}`);
  }

  return response.json();
};

export const fetchSuggestions = async (
  suggestRequest: SuggestRequest
): Promise<Suggestion[]> => {
  const response = await fetch(buildApiUrl(API_CONFIG.ENDPOINTS.SUGGEST), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(suggestRequest),
  });

  if (!response.ok) {
    throw new Error(`Suggestion fetch failed: ${response.statusText}`);
  }

  const data: SuggestResponse = await response.json();
  return data.suggestions;
}; 