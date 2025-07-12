// API Configuration
export const API_CONFIG = {
  // Base URL for API calls
  BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  
  // API endpoints
  ENDPOINTS: {
    SEARCH: '/api/search',
    DOCUMENTS: '/api/documents',
    HEALTH: '/api/health',
    DATASETS: '/api/datasets',
  },
  
  // Request timeout in milliseconds
  TIMEOUT: 30000,
  
  // Retry configuration
  RETRY: {
    MAX_ATTEMPTS: 3,
    DELAY: 1000,
  },
  
  // Development settings
  DEV: {
    ENABLE_MOCK_DATA: import.meta.env.VITE_ENABLE_MOCK_DATA === 'true',
    LOG_REQUESTS: import.meta.env.VITE_DEV_MODE === 'true',
  }
};

// Helper function to build full API URL
export const buildApiUrl = (endpoint: string): string => {
  return `${API_CONFIG.BASE_URL}${endpoint}`;
}; 