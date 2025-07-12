# API Integration Guide

This document explains where and how to integrate APIs in your React frontend project.

## 📍 **Primary API Integration Points**

### 1. **SearchContext.tsx** - Main Search API
**Location**: `src/context/SearchContext.tsx`
**Function**: `performSearch()`

This is the **primary place** to integrate your search API. The function has been updated to:
- Make real API calls to your backend
- Fall back to mock data if the API fails
- Handle errors gracefully

**API Endpoint**: `POST /api/search`
**Request Body**:
```json
{
  "query": "search term",
  "algorithm": "hybrid",
  "dataset": "antique",
  "resultCount": 10,
  "useIndexing": true,
  "useVectorStore": false
}
```

### 2. **Document.tsx** - Document Retrieval API
**Location**: `src/pages/Document.tsx`

For fetching individual document details by ID.

**API Endpoint**: `GET /api/documents/{id}`

### 3. **API Service Layer** - Centralized API Calls
**Location**: `src/services/api.ts`

All API calls are centralized here for:
- Consistent error handling
- Request/response typing
- Easy maintenance

## 🔧 **Configuration**

### Environment Variables
Create a `.env` file in your project root:
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_DEV_MODE=true
VITE_ENABLE_MOCK_DATA=false
```

### API Configuration
**Location**: `src/config/api.ts`

Configure:
- Base URL
- Endpoints
- Timeouts
- Retry logic
- Development settings

## 📡 **Available API Endpoints**

| Endpoint | Method | Purpose | Location |
|----------|--------|---------|----------|
| `/api/search` | POST | Search documents | `SearchContext.tsx` |
| `/api/documents/{id}` | GET | Get document by ID | `Document.tsx` |
| `/api/health` | GET | API health check | `api.ts` |
| `/api/datasets` | GET | Get dataset info | `api.ts` |

## 🚀 **How to Add New APIs**

### 1. Add to API Service
```typescript
// In src/services/api.ts
export const newApiCall = async (params: NewApiParams): Promise<NewApiResponse> => {
  const response = await fetch(buildApiUrl(API_CONFIG.ENDPOINTS.NEW_ENDPOINT), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    throw new Error(`API call failed: ${response.statusText}`);
  }

  return response.json();
};
```

### 2. Add to Configuration
```typescript
// In src/config/api.ts
ENDPOINTS: {
  // ... existing endpoints
  NEW_ENDPOINT: '/api/new-endpoint',
}
```

### 3. Use in Components
```typescript
// In your component
import { newApiCall } from '../services/api';

const handleApiCall = async () => {
  try {
    const result = await newApiCall(params);
    // Handle success
  } catch (error) {
    // Handle error
  }
};
```

## 🔄 **Error Handling**

The API service includes:
- Automatic error throwing for non-200 responses
- Fallback to mock data in development
- Console logging for debugging
- Graceful degradation

## 🧪 **Development vs Production**

### Development
- Uses mock data as fallback
- Detailed error logging
- Configurable via environment variables

### Production
- Real API calls only
- Minimal error logging
- Optimized for performance

## 📝 **Best Practices**

1. **Always use the API service layer** - Don't make direct fetch calls in components
2. **Type your requests/responses** - Use TypeScript interfaces
3. **Handle errors gracefully** - Provide fallbacks and user feedback
4. **Use environment variables** - Don't hardcode API URLs
5. **Test API integration** - Mock API responses in tests

## 🔍 **Testing API Integration**

```typescript
// Example test
import { searchDocuments } from '../services/api';

// Mock the fetch function
global.fetch = jest.fn();

test('searchDocuments makes correct API call', async () => {
  const mockResponse = { results: [], totalResults: 0, searchTime: 100 };
  (fetch as jest.Mock).mockResolvedValueOnce({
    ok: true,
    json: async () => mockResponse,
  });

  const result = await searchDocuments({
    query: 'test',
    algorithm: 'hybrid',
    dataset: 'antique',
    resultCount: 10,
    useIndexing: true,
    useVectorStore: false,
  });

  expect(result).toEqual(mockResponse);
});
``` 