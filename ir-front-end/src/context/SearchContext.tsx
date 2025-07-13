import React, { createContext, useContext, useState, ReactNode } from "react";
import {
  searchDocuments,
  SearchRequest,
  SearchResponse,
} from "../services/api";

export type SearchAlgorithm = "TF-IDF" | "EMBEDDING" | "BM25" | "HYBRID";
export type Dataset = "antique" | "quora";

export interface SearchResult {
  doc_id: string;
  snippet: string;
  score: number;
  url: string;
}

export interface SearchState {
  query: string;
  model: SearchAlgorithm;
  dataset_name: Dataset;
  resultCount: number;
  useIndexing: boolean;
  useVectorStore: boolean;
  results: SearchResult[];
  searchTime: number;
  totalResults: number;
}

interface SearchContextType {
  searchState: SearchState;
  setSearchState: (state: Partial<SearchState>) => void;
  performSearch: (
    query: string,
    model: SearchAlgorithm,
    dataset_name: Dataset,
    resultCount: number,
    useIndexing: boolean,
    useVectorStore: boolean
  ) => Promise<void>;
}

const SearchContext = createContext<SearchContextType | undefined>(undefined);

export const useSearch = () => {
  const context = useContext(SearchContext);
  if (!context) {
    throw new Error("useSearch must be used within a SearchProvider");
  }
  return context;
};

export const SearchProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [searchState, setSearchStateInternal] = useState<SearchState>({
    query: "",
    model: "TF-IDF",
    dataset_name: "antique",
    resultCount: 10,
    useIndexing: true,
    useVectorStore: false,
    results: [],
    searchTime: 0,
    totalResults: 0,
  });

  const setSearchState = (newState: Partial<SearchState>) => {
    setSearchStateInternal((prev) => ({ ...prev, ...newState }));
  };

  const performSearch = async (
    query: string,
    model: SearchAlgorithm,
    dataset_name: Dataset,
    resultCount: number,
    useIndexing: boolean,
    useVectorStore: boolean
  ) => {
    const startTime = Date.now();

    try {
      const searchRequest: SearchRequest = {
        query,
        model,
        dataset_name,
        top_k: resultCount,
        use_inverted_index: useIndexing,
        use_vector_store: useVectorStore,
      };

      const searchResponse: SearchResponse = await searchDocuments(
        searchRequest
      );

      setSearchState({
        query,
        model,
        dataset_name,
        resultCount,
        useIndexing,
        useVectorStore,
        results: searchResponse.results.map((res) => ({
          ...res,
          url: "/document/" + res.doc_id,
        })),
        searchTime: Date.now() - startTime,
        totalResults: searchResponse.totalResults,
      });
    } catch (error) {
      console.error("Search failed:", error);
    }
  };

  return (
    <SearchContext.Provider
      value={{ searchState, setSearchState, performSearch }}
    >
      {children}
    </SearchContext.Provider>
  );
};

export const getDocumentById = (id: string) => {
  return {};
};
