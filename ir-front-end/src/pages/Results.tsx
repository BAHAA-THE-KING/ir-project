import React, { useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Search, Clock, FileText, ArrowLeft, ExternalLink, Database, CheckCircle, Box, Cpu, HardDrive } from 'lucide-react';
import { SearchResult, useSearch } from '../context/SearchContext';

const Results: React.FC = () => {
  const { searchState } = useSearch();
  const navigate = useNavigate();

  useEffect(() => {
    // Redirect to home if no search has been performed
    if (!searchState.query) {
      navigate('/');
    }
  }, [searchState.query, navigate]);

  if (!searchState.query) {
    return null;
  }

  const algorithmLabels = {
    'TF-IDF': 'TF-IDF',
    'EMBEDDING': 'Embedding',
    'BM25': 'BM25',
    'HYBRID': 'Hybrid',
  };

  const datasetLabels = {
    'antique': 'Antique',
    'quora': 'BEIR/Quora',
  };

  const renderResults = () => {
    return (
      <div className="space-y-6">
        {searchState.results.map((result, index) => renderResultCard(result, index))}
      </div>
    );
  };

  const renderResultCard = (result: SearchResult, index: number) => (
    <div
      key={result.doc_id}
      className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm rounded-2xl p-6 border border-slate-200/60 dark:border-slate-700/60 hover:border-slate-300/60 dark:hover:border-slate-600/60 transition-all duration-200 hover:shadow-lg"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-500 dark:to-indigo-500 rounded-lg flex items-center justify-center text-white font-bold text-sm">
            {index + 1}
          </div>
          <div>
            <div className="flex items-center space-x-2 text-sm text-slate-500 dark:text-slate-400">
              <span>Document ID: {result.doc_id}</span>
              <span>•</span>
              <span>Score: {result.score.toFixed(2)}</span>
              {searchState.useIndexing && (
                <>
                  <span>•</span>
                  <span className="text-emerald-600 dark:text-emerald-400 font-medium">Indexed</span>
                </>
              )}
              {searchState.useVectorStore && (
                <>
                  <span>•</span>
                  <span className="text-orange-600 dark:text-orange-400 font-medium">Vector</span>
                </>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Link
            to={`/document/${result.doc_id}`}
            className="inline-flex items-center space-x-1 px-3 py-1.5 bg-blue-600 dark:bg-blue-500 text-white rounded-lg hover:bg-blue-700 dark:hover:bg-blue-400 transition-colors duration-200 text-sm font-medium"
          >
            <ExternalLink className="w-3 h-3" />
            <span>View</span>
          </Link>
        </div>
      </div>

      <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
        {result.snippet}
      </p>

      <div className="flex items-center justify-between pt-4 border-t border-slate-200/60 dark:border-slate-700/60">
        <div className="flex items-center space-x-2 text-sm text-slate-500 dark:text-slate-400">
          <FileText className="w-4 h-4" />
          <span>Full document available</span>
        </div>
        <div className="text-xs text-slate-400 dark:text-slate-500">
          Relevance: {Math.round(result.score * 100)}%
        </div>
      </div>
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <Link
          to="/"
          className="inline-flex items-center space-x-2 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 mb-6 transition-colors duration-200"
        >
          <ArrowLeft className="w-4 h-4" />
          <span className="font-medium">Back to Search</span>
        </Link>

        <div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm rounded-2xl p-6 border border-slate-200/60 dark:border-slate-700/60 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">Search Results</h1>
            <div className="flex items-center space-x-4 text-sm text-slate-600 dark:text-slate-400">
              <div className="flex items-center space-x-1">
                <Clock className="w-4 h-4" />
                <span>{searchState.searchTime}ms</span>
              </div>
              <div className="flex items-center space-x-1">
                <FileText className="w-4 h-4" />
                <span>{searchState.totalResults} results</span>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4 text-sm">
            <div className="flex items-center space-x-2">
              <Search className="w-4 h-4 text-slate-400 dark:text-slate-500" />
              <span className="text-slate-600 dark:text-slate-400">Query:</span>
              <span className="font-semibold text-slate-900 dark:text-slate-100 bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded-md">
                "{searchState.query}"
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <HardDrive className="w-4 h-4 text-slate-400 dark:text-slate-500" />
              <span className="text-slate-600 dark:text-slate-400">Dataset:</span>
              <span className="font-semibold text-indigo-600 dark:text-indigo-400 bg-indigo-100 dark:bg-indigo-900/50 px-2 py-1 rounded-md">
                {datasetLabels[searchState.dataset_name]}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-slate-600 dark:text-slate-400">Algorithm:</span>
              <span className="font-semibold text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/50 px-2 py-1 rounded-md">
                {algorithmLabels[searchState.model]}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-slate-600 dark:text-slate-400">Indexing:</span>
              <span className={`font-semibold px-2 py-1 rounded-md flex items-center space-x-1 ${
                searchState.useIndexing 
                  ? 'text-emerald-700 dark:text-emerald-300 bg-emerald-100 dark:bg-emerald-900/50' 
                  : 'text-amber-700 dark:text-amber-300 bg-amber-100 dark:bg-amber-900/50'
              }`}>
                {searchState.useIndexing ? (
                  <>
                    <CheckCircle className="w-3 h-3" />
                    <span>Enabled</span>
                  </>
                ) : (
                  <>
                    <Database className="w-3 h-3" />
                    <span>Disabled</span>
                  </>
                )}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-slate-600 dark:text-slate-400">Vector Store:</span>
              <span className={`font-semibold px-2 py-1 rounded-md flex items-center space-x-1 ${
                searchState.useVectorStore 
                  ? 'text-orange-700 dark:text-orange-300 bg-orange-100 dark:bg-orange-900/50' 
                  : 'text-slate-700 dark:text-slate-300 bg-slate-100 dark:bg-slate-700'
              }`}>
                {searchState.useVectorStore ? (
                  <>
                    <Cpu className="w-3 h-3" />
                    <span>Enabled</span>
                  </>
                ) : (
                  <>
                    <Box className="w-3 h-3" />
                    <span>Disabled</span>
                  </>
                )}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-slate-600 dark:text-slate-400">Limit:</span>
              <span className="font-semibold text-slate-900 dark:text-slate-100 bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded-md">
                {searchState.resultCount}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Results */}
      {searchState.results.length === 0 ? (
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-slate-100 dark:bg-slate-700 rounded-full flex items-center justify-center mx-auto mb-4">
            <Search className="w-8 h-8 text-slate-400 dark:text-slate-500" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">No Results Found</h3>
          <p className="text-slate-600 dark:text-slate-400 max-w-md mx-auto">
            We couldn't find any documents matching your search query. Try different keywords or search terms.
          </p>
        </div>
      ) : (
        renderResults()
      )}

      {/* Pagination Placeholder */}
      {searchState.results.length > 0 && (
        <div className="mt-12 text-center">
          <div className="inline-flex items-center justify-center space-x-2 text-sm text-slate-500 dark:text-slate-400">
            <span>Showing {searchState.results.length} of {searchState.totalResults} results</span>
            {searchState.useIndexing && (
              <>
                <span>•</span>
                <span className="text-emerald-600 dark:text-emerald-400">Enhanced with indexing</span>
              </>
            )}
            {searchState.useVectorStore && (
              <>
                <span>•</span>
                <span className="text-orange-600 dark:text-orange-400">Semantic vector search</span>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Results;