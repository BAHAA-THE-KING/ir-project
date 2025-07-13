import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Search,
  Sparkles,
  Zap,
  Target,
  Shuffle,
  Database,
  CheckCircle,
  Box,
  Cpu,
  HardDrive,
  ChevronDown,
  Clock,
} from "lucide-react";
import { useSearch, SearchAlgorithm, Dataset } from "../context/SearchContext";

const Home: React.FC = () => {
  const [query, setQuery] = useState("");
  const [algorithm, setAlgorithm] = useState<SearchAlgorithm>("TF-IDF");
  const [dataset, setDataset] = useState<Dataset>("antique");
  const [resultCount, setResultCount] = useState(10);
  const [useIndexing, setUseIndexing] = useState(true);
  const [useVectorStore, setUseVectorStore] = useState(false);
  const [useQuerySuggestions, setUseQuerySuggestions] = useState(true);
  const [isSearching, setIsSearching] = useState(false);
  const [isDatasetDropdownOpen, setIsDatasetDropdownOpen] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const { performSearch } = useSearch();
  const navigate = useNavigate();

  // Update CSS custom property for slider progress
  useEffect(() => {
    const percentage = ((resultCount - 5) / (50 - 5)) * 100;
    document.documentElement.style.setProperty(
      "--slider-value",
      `${percentage}%`
    );
  }, [resultCount]);

  // Query suggestions
  const allSuggestions = [
    "machine learning algorithms",
    "deep learning neural networks",
    "natural language processing",
    "computer vision applications",
    "reinforcement learning",
    "artificial intelligence",
    "supervised learning",
    "unsupervised learning",
    "convolutional neural networks",
    "transformer models",
    "image recognition",
    "text classification",
    "recommendation systems",
    "data mining techniques",
    "pattern recognition",
  ];

  // Filter suggestions based on query
  useEffect(() => {
    if (query.trim().length > 0 && useQuerySuggestions) {
      const filtered = allSuggestions
        .filter(
          (suggestion) =>
            suggestion.toLowerCase().includes(query.toLowerCase()) &&
            suggestion.toLowerCase() !== query.toLowerCase()
        )
        .slice(0, 5);
      setSuggestions(filtered);
      setShowSuggestions(filtered.length > 0);
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
    }
  }, [query]);

  // Handle algorithm-specific option constraints
  useEffect(() => {
    switch (algorithm) {
      case "TF-IDF":
        // TF-IDF: Disable vector store (incompatible with traditional term frequency)
        setUseVectorStore(false);
        break;
      case "EMBEDDING":
        // Embedding: Disable indexing (pure vector-based approach)
        setUseIndexing(false);
        // Enable vector store as it's essential for embeddings
        setUseVectorStore(false);
        break;
      case "BM25":
        // BM25: Disable vector store (traditional ranking function)
        setUseVectorStore(false);
        break;
      case "HYBRID":
        // Hybrid: All options available (combines multiple approaches)
        break;
    }
  }, [algorithm]);

  // Check if an option should be disabled based on current algorithm
  const isOptionDisabled = (option: "indexing" | "vectorStore") => {
    switch (algorithm) {
      case "TF-IDF":
        return option === "vectorStore";
      case "EMBEDDING":
        return option === "indexing";
      case "BM25":
        return option === "vectorStore";
        case "HYBRID":
        return option === "vectorStore";
      default:
        return false;
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsSearching(true);
    await performSearch(
      query,
      algorithm,
      dataset,
      resultCount,
      useIndexing,
      useVectorStore
    );

    // Navigate after a brief delay to show loading state
    setTimeout(() => {
      navigate("/results");
      setIsSearching(false);
    }, 500);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    setShowSuggestions(false);
  };

  const algorithmOptions = [
    {
      value: "TF-IDF",
      label: "TF-IDF",
      icon: Target,
      description: "Term frequency-inverse document frequency",
    },
    {
      value: "EMBEDDING",
      label: "Embedding",
      icon: Sparkles,
      description: "Semantic vector search",
    },
    {
      value: "BM25",
      label: "BM25",
      icon: Zap,
      description: "Best matching ranking function",
    },
    {
      value: "HYBRID",
      label: "Hybrid",
      icon: Shuffle,
      description: "Combined algorithm approach",
    },
  ];

  const datasetOptions = [
    {
      value: "antique",
      label: "Antique",
      description: "Classical information retrieval dataset",
      icon: Database,
    },
    {
      value: "quora",
      label: "BEIR/Quora",
      description: "Question-answering benchmark dataset",
      icon: HardDrive,
    },
  ];

  return (
    <div className="relative">
      {/* Hero Section */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16">
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center p-3 mb-6 bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-500 dark:to-indigo-500 rounded-2xl shadow-lg">
            <Search className="w-8 h-8 text-white" />
          </div>
          <h1
            className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-slate-800 via-blue-800 to-indigo-800 dark:from-slate-200 dark:via-blue-200 dark:to-indigo-200 bg-clip-text text-transparent mb-6 py-2"
            style={{ lineHeight: "normal" }}
          >
            Intelligent Search
          </h1>
          <p className="text-xl text-slate-600 dark:text-slate-300 max-w-2xl mx-auto leading-relaxed">
            Discover information with advanced search algorithms. Choose from
            TF-IDF, semantic embeddings, BM25, or hybrid approaches.
          </p>
        </div>

        {/* Search Form */}
        <form onSubmit={handleSearch} className="space-y-8">
          {/* Search Input */}
          <div
            className="relative"
            onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
          >
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <Search className="h-5 w-5 text-slate-400" />
            </div>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => {
                if (suggestions.length > 0) {
                  setShowSuggestions(true);
                }
              }}
              placeholder="Enter your search query..."
              className="w-full pl-12 pr-4 py-4 text-lg border-2 border-slate-200 dark:border-slate-600 rounded-2xl focus:border-blue-500 dark:focus:border-blue-400 focus:ring-4 focus:ring-blue-100 dark:focus:ring-blue-900/50 transition-all duration-200 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm text-slate-900 dark:text-slate-100 placeholder-slate-500 dark:placeholder-slate-400"
              disabled={isSearching}
            />

            {/* Query Suggestions */}
            {showSuggestions &&
              suggestions.length > 0 &&
              useQuerySuggestions && (
                <div className="absolute top-full left-0 right-0 mt-2 bg-white/95 dark:bg-slate-800/95 backdrop-blur-md border border-slate-200 dark:border-slate-600 rounded-2xl shadow-xl z-50 overflow-hidden">
                  {suggestions.map((suggestion, index) => (
                    <button
                      key={index}
                      type="button"
                      onClick={() => handleSuggestionClick(suggestion)}
                      className="w-full flex items-center space-x-3 px-4 py-3 text-left hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-all duration-200 text-slate-900 dark:text-slate-100"
                    >
                      <div className="w-8 h-8 bg-gradient-to-r from-purple-600 to-indigo-600 dark:from-purple-500 dark:to-indigo-500 rounded-lg flex items-center justify-center">
                        <Search className="w-4 h-4 text-white" />
                      </div>
                      <div className="flex-1">
                        <div className="font-medium">{suggestion}</div>
                      </div>
                      <Clock className="w-4 h-4 text-slate-400 dark:text-slate-500" />
                    </button>
                  ))}
                </div>
              )}
          </div>

          {/* Dataset Selection */}
          <div className="space-y-4">
            <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
              Dataset
            </label>
            <div
              className="relative"
              onBlur={() =>
                setTimeout(() => setIsDatasetDropdownOpen(false), 150)
              }
            >
              <button
                type="button"
                onClick={() => setIsDatasetDropdownOpen(!isDatasetDropdownOpen)}
                disabled={isSearching}
                className={`w-full flex items-center justify-between px-4 py-4 text-lg border-2 rounded-2xl transition-all duration-200 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm text-slate-900 dark:text-slate-100 ${
                  isDatasetDropdownOpen
                    ? "border-blue-500 dark:border-blue-400 ring-4 ring-blue-100 dark:ring-blue-900/50"
                    : "border-slate-200 dark:border-slate-600 hover:border-slate-300 dark:hover:border-slate-500"
                } ${
                  isSearching
                    ? "cursor-not-allowed opacity-60"
                    : "cursor-pointer"
                }`}
              >
                <div className="flex items-center space-x-3">
                  {(() => {
                    const selectedOption = datasetOptions.find(
                      (opt) => opt.value === dataset
                    );
                    const Icon = selectedOption?.icon || Database;
                    return (
                      <>
                        <div className="w-10 h-10 bg-gradient-to-r from-slate-600 to-slate-700 dark:from-slate-500 dark:to-slate-600 rounded-lg flex items-center justify-center">
                          <Icon className="w-5 h-5 text-white" />
                        </div>
                        <div className="text-left">
                          <div className="font-semibold">
                            {selectedOption?.label}
                          </div>
                          <div className="text-sm text-slate-600 dark:text-slate-400">
                            {selectedOption?.description}
                          </div>
                        </div>
                      </>
                    );
                  })()}
                </div>
                <ChevronDown
                  className={`w-5 h-5 text-slate-400 dark:text-slate-500 transition-transform duration-200 ${
                    isDatasetDropdownOpen ? "rotate-180" : ""
                  }`}
                />
              </button>

              {isDatasetDropdownOpen && (
                <div className="absolute top-full left-0 right-0 mt-2 bg-white/95 dark:bg-slate-800/95 backdrop-blur-md border border-slate-200 dark:border-slate-600 rounded-2xl shadow-xl z-50 overflow-hidden">
                  {datasetOptions.map((option) => {
                    const Icon = option.icon;
                    return (
                      <button
                        key={option.value}
                        type="button"
                        onClick={() => {
                          setDataset(option.value as Dataset);
                          setIsDatasetDropdownOpen(false);
                        }}
                        className={`w-full flex items-center space-x-3 px-4 py-4 text-left transition-all duration-200 ${
                          dataset === option.value
                            ? "bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
                            : "hover:bg-slate-50 dark:hover:bg-slate-700/50 text-slate-900 dark:text-slate-100"
                        }`}
                      >
                        <div
                          className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                            dataset === option.value
                              ? "bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-500 dark:to-indigo-500 text-white"
                              : "bg-gradient-to-r from-slate-600 to-slate-700 dark:from-slate-500 dark:to-slate-600 text-white"
                          }`}
                        >
                          <Icon className="w-5 h-5" />
                        </div>
                        <div className="flex-1">
                          <div className="font-semibold">{option.label}</div>
                          <div
                            className={`text-sm ${
                              dataset === option.value
                                ? "text-blue-600 dark:text-blue-400"
                                : "text-slate-600 dark:text-slate-400"
                            }`}
                          >
                            {option.description}
                          </div>
                        </div>
                        {dataset === option.value && (
                          <CheckCircle className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                        )}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* Algorithm Selection */}
          <div className="space-y-4">
            <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
              Search Algorithm
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {algorithmOptions.map((option) => {
                const Icon = option.icon;
                return (
                  <label
                    key={option.value}
                    className={`relative flex items-center p-4 rounded-xl border-2 cursor-pointer transition-all duration-200 ${
                      algorithm === option.value
                        ? "border-blue-500 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/30 ring-2 ring-blue-100 dark:ring-blue-900/50"
                        : "border-slate-200 dark:border-slate-600 bg-white/60 dark:bg-slate-800/60 hover:border-slate-300 dark:hover:border-slate-500 hover:bg-white/80 dark:hover:bg-slate-800/80"
                    }`}
                  >
                    <input
                      type="radio"
                      name="algorithm"
                      value={option.value}
                      checked={algorithm === option.value}
                      onChange={(e) =>
                        setAlgorithm(e.target.value as SearchAlgorithm)
                      }
                      className="sr-only"
                    />
                    <div
                      className={`flex items-center justify-center w-10 h-10 rounded-lg mr-3 ${
                        algorithm === option.value
                          ? "bg-blue-600 dark:bg-blue-500 text-white"
                          : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
                      }`}
                    >
                      <Icon className="w-5 h-5" />
                    </div>
                    <div className="flex-1">
                      <div className="font-semibold text-slate-900 dark:text-slate-100">
                        {option.label}
                      </div>
                      <div className="text-sm text-slate-600 dark:text-slate-400">
                        {option.description}
                      </div>
                    </div>
                  </label>
                );
              })}
            </div>
          </div>

          {/* Search Optimization Options */}
          <div className="space-y-4">
            <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
              Search Optimization
            </label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {/* Query Suggestions Checkbox */}
              <label
                className={`relative flex items-center p-4 rounded-xl border-2 transition-all duration-200 ${
                  useQuerySuggestions
                    ? "border-purple-500 dark:border-purple-400 bg-purple-50 dark:bg-purple-900/30 ring-2 ring-purple-100 dark:ring-purple-900/50 cursor-pointer"
                    : "border-slate-200 dark:border-slate-600 bg-white/60 dark:bg-slate-800/60 hover:border-slate-300 dark:hover:border-slate-500 hover:bg-white/80 dark:hover:bg-slate-800/80 cursor-pointer"
                }`}
              >
                <input
                  type="checkbox"
                  checked={useQuerySuggestions}
                  onChange={(e) => setUseQuerySuggestions(e.target.checked)}
                  className="sr-only"
                />
                <div
                  className={`flex items-center justify-center w-10 h-10 rounded-lg mr-3 ${
                    useQuerySuggestions
                      ? "bg-purple-600 dark:bg-purple-500 text-white"
                      : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
                  }`}
                >
                  {useQuerySuggestions ? (
                    <CheckCircle className="w-5 h-5" />
                  ) : (
                    <Search className="w-5 h-5" />
                  )}
                </div>
                <div className="flex-1">
                  <div className="font-semibold text-slate-900 dark:text-slate-100">
                    Query Suggestions
                  </div>
                  <div className="text-sm text-slate-600 dark:text-slate-400">
                    {useQuerySuggestions
                      ? "Smart search suggestions"
                      : "Manual query input only"}
                  </div>
                </div>
              </label>

              {/* Indexing Checkbox */}
              <label
                className={`relative flex items-center p-4 rounded-xl border-2 transition-all duration-200 ${
                  isOptionDisabled("indexing")
                    ? "border-slate-200 dark:border-slate-600 bg-slate-50 dark:bg-slate-800/50 cursor-not-allowed opacity-60"
                    : useIndexing
                    ? "border-emerald-500 dark:border-emerald-400 bg-emerald-50 dark:bg-emerald-900/30 ring-2 ring-emerald-100 dark:ring-emerald-900/50 cursor-pointer"
                    : "border-slate-200 dark:border-slate-600 bg-white/60 dark:bg-slate-800/60 hover:border-slate-300 dark:hover:border-slate-500 hover:bg-white/80 dark:hover:bg-slate-800/80 cursor-pointer"
                }`}
              >
                <input
                  type="checkbox"
                  checked={useIndexing}
                  onChange={(e) => setUseIndexing(e.target.checked)}
                  disabled={isOptionDisabled("indexing")}
                  className="sr-only"
                />
                <div
                  className={`flex items-center justify-center w-10 h-10 rounded-lg mr-3 ${
                    isOptionDisabled("indexing")
                      ? "bg-slate-200 dark:bg-slate-700 text-slate-400 dark:text-slate-500"
                      : useIndexing
                      ? "bg-emerald-600 dark:bg-emerald-500 text-white"
                      : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
                  }`}
                >
                  {useIndexing && !isOptionDisabled("indexing") ? (
                    <CheckCircle className="w-5 h-5" />
                  ) : (
                    <Database className="w-5 h-5" />
                  )}
                </div>
                <div className="flex-1">
                  <div
                    className={`font-semibold ${
                      isOptionDisabled("indexing")
                        ? "text-slate-500 dark:text-slate-400"
                        : "text-slate-900 dark:text-slate-100"
                    }`}
                  >
                    Use Indexing
                    {isOptionDisabled("indexing") && (
                      <span className="text-xs text-slate-400 dark:text-slate-500 ml-2">
                        (Not compatible)
                      </span>
                    )}
                  </div>
                  <div
                    className={`text-sm ${
                      isOptionDisabled("indexing")
                        ? "text-slate-400 dark:text-slate-500"
                        : "text-slate-600 dark:text-slate-400"
                    }`}
                  >
                    {useIndexing && !isOptionDisabled("indexing")
                      ? "Enhanced search performance"
                      : "Basic search without indexing"}
                  </div>
                </div>
              </label>

              {/* Vector Store Checkbox */}
              <label
                className={`relative flex items-center p-4 rounded-xl border-2 transition-all duration-200 ${
                  isOptionDisabled("vectorStore")
                    ? "border-slate-200 dark:border-slate-600 bg-slate-50 dark:bg-slate-800/50 cursor-not-allowed opacity-60"
                    : useVectorStore
                    ? "border-orange-500 dark:border-orange-400 bg-orange-50 dark:bg-orange-900/30 ring-2 ring-orange-100 dark:ring-orange-900/50 cursor-pointer"
                    : "border-slate-200 dark:border-slate-600 bg-white/60 dark:bg-slate-800/60 hover:border-slate-300 dark:hover:border-slate-500 hover:bg-white/80 dark:hover:bg-slate-800/80 cursor-pointer"
                }`}
              >
                <input
                  type="checkbox"
                  checked={useVectorStore}
                  onChange={(e) => setUseVectorStore(e.target.checked)}
                  disabled={isOptionDisabled("vectorStore")}
                  className="sr-only"
                />
                <div
                  className={`flex items-center justify-center w-10 h-10 rounded-lg mr-3 ${
                    isOptionDisabled("vectorStore")
                      ? "bg-slate-200 dark:bg-slate-700 text-slate-400 dark:text-slate-500"
                      : useVectorStore
                      ? "bg-orange-600 dark:bg-orange-500 text-white"
                      : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
                  }`}
                >
                  {useVectorStore && !isOptionDisabled("vectorStore") ? (
                    <Cpu className="w-5 h-5" />
                  ) : (
                    <Box className="w-5 h-5" />
                  )}
                </div>
                <div className="flex-1">
                  <div
                    className={`font-semibold ${
                      isOptionDisabled("vectorStore")
                        ? "text-slate-500 dark:text-slate-400"
                        : "text-slate-900 dark:text-slate-100"
                    }`}
                  >
                    Vector Store
                    {isOptionDisabled("vectorStore") && (
                      <span className="text-xs text-slate-400 dark:text-slate-500 ml-2">
                        (Not compatible)
                      </span>
                    )}
                  </div>
                  <div
                    className={`text-sm ${
                      isOptionDisabled("vectorStore")
                        ? "text-slate-400 dark:text-slate-500"
                        : "text-slate-600 dark:text-slate-400"
                    }`}
                  >
                    {useVectorStore && !isOptionDisabled("vectorStore")
                      ? "Semantic vector search"
                      : "Keyword-based search"}
                  </div>
                </div>
              </label>
            </div>
          </div>

          {/* Result Count */}
          <div className="space-y-4">
            <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300">
              Number of Results
            </label>
            <div className="flex items-center space-x-4">
              <input
                type="range"
                min="5"
                max="50"
                step="5"
                value={resultCount}
                onChange={(e) => setResultCount(parseInt(e.target.value))}
                className="flex-1 slider"
                style={{
                  background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${
                    ((resultCount - 5) / (50 - 5)) * 100
                  }%, #e2e8f0 ${
                    ((resultCount - 5) / (50 - 5)) * 100
                  }%, #e2e8f0 100%)`,
                }}
              />
              <div className="bg-white/80 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-600 rounded-lg px-4 py-2 min-w-[60px] text-center font-semibold text-slate-700 dark:text-slate-300">
                {resultCount}
              </div>
            </div>
          </div>

          {/* Search Button */}
          <button
            type="submit"
            disabled={!query.trim() || isSearching}
            className={`w-full py-4 px-8 rounded-2xl font-semibold text-lg transition-all duration-200 ${
              !query.trim() || isSearching
                ? "bg-slate-200 dark:bg-slate-700 text-slate-400 dark:text-slate-500 cursor-not-allowed"
                : "bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-500 dark:to-indigo-500 hover:from-blue-700 hover:to-indigo-700 dark:hover:from-blue-400 dark:hover:to-indigo-400 text-white shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
            }`}
          >
            {isSearching ? (
              <div className="flex items-center justify-center space-x-2">
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                <span>Searching...</span>
              </div>
            ) : (
              <div className="flex items-center justify-center space-x-2">
                <Search className="w-5 h-5" />
                <span>Search</span>
              </div>
            )}
          </button>
        </form>
      </div>

      {/* Features Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
            Advanced Search Capabilities
          </h2>
          <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
            Choose from multiple search algorithms and optimization techniques,
            each designed for different types of queries and content.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-12">
          {algorithmOptions.map((option) => {
            const Icon = option.icon;
            return (
              <div
                key={option.value}
                className="bg-white/60 dark:bg-slate-800/60 backdrop-blur-sm rounded-2xl p-6 border border-slate-200/60 dark:border-slate-700/60 hover:border-slate-300/60 dark:hover:border-slate-600/60 transition-all duration-200 hover:shadow-lg"
              >
                <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-500 dark:to-indigo-500 rounded-xl mb-4">
                  <Icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                  {option.label}
                </h3>
                <p className="text-slate-600 dark:text-slate-400 text-sm">
                  {option.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Optimization Features */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Query Suggestions Feature Highlight */}
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-2xl p-8 border border-purple-200/60 dark:border-purple-700/60">
            <div className="flex items-center justify-center mb-6">
              <div className="flex items-center justify-center w-16 h-16 bg-gradient-to-r from-purple-600 to-indigo-600 dark:from-purple-500 dark:to-indigo-500 rounded-2xl">
                <Search className="w-8 h-8 text-white" />
              </div>
            </div>
            <div className="text-center">
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-4">
                Query Suggestions
              </h3>
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                Intelligent query suggestions powered by machine learning to
                help users discover relevant search terms and improve their
                search experience with contextual recommendations.
              </p>
            </div>
          </div>

          {/* Indexing Feature Highlight */}
          <div className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-2xl p-8 border border-emerald-200/60 dark:border-emerald-700/60">
            <div className="flex items-center justify-center mb-6">
              <div className="flex items-center justify-center w-16 h-16 bg-gradient-to-r from-emerald-600 to-teal-600 dark:from-emerald-500 dark:to-teal-500 rounded-2xl">
                <Database className="w-8 h-8 text-white" />
              </div>
            </div>
            <div className="text-center">
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-4">
                Indexing Techniques
              </h3>
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                Advanced indexing system with pre-processed documents, optimized
                data structures, inverted indexes, and term frequency caching
                for enhanced search performance.
              </p>
            </div>
          </div>

          {/* Vector Store Feature Highlight */}
          <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-2xl p-8 border border-orange-200/60 dark:border-orange-700/60">
            <div className="flex items-center justify-center mb-6">
              <div className="flex items-center justify-center w-16 h-16 bg-gradient-to-r from-orange-600 to-red-600 dark:from-orange-500 dark:to-red-500 rounded-2xl">
                <Box className="w-8 h-8 text-white" />
              </div>
            </div>
            <div className="text-center">
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-4">
                Vector Store
              </h3>
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                High-dimensional vector embeddings for semantic search
                capabilities, enabling contextual understanding and similarity
                matching beyond keyword-based retrieval.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
