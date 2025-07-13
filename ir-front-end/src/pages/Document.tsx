import React, { useEffect, useState } from 'react';
import { useParams, Link, useNavigate, useLocation } from 'react-router-dom';
import { ArrowLeft, FileText, Calendar, User, ExternalLink } from 'lucide-react';

const Document: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const [document, setDocument] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  // Get dataset_name from location state
  const datasetName = location.state?.dataset_name;

  useEffect(() => {
    if (id && datasetName) {
      setLoading(true);
      fetch(`http://127.0.0.1:8000/document/${id}?dataset_name=${encodeURIComponent(datasetName)}`)
        .then(res => {
          if (!res.ok) throw new Error('Document not found');
          return res.text();
        })
        .then(text => {
          setDocument({
            id,
            fullText: text,
            // Add more fields if needed
          });
        })
        .catch(() => {
          setDocument(null);
        })
        .finally(() => setLoading(false));
    } else {
      navigate('/');
    }
  }, [id, datasetName, navigate]);

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="animate-pulse">
          <div className="h-4 bg-slate-200 rounded w-1/4 mb-6"></div>
          <div className="h-8 bg-slate-200 rounded w-3/4 mb-4"></div>
          <div className="h-4 bg-slate-200 rounded w-1/2 mb-8"></div>
          <div className="space-y-3">
            <div className="h-4 bg-slate-200 rounded"></div>
            <div className="h-4 bg-slate-200 rounded"></div>
            <div className="h-4 bg-slate-200 rounded w-5/6"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!document) {
    return (
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-slate-100 dark:bg-slate-700 rounded-full flex items-center justify-center mx-auto mb-4">
            <FileText className="w-8 h-8 text-slate-400 dark:text-slate-500" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">Document Not Found</h3>
          <p className="text-slate-600 dark:text-slate-400 max-w-md mx-auto mb-6">
            The document you're looking for doesn't exist or has been removed.
          </p>
          <Link
            to="/"
            className="inline-flex items-center space-x-2 px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded-lg hover:bg-blue-700 dark:hover:bg-blue-400 transition-colors duration-200"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Search</span>
          </Link>
        </div>
      </div>
    );
  }

  const formatContent = (content: string) => {
    // Simple markdown-like formatting
    const lines = content.split('\n');
    const formatted = lines.map((line, index) => {
      if (line.startsWith('# ')) {
        return <h1 key={index} className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-6 mt-8 first:mt-0">{line.substring(2)}</h1>;
      } else if (line.startsWith('## ')) {
        return <h2 key={index} className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-4 mt-6">{line.substring(3)}</h2>;
      } else if (line.startsWith('### ')) {
        return <h3 key={index} className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-3 mt-5">{line.substring(4)}</h3>;
      } else if (line.startsWith('- ')) {
        return <li key={index} className="text-slate-700 dark:text-slate-300 mb-1">{line.substring(2)}</li>;
      } else if (line.trim() === '') {
        return <div key={index} className="h-4"></div>;
      } else {
        return <p key={index} className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">{line}</p>;
      }
    });
    return formatted;
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <Link
          to="/results"
          className="inline-flex items-center space-x-2 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 mb-6 transition-colors duration-200"
        >
          <ArrowLeft className="w-4 h-4" />
          <span className="font-medium">Back to Results</span>
        </Link>

        <div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm rounded-2xl p-6 border border-slate-200/60 dark:border-slate-700/60 shadow-sm mb-8">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-500 dark:to-indigo-500 rounded-xl flex items-center justify-center">
                <FileText className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">Document {document.id}</h1>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <button className="p-2 text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 transition-colors duration-200">
                <ExternalLink className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm rounded-2xl p-8 border border-slate-200/60 dark:border-slate-700/60 shadow-sm">
        <div className="prose prose-slate max-w-none">
          <div className="space-y-4">
            {formatContent(document.fullText)}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-12 text-center">
        <div className="inline-flex items-center space-x-4">
          <Link
            to="/results"
            className="inline-flex items-center space-x-2 px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded-lg hover:bg-blue-700 dark:hover:bg-blue-400 transition-colors duration-200"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Results</span>
          </Link>
          <Link
            to="/"
            className="inline-flex items-center space-x-2 px-4 py-2 border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors duration-200"
          >
            <span>New Search</span>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Document;