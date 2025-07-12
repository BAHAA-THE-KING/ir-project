import React, { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Search, FileText, Home, Sun, Moon } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 transition-colors duration-300">
      {/* Header */}
      <header className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200/60 dark:border-slate-700/60 sticky top-0 z-50 transition-colors duration-300">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link to="/" className="flex items-center space-x-3 group">
              <div className="p-2 bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-500 dark:to-indigo-500 rounded-lg group-hover:from-blue-700 group-hover:to-indigo-700 dark:group-hover:from-blue-400 dark:group-hover:to-indigo-400 transition-all duration-200">
                <Search className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400 bg-clip-text text-transparent">
                SearchEngine
              </span>
            </Link>
            
            <nav className="flex items-center space-x-2">
              <Link
                to="/"
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                  location.pathname === '/'
                    ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300'
                    : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800'
                }`}
              >
                <Home className="w-4 h-4" />
                <span className="font-medium">Home</span>
              </Link>
              {location.pathname.startsWith('/results') && (
                <Link
                  to="/results"
                  className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300"
                >
                  <Search className="w-4 h-4" />
                  <span className="font-medium">Results</span>
                </Link>
              )}
              {location.pathname.startsWith('/document') && (
                <div className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300">
                  <FileText className="w-4 h-4" />
                  <span className="font-medium">Document</span>
                </div>
              )}
              
              {/* Theme Toggle */}
              <button
                onClick={toggleTheme}
                className="p-2 rounded-lg text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition-all duration-200"
                aria-label="Toggle theme"
              >
                {theme === 'light' ? (
                  <Moon className="w-5 h-5" />
                ) : (
                  <Sun className="w-5 h-5" />
                )}
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm border-t border-slate-200/60 dark:border-slate-700/60 mt-20 transition-colors duration-300">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-2 mb-4">
              <div className="p-2 bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-500 dark:to-indigo-500 rounded-lg">
                <Search className="w-5 h-5 text-white" />
              </div>
              <span className="text-lg font-bold bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400 bg-clip-text text-transparent">
                SearchEngine
              </span>
            </div>
            <p className="text-slate-600 dark:text-slate-400 text-sm">
              Advanced search algorithms for intelligent information retrieval
            </p>
            <div className="mt-4 flex items-center justify-center space-x-6 text-xs text-slate-500 dark:text-slate-400">
              <span>TF-IDF</span>
              <span>•</span>
              <span>Embedding</span>
              <span>•</span>
              <span>BM25</span>
              <span>•</span>
              <span>Hybrid</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout;