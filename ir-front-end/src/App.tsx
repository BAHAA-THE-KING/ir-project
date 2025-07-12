import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Results from './pages/Results';
import Document from './pages/Document';
import { SearchProvider } from './context/SearchContext';
import { ThemeProvider } from './context/ThemeContext';

function App() {
  return (
    <ThemeProvider>
      <SearchProvider>
        <Router>
          <Layout>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/results" element={<Results />} />
              <Route path="/document/:id" element={<Document />} />
            </Routes>
          </Layout>
        </Router>
      </SearchProvider>
    </ThemeProvider>
  );
}

export default App;