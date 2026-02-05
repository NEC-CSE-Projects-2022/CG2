import { Routes, Route, useLocation, Link } from 'react-router-dom';
import { useState, Suspense, useEffect } from 'react';
import { AnimatePresence } from 'framer-motion';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Predict from './pages/Predict';
import Results from './pages/Results';
import { Toaster } from 'react-hot-toast';
import './App.css';

function App() {
  const location = useLocation();
  const [predictionResult, setPredictionResult] = useState(null);

  // Set document title based on current route
  useEffect(() => {
    const routeTitles = {
      '/': 'Home - CardioRisk AI',
      '/predict': 'Risk Assessment - CardioRisk AI',
      '/results': 'Results - CardioRisk AI',
    };
    document.title = routeTitles[location.pathname] || 'CardioRisk AI';
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Toaster position="top-center" />
      <Navbar />
      <main className="flex-grow">
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<Home />} />
            <Route 
              path="/predict" 
              element={
                <Suspense fallback={<div className="min-h-screen flex items-center justify-center">Loading...</div>}>
                  <Predict setPredictionResult={setPredictionResult} />
                </Suspense>
              } 
            />
            <Route 
              path="/results" 
              element={
                <Suspense fallback={<div className="min-h-screen flex items-center justify-center">Loading results...</div>}>
                  <Results result={predictionResult} setPredictionResult={setPredictionResult} />
                </Suspense>
              } 
            />
            <Route 
              path="*" 
              element={
                <div className="min-h-[60vh] flex items-center justify-center">
                  <div className="text-center">
                    <h2 className="text-3xl font-bold text-gray-900">404</h2>
                    <p className="mt-2 text-gray-600">Page not found</p>
                    <Link 
                      to="/" 
                      className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                    >
                      Go back home
                    </Link>
                  </div>
                </div>
              } 
            />
          </Routes>
        </AnimatePresence>
      </main>
      <Footer />
    </div>
  );
}

export default App;
