import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { HeartPulse, Menu, X } from 'lucide-react';
import { useState, useEffect } from 'react';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 10;
      if (isScrolled !== scrolled) {
        setScrolled(isScrolled);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [scrolled]);

  const navLinks = [
    { name: 'Home', path: '/' },
    { name: 'Risk Assessment', path: '/predict' },
    { name: 'Results', path: '/results' }
  ];

  const isActive = (path) => {
    if (location.pathname === path) {
      return 'text-white font-semibold text-base border-b-2 border-white';
    } else if (location.pathname.startsWith(path) && path !== '/') {
      return 'text-white font-medium text-base';
    }
    return 'text-blue-100 hover:text-white text-base';
  };

  return (
    <motion.nav 
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className={`fixed w-full z-50 py-1 transition-all duration-300 ${
        scrolled ? 'bg-gradient-to-r from-blue-800 to-blue-600 shadow-lg' : 'bg-gradient-to-r from-blue-800 to-blue-600/95 backdrop-blur-sm'
      }`}
    >
      <div className="max-w-7xl mx-auto px-3 sm:px-5 lg:px-6">
        <div className="flex justify-between items-center h-12">
          {/* Logo */}
          <Link to="/" className="flex-shrink-0 flex items-center">
            <HeartPulse className="h-5 w-5 text-blue-600" />
            <span className="ml-1.5 text-lg font-bold text-white whitespace-nowrap">
              CardioRisk AI
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:block">
            <div className="ml-4 flex items-center space-x-4">
              {navLinks.map((link) => (
                <Link
                  key={link.name}
                  to={link.path}
                  className={`${isActive(link.path)} px-3 py-2 font-medium transition-all duration-200 whitespace-nowrap`}
                >
                  {link.name}
                </Link>
              ))}
              <Link
                to="/predict"
                className="ml-2 px-4 py-2 bg-white text-blue-700 text-sm font-medium rounded-md hover:bg-blue-50 transition-all duration-200 shadow-sm whitespace-nowrap"
              >
                Get Started
              </Link>
            </div>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-white hover:bg-blue-700/30 focus:outline-none"
            >
              {isOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="md:hidden bg-blue-800 shadow-lg rounded-b-lg overflow-hidden"
        >
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            {navLinks.map((link) => (
              <Link
                key={link.name}
                to={link.path}
                className={`block px-4 py-3 rounded-md text-base font-medium ${
                  location.pathname === link.path
                    ? 'bg-blue-700/40 text-white font-semibold border-l-4 border-white pl-3.5'
                    : 'text-blue-100 hover:bg-blue-700/50 hover:pl-4 transition-all duration-200'
                }`}
                onClick={() => setIsOpen(false)}
              >
                {link.name}
              </Link>
            ))}
            <div className="px-3 pt-2">
              <Link
                to="/predict"
                className="block w-full text-center px-4 py-2.5 bg-white text-blue-700 font-medium rounded-md hover:bg-blue-50 text-base"
                onClick={() => setIsOpen(false)}
              >
                Get Started
              </Link>
            </div>
          </div>
        </motion.div>
      )}
    </motion.nav>
  );
};

export default Navbar;