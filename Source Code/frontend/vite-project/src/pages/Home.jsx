import { motion, AnimatePresence } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { HeartPulse, Activity, Stethoscope, ClipboardCheck, ArrowRight, Heart } from 'lucide-react';

const features = [
  {
    icon: <ClipboardCheck className="h-8 w-8 text-blue-600" />,
    title: 'Comprehensive Analysis',
    description: 'Advanced AI evaluates multiple health parameters to provide accurate cardiovascular risk assessment.'
  },
  {
    icon: <Activity className="h-8 w-8 text-blue-500" />,
    title: 'Real-time Insights',
    description: 'Get immediate feedback on your heart health with detailed risk analysis and visual reports.'
  },
  {
    icon: <Stethoscope className="h-8 w-8 text-blue-400" />,
    title: 'Clinical Precision',
    description: 'Developed using the Sulianova CVD dataset and validated by cardiology experts.'
  }
];

const Home = () => {
  const navigate = useNavigate();
  
  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <AnimatePresence>
        {/* Hero Section */}
        <div className="relative overflow-hidden pt-16 pb-12 bg-gradient-to-br from-blue-50 to-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 md:py-20">
            <div className="flex flex-col lg:flex-row items-center">
              <div className="lg:w-1/2 text-center lg:text-left">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="inline-flex items-center justify-center px-4 py-2 rounded-full bg-blue-50 text-blue-700 text-sm font-medium mb-6"
              >
                <HeartPulse className="h-5 w-5 mr-2" />
                <span>Cardiovascular Risk Assessment</span>
              </motion.div>
              
              <motion.h1 
                className="text-4xl tracking-tight font-extrabold text-gray-900 sm:text-5xl md:text-6xl"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                <span className="block">Take Control of Your</span>
                <span className="block text-blue-600">Heart Health Today</span>
              </motion.h1>
              
              <motion.p 
                className="mt-3 max-w-md mx-auto text-base text-gray-500 sm:text-lg md:mt-5 md:text-xl md:max-w-3xl"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                Our AI-powered platform helps you understand your cardiovascular risk factors and provides personalized recommendations to improve your heart health.
              </motion.p>
              
              <motion.div 
                className="mt-8 flex flex-col sm:flex-row justify-center gap-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
              >
                <Link
                  to="/predict"
                  className="w-full sm:w-auto inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 md:py-4 md:text-lg md:px-10 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1"
                >
                  Get Started
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Link>
                
                <button
                  onClick={() => scrollToSection('how-it-works')}
                  className="w-full sm:w-auto inline-flex items-center justify-center px-8 py-3 border border-gray-200 text-base font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 md:py-4 md:text-lg md:px-10 shadow hover:shadow-md transition-all duration-300"
                >
                  Learn More
                </button>
              </motion.div>
              </div>
              
              {/* Animated Border Image */}
              <div className="lg:w-1/2 mt-12 lg:mt-0 flex justify-center lg:justify-end">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ 
                    opacity: 1, 
                    y: 0,
                  }}
                  transition={{ duration: 0.5 }}
                  className="relative w-full max-w-xl rounded-2xl p-1"
                  style={{
                    background: 'linear-gradient(45deg, #3b82f6, #60a5fa, #93c5fd, #60a5fa, #3b82f6)',
                    backgroundSize: '300% 300%',
                    animation: 'gradient 8s ease infinite',
                  }}
                >
                  <div className="relative rounded-xl overflow-hidden">
                    <img
                      src="https://www.saralamemorialhospital.com/public/assets/admin/uploads/departments/Cardiologyimage1707214184.jpg"
                      alt="Cardiology Department"
                      className="w-full h-auto object-cover"
                      style={{ minHeight: '400px' }}
                    />
                    <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-blue-700/20"></div>
                  </div>
                  {/* Animated border effect */}
                  <motion.div 
                    className="absolute inset-0 rounded-2xl"
                    style={{
                      background: 'linear-gradient(45deg, #3b82f6, #60a5fa, #93c5fd, #60a5fa, #3b82f6)',
                      backgroundSize: '300% 300%',
                      filter: 'blur(10px)',
                      opacity: 0.7,
                      zIndex: -1,
                    }}
                    animate={{
                      backgroundPosition: ['0% 0%', '100% 100%'],
                    }}
                    transition={{
                      duration: 8,
                      repeat: Infinity,
                      repeatType: 'reverse',
                      ease: 'linear',
                    }}
                  />
                </motion.div>
              </div>
              <style jsx global>{`
                @keyframes gradient {
                  0% { background-position: 0% 50%; }
                  50% { background-position: 100% 50%; }
                  100% { background-position: 0% 50%; }
                }
              `}</style>
            </div>
          </div>
          
          {/* Animated Heartbeat */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ 
              opacity: 1, 
              y: 0,
              scale: [1, 1.1, 1],
            }}
            transition={{ 
              duration: 0.5, 
              delay: 0.4,
              scale: {
                repeat: Infinity,
                duration: 1.5,
                ease: "easeInOut"
              }
            }}
            className="mt-12 flex justify-center"
          >
            <div className="flex items-center justify-center space-x-4">
              <Heart className="h-10 w-10 text-red-500" />
              <span className="text-xl font-medium text-gray-700">Healthy Heart, Happy Life</span>
            </div>
          </motion.div>
        </div>

        {/* Features Section */}
        <div id="how-it-works" className="py-12 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="lg:text-center">
              <h2 className="text-base text-blue-600 font-semibold tracking-wide uppercase">How It Works</h2>
              <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
                Simple Steps to Better Heart Health
              </p>
              <p className="mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
                Our process is designed to be simple, fast, and informative.
              </p>
            </div>

            <div className="mt-10">
              <div className="space-y-10 md:space-y-0 md:grid md:grid-cols-3 md:gap-x-8 md:gap-y-10">
                {features.map((feature, index) => (
                  <motion.div 
                    key={feature.title}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    className="relative"
                  >
                    <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-blue-500 text-white">
                      {feature.icon}
                    </div>
                    <div className="ml-16">
                      <h3 className="text-lg leading-6 font-medium text-gray-900">{feature.title}</h3>
                      <p className="mt-2 text-base text-gray-500">{feature.description}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div id="contact" className="bg-blue-700 text-white">
          <div className="max-w-2xl mx-auto text-center py-16 px-4 sm:py-20 sm:px-6 lg:px-8">
            <h2 className="text-3xl font-extrabold text-white sm:text-4xl">
              <span className="block">Ready to take control of your heart health?</span>
            </h2>
            <p className="mt-4 text-lg leading-6 text-blue-100">
              It only takes a few minutes to understand your risk and get personalized recommendations.
            </p>
            <motion.div 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="mt-8 flex justify-center"
            >
              <Link
                to="/predict"
                className="inline-flex items-center px-8 py-4 border-2 border-white text-base font-bold rounded-lg text-blue-700 bg-white hover:bg-blue-50 transition-all duration-300"
              >
                Start Your Assessment
                <ArrowRight className="ml-3 h-5 w-5" />
              </Link>
            </motion.div>
          </div>
        </div>
      </AnimatePresence>
    </div>
  );
};

export default Home;