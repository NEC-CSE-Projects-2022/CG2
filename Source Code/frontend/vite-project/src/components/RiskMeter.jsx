import { motion, useAnimation } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';
import { Info } from 'lucide-react';

const RiskMeter = ({
  percentage = 50,
  size = 200,
  lineWidth = 20,
  label = 'Cardiovascular Risk',
  showInfo = true,
  className = '',
}) => {
  const controls = useAnimation();
  const [isHovered, setIsHovered] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);
  const tooltipRef = useRef(null);
  
  // Calculate the radius and circumference for the gauge
  const radius = (size - lineWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (percentage / 100) * circumference;
  
  // Determine risk level and color - aligned with getRiskDetails in Results.jsx
  const getRiskLevel = (percent) => {
    // Ensure high risk is always red
    if (percent >= 75) return { level: 'High Risk', color: '#EF4444', description: 'Significantly elevated risk of cardiovascular disease. Medical consultation strongly recommended.' };
    if (percent >= 60) return { level: 'Moderate-High Risk', color: '#F97316', description: 'Elevated risk. Consider lifestyle changes and consult with a healthcare provider.' };
    if (percent >= 40) return { level: 'Moderate Risk', color: '#F59E0B', description: 'Moderate risk. Lifestyle changes and regular monitoring are advised.' };
    if (percent >= 20) return { level: 'Low-Moderate Risk', color: '#84CC16', description: 'Slightly elevated risk. Maintain healthy habits and monitor your health.' };
    return { level: 'Low Risk', color: '#22C55E', description: 'Low risk. Continue your healthy lifestyle.' };
  };
  
  const risk = getRiskLevel(percentage);
  
  // Animate the gauge when percentage changes
  useEffect(() => {
    controls.start({
      strokeDashoffset: offset,
      transition: { duration: 1.5, ease: 'easeInOut' },
    });
  }, [percentage, offset, controls]);
  
  // Handle tooltip visibility
  useEffect(() => {
    if (isHovered) {
      const timer = setTimeout(() => setShowTooltip(true), 300);
      return () => clearTimeout(timer);
    } else {
      setShowTooltip(false);
    }
  }, [isHovered]);
  
  // Close tooltip when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (tooltipRef.current && !tooltipRef.current.contains(event.target)) {
        setShowTooltip(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className={`relative flex flex-col items-center ${className}`}>
      {/* Gauge */}
      <div className="relative" style={{ width: size, height: size }}>
        {/* Background circle */}
        <svg width={size} height={size} className="transform -rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="#E5E7EB"
            strokeWidth={lineWidth}
            strokeLinecap="round"
          />
          
          {/* Animated progress circle */}
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={risk.color}
            strokeWidth={lineWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={controls}
            style={{ filter: 'drop-shadow(0 0 8px rgba(59, 130, 246, 0.4))' }}
          />
        </svg>
        
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-4xl font-bold" style={{ color: risk.color }}>
            {Math.round(percentage)}%
          </span>
          <span className="text-sm text-gray-500 mt-1">
            {risk.level} 
          </span>
        </div>
        
        {/* Risk indicator dot */}
        <div 
          className="absolute top-0 left-1/2 w-3 h-3 rounded-full -mt-1.5 -ml-1.5"
          style={{
            backgroundColor: risk.color,
            transform: `rotate(${180 + (percentage / 100) * 180}deg)`,
            transformOrigin: `center ${size / 2 + lineWidth / 2}px`,
            transition: 'transform 1.5s ease-in-out, background-color 0.5s ease',
          }}
        />
      </div>
      
      {/* Label and info */}
      <div className="mt-4 text-center">
        <div className="flex items-center justify-center space-x-2">
          <h3 className="text-lg font-medium text-gray-800">{label}</h3>
          {showInfo && (
            <div className="relative">
              <button
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
                onClick={() => setShowTooltip(!showTooltip)}
                className="text-gray-400 hover:text-blue-500 transition-colors"
                aria-label="Risk information"
              >
                <Info size={16} />
              </button>
              
              {/* Tooltip */}
              {showTooltip && (
                <div 
                  ref={tooltipRef}
                  className="absolute z-10 w-64 p-3 mt-2 text-sm text-left text-gray-700 bg-white border border-gray-200 rounded-lg shadow-lg -left-32"
                  onMouseEnter={() => setIsHovered(true)}
                  onMouseLeave={() => setIsHovered(false)}
                >
                  <p className="font-medium mb-1">{risk.level} Risk ({percentage}%)</p>
                  <p className="text-gray-600 text-xs">{risk.description} based on the provided health metrics.</p>
                  <div className="mt-2 pt-2 border-t border-gray-100">
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>0%</span>
                      <span>50%</span>
                      <span>100%</span>
                    </div>
                    <div className="h-2 w-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded-full mt-1"></div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
        
        {/* Risk description */}
        <p className="mt-2 text-sm text-gray-600 max-w-xs">
          {risk.description}
        </p>
      </div>
      
      {/* Risk indicators with improved drag visualization */}
      <div className="w-full mt-6 px-2">
        <div className="flex justify-between text-xs text-gray-500 mb-1 px-1">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
        <div className="relative h-3 w-full">
          <div className="absolute h-2 w-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded-full top-1/2 -translate-y-1/2"></div>
          <motion.div 
            className="absolute h-4 w-1 bg-white border-2 border-gray-700 rounded-full -top-0.5 -ml-0.5 shadow-md z-10"
            style={{
              left: `${percentage}%`,
              backgroundColor: risk.color,
              borderColor: 'white',
              boxShadow: '0 0 0 2px white, 0 0 0 3px ' + risk.color,
            }}
            initial={{ x: 0 }}
            animate={{ x: 0 }}
            transition={{ duration: 1.5, ease: 'easeInOut' }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1 px-1">
          <span className="text-green-600 font-medium">Low</span>
          <span className="text-yellow-600 font-medium">Medium</span>
          <span className="text-red-600 font-medium">High</span>
        </div>
      </div>
    </div>
  );
};

export default RiskMeter;