import { useState, useRef, useEffect } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Download, 
  ArrowLeft, 
  CheckCircle, 
  AlertTriangle,
  HeartPulse,
  Activity,
  TrendingUp,
  Clock,
  FileText,
  Share2,
  Printer,
  ChevronRight,
  Info,
  Droplet,
  Heart,
  Activity as ActivityIcon,
  Smile,
  Frown,
  Zap,
  Crosshair,
  Thermometer,
  Droplets,
  Scale,
  User
} from 'lucide-react';
import RiskMeter from '../components/RiskMeter';
import { toast } from 'react-hot-toast';
import { saveAs } from 'file-saver';
import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';

// Helper function to format dates
const formatDate = (dateString) => {
  const options = { 
    year: 'numeric', 
    month: 'long', 
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  };
  return new Date(dateString).toLocaleDateString(undefined, options);
};

// Helper function to get risk category and color
const getRiskDetails = (score) => {
  // Aligned with RiskMeter component colors
  if (score >= 75) return { 
    category: 'High Risk', 
    color: 'red-500',  // Using red-500 to match #EF4444 from RiskMeter
    description: 'Significantly elevated risk of cardiovascular disease. Medical consultation strongly recommended.'
  };
  if (score >= 60) return { 
    category: 'Moderate-High Risk', 
    color: 'orange-500',  // Using orange-500 to match #F97316 from RiskMeter
    description: 'Elevated risk. Consider lifestyle changes and consult with a healthcare provider.'
  };
  if (score >= 40) return { 
    category: 'Moderate Risk', 
    color: 'yellow-500',  // Using yellow-500 to match #F59E0B from RiskMeter
    description: 'Moderate risk. Lifestyle changes and regular monitoring are advised.'
  };
  if (score >= 20) return { 
    category: 'Low-Moderate Risk', 
    color: 'lime-500',  // Using lime-500 to match #84CC16 from RiskMeter
    description: 'Slightly elevated risk. Maintain healthy habits and monitor your health.'
  };
  return { 
    category: 'Low Risk', 
    color: 'green-500',  // Using green-500 to match #22C55E from RiskMeter
    description: 'Low risk. Continue your healthy lifestyle.'
  };
};

// Helper function to get cholesterol label
const getCholesterolLabel = (level) => {
  if (level === undefined || level === null) return 'Not Specified';
  const levelStr = level.toString();
  switch(levelStr) {
    case '1': return 'Normal (Below 200 mg/dL)';
    case '2': return 'Above Normal (200-239 mg/dL)';
    case '3': return 'High (240 mg/dL and above)';
    default: return 'Not Specified';
  }
};

// Helper function to get glucose label
const getGlucoseLabel = (level) => {
  if (level === undefined || level === null) return 'Not Specified';
  const levelStr = level.toString();
  switch(levelStr) {
    case '1': return 'Normal (Below 100 mg/dL)';
    case '2': return 'Prediabetes (100-125 mg/dL)';
    case '3': return 'Diabetes (126 mg/dL and above)';
    default: return 'Not Specified';
  }
};

// Helper function to get yes/no label
const getYesNoLabel = (value) => {
  // Handle both string '1'/'0' and number 1/0
  return value === '1' || value === 1 ? 'Yes' : 'No';
};

const Results = ({ result: propResult, setPredictionResult }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const reportRef = useRef(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  
  // Get result data from either navigation state or props
  const resultData = location.state || propResult;

  // If no result data is available, redirect to home
  useEffect(() => {
    if (!resultData) {
      navigate('/');
    }
  }, [resultData, navigate]);

  // Get data from result or show error
  if (!resultData) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-white p-4">
        <div className="text-center max-w-md mx-auto">
          <div className="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-red-100 mb-4">
            <AlertTriangle className="h-8 w-8 text-red-600" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900">No Results Found</h2>
          <p className="mt-2 text-gray-600">
            We couldn't find any assessment results. Please complete the assessment to view your personalized report.
          </p>
          <div className="mt-6">
            <Link
              to="/predict"
              className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              <ArrowLeft className="h-5 w-5 mr-2" />
              Back to Assessment
            </Link>
          </div>
        </div>
      </div>
    );
  }

  // Process result data
  const {
    age,
    gender,
    height,
    weight,
    ap_hi,
    ap_lo,
    cholesterol,
    glucose,
    smoke,
    alco,
    active,
    riskScore = 50, // Default fallback
    timestamp = new Date().toISOString()
  } = resultData;

  // Calculate BMI and blood pressure category
  const bmi = weight && height ? (weight / ((height / 100) ** 2)).toFixed(1) : null;
  
  // Determine blood pressure category
  const getBPCategory = (systolic, diastolic) => {
    if (systolic >= 180 || diastolic >= 120) return 'Hypertensive Crisis';
    if (systolic >= 140 || diastolic >= 90) return 'High Blood Pressure (Stage 2)';
    if (systolic >= 130 || diastolic >= 80) return 'High Blood Pressure (Stage 1)';
    if (systolic >= 120) return 'Elevated';
    return 'Normal';
  };
  
  const bpCategory = getBPCategory(ap_hi, ap_lo);
  
  const riskDetails = getRiskDetails(riskScore);

  // Generate recommendations based on risk factors
  const generateRecommendations = () => {
    const recommendations = [];
    
    // Blood pressure recommendations
    if (bpCategory === 'High') {
      recommendations.push({
        title: 'Monitor Blood Pressure',
        description: 'Your blood pressure is in the high range. Consider regular monitoring and consult with a healthcare provider.',
        icon: <Thermometer className="h-6 w-6 text-red-500" />
      });
    }
    
    // Cholesterol recommendations
    if (cholesterol && parseInt(cholesterol) > 1) {
      recommendations.push({
        title: 'Cholesterol Management',
        description: 'Your cholesterol level is above normal. Consider dietary changes and regular exercise.',
        icon: <Droplet className="h-6 w-6 text-orange-500" />
      });
    }
    
    // Smoking recommendations
    if (smoke === '1') {
      recommendations.push({
        title: 'Smoking Cessation',
        description: 'Quitting smoking can significantly reduce your cardiovascular risk.',
        icon: <ActivityIcon className="h-6 w-6 text-blue-500" />
      });
    }
    
    // Activity recommendations
    if (active === '0') {
      recommendations.push({
        title: 'Increase Physical Activity',
        description: 'Aim for at least 150 minutes of moderate exercise per week.',
        icon: <Zap className="h-6 w-6 text-green-500" />
      });
    }
    
    // Add general recommendations
    recommendations.push({
      title: 'Regular Check-ups',
      description: 'Schedule regular health check-ups to monitor your cardiovascular health.',
      icon: <Heart className="h-6 w-6 text-pink-500" />
    });
    
    return recommendations;
  };

  const recommendations = generateRecommendations();

  // Generate PDF report
  const generatePDF = async () => {
    setIsGenerating(true);
    try {
      const canvas = await html2canvas(reportRef.current, {
        scale: 2,
        useCORS: true,
        logging: false,
        backgroundColor: '#ffffff',
        scrollY: -window.scrollY
      });

      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
      });

      const imgProps = pdf.getImageProperties(imgData);
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;

      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save(`CardioRisk_Report_${new Date().toISOString().split('T')[0]}.pdf`);
      
      toast.success('Report downloaded successfully!');
      setIsSaved(true);
      setTimeout(() => setIsSaved(false), 3000);
    } catch (error) {
      console.error('Error generating PDF:', error);
      toast.error('Failed to generate report. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  // Share functionality
  const handleShare = async () => {
    try {
      if (navigator.share) {
        await navigator.share({
          title: 'My Cardiovascular Risk Assessment',
          text: `My cardiovascular risk score is ${riskScore}%. ${riskDetails.description}`,
          url: window.location.href,
        });
      } else {
        // Fallback for browsers that don't support Web Share API
        await navigator.clipboard.writeText(window.location.href);
        toast.success('Link copied to clipboard!');
      }
    } catch (error) {
      console.error('Error sharing:', error);
    }
  };

  // Print functionality
  const handlePrint = () => {
    window.print();
  };

  // Blood pressure category is already defined above
  
  // Health metric cards
  const healthMetrics = [
    {
      name: 'Blood Pressure',
      value: `${ap_hi}/${ap_lo} mmHg`,
      status: bpCategory.toLowerCase(),
      description: bpCategory,
      icon: <Activity className="h-6 w-6 text-blue-500" />,
      risk: bpCategory !== 'Normal' && bpCategory !== 'Elevated'
    },
    {
      name: 'Cholesterol',
      value: getCholesterolLabel(cholesterol),
      status: parseInt(cholesterol) > 1 ? 'elevated' : 'normal',
      description: getCholesterolLabel(cholesterol),
      icon: <Droplets className="h-6 w-6 text-purple-500" />,
      risk: parseInt(cholesterol) > 2
    },
    {
      name: 'Glucose',
      value: getGlucoseLabel(glucose),
      status: parseInt(glucose) > 1 ? 'elevated' : 'normal',
      description: getGlucoseLabel(glucose),
      icon: <ActivityIcon className="h-6 w-6 text-green-500" />,
      risk: parseInt(glucose) > 2
    },
    {
      name: 'BMI',
      value: bmi,
      status: bmi < 18.5 ? 'underweight' : bmi < 25 ? 'normal' : bmi < 30 ? 'overweight' : 'obese',
      description: bmi < 18.5 ? 'Underweight' : bmi < 25 ? 'Normal' : bmi < 30 ? 'Overweight' : 'Obese',
      icon: <Scale className="h-6 w-6 text-orange-500" />,
      risk: bmi >= 25
    }
  ];

  // Risk factors with more detailed information
  const riskFactors = [
    {
      name: 'Smoking Status',
      value: getYesNoLabel(smoke),
      risk: smoke === 1 || smoke === '1',
      description: smoke === 1 || smoke === '1' ? 'Increases cardiovascular risk' : 'Non-smoker - lower risk',
      icon: smoke === 1 || smoke === '1' ? <Frown className="h-5 w-5 text-red-500" /> : <Smile className="h-5 w-5 text-green-500" />
    },
    {
      name: 'Alcohol Consumption',
      value: getYesNoLabel(alco),
      risk: alco === 1 || alco === '1',
      description: alco === 1 || alco === '1' ? 'Regular alcohol use may increase risk' : 'No regular alcohol use',
      icon: alco === 1 || alco === '1' ? <Frown className="h-5 w-5 text-orange-500" /> : <Smile className="h-5 w-5 text-green-500" />
    },
    {
      name: 'Physical Activity',
      value: active === 1 || active === '1' ? 'Active' : 'Inactive',
      risk: active === 0 || active === '0',
      description: active === 1 || active === '1' ? 'Regular activity reduces risk' : 'Inactivity increases risk',
      icon: active === 1 || active === '1' ? <Smile className="h-5 w-5 text-green-500" /> : <Frown className="h-5 w-5 text-orange-500" />
    },
    {
      name: 'Age',
      value: age,
      risk: age > 50,
      description: age > 50 ? 'Increased age is a risk factor' : 'Younger age is protective',
      icon: <User className="h-5 w-5 text-blue-500" />
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto" ref={reportRef}>
        {/* Header */}
        <div className="text-center mb-10">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center justify-center px-6 py-2 rounded-full bg-blue-100 text-blue-700 text-sm font-medium mb-6"
          >
            <HeartPulse className="h-5 w-5 mr-2" />
            Cardiovascular Risk Assessment
          </motion.div>
          
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-3xl font-extrabold text-gray-900 sm:text-4xl"
          >
            Your Results
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mt-3 text-lg text-gray-600"
          >
            Assessment completed on {formatDate(timestamp)}
          </motion.p>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="mt-6 flex flex-wrap justify-center gap-3"
          >
            <button
              onClick={generatePDF}
              disabled={isGenerating}
              className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${
                isGenerating ? 'bg-blue-400' : 'bg-blue-600 hover:bg-blue-700'
              } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
            >
              <Download className="h-4 w-4 mr-2" />
              {isGenerating ? 'Generating...' : 'Download Report'}
            </button>
            
            <button
              onClick={handleShare}
              className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              <Share2 className="h-4 w-4 mr-2" />
              Share
            </button>
            
            <button
              onClick={handlePrint}
              className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              <Printer className="h-4 w-4 mr-2" />
              Print
            </button>
          </motion.div>
        </div>

        {/* Navigation Tabs */}
        <div className="border-b border-gray-200 mb-8">
          <nav className="-mb-px flex space-x-8 overflow-x-auto">
            {['overview', 'metrics', 'recommendations'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`${
                  activeTab === tab
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm capitalize`}
              >
                {tab === 'overview' ? 'Risk Overview' : tab}
              </button>
            ))}
          </nav>
        </div>

        {/* Main Content */}
        <div className="bg-white shadow-xl rounded-xl overflow-hidden mb-8">
          {/* Risk Summary */}
          <div className="bg-gradient-to-r from-blue-600 to-blue-800 px-8 py-6 text-white">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center">
              <div>
                <h2 className="text-2xl font-bold">Your Cardiovascular Risk</h2>
                <p className="mt-1 text-blue-100">
                  {age} years • {gender === '1' ? 'Male' : 'Female'} • {bmi ? `BMI: ${bmi}` : ''}
                </p>
              </div>
              <div className="mt-4 md:mt-0">
                <div 
                  className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${
                    riskDetails.color === 'red' ? 'bg-red-100 text-red-800' :
                    riskDetails.color === 'orange' ? 'bg-orange-100 text-orange-800' :
                    'bg-green-100 text-green-800'
                  }`}
                >
                  <span className={`h-2 w-2 rounded-full mr-2 ${
                    riskDetails.color === 'red' ? 'bg-red-500' :
                    riskDetails.color === 'orange' ? 'bg-orange-500' :
                    'bg-green-500'
                  }`}></span>
                  {riskDetails.category}
                </div>
              </div>
            </div>
          </div>

          {/* Risk Score Section */}
          <div className="p-8 border-b border-gray-200">
            <h3 className="text-xl font-semibold text-gray-900 mb-6">Your Cardiovascular Risk Score</h3>
            
            <div className="flex flex-col lg:flex-row items-center justify-between gap-8">
              <div className="w-full lg:w-1/3 flex justify-center">
                <RiskMeter 
                  percentage={parseFloat(riskScore)} 
                  size={200}
                  lineWidth={20}
                  label="10-Year Risk"
                />
              </div>
              
              <div className="w-full lg:w-2/3">
                <div className="bg-blue-50 p-6 rounded-lg">
                  <h4 className="text-lg font-medium text-gray-900 mb-3">What does this mean?</h4>
                  <p className="text-gray-700 mb-4">{riskDetails.description}</p>
                  
                  <div className="mt-4 space-y-2">
                    <div className="flex items-center">
                      <div className="w-4 h-4 rounded-full" style={{backgroundColor: '#22C55E'}}></div>
                      <span className="text-sm text-gray-700 ml-2">0-19%: Low Risk</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-4 h-4 rounded-full" style={{backgroundColor: '#84CC16'}}></div>
                      <span className="text-sm text-gray-700 ml-2">20-39%: Low-Moderate Risk</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-4 h-4 rounded-full" style={{backgroundColor: '#F59E0B'}}></div>
                      <span className="text-sm text-gray-700 ml-2">40-59%: Moderate Risk</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-4 h-4 rounded-full" style={{backgroundColor: '#F97316'}}></div>
                      <span className="text-sm text-gray-700 ml-2">60-74%: Moderate-High Risk</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-4 h-4 rounded-full" style={{backgroundColor: '#EF4444'}}></div>
                      <span className="text-sm text-gray-700 ml-2">75-100%: High Risk</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Active Tab Content */}
          <div className="p-8">
            {activeTab === 'overview' && (
              <div className="space-y-8">
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Key Health Metrics</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {healthMetrics.map((metric, index) => (
                      <div key={index} className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                        <div className="flex items-center justify-between">
                          <h4 className="text-sm font-medium text-gray-500">{metric.name}</h4>
                          {metric.icon}
                        </div>
                        <p className="mt-2 text-2xl font-semibold text-gray-900">{metric.value}</p>
                        <p className="mt-1 text-sm text-gray-500 capitalize">{metric.status}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Risk Factors</h3>
                  <div className="space-y-3">
                    {riskFactors.map((factor, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center">
                          <div className={`p-2 rounded-full ${factor.risk ? 'bg-red-50' : 'bg-green-50'}`}>
                            {factor.icon}
                          </div>
                          <span className="ml-3 text-sm font-medium text-gray-900">{factor.name}</span>
                        </div>
                        <span className={`text-sm font-medium ${
                          factor.risk ? 'text-red-600' : 'text-green-600'
                        }`}>
                          {factor.value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'metrics' && (
              <div className="space-y-6">
                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Detailed Health Metrics</h3>
                  
                  <div className="space-y-6">
                    <div>
                      <h4 className="text-sm font-medium text-gray-500 mb-2">Blood Pressure</h4>
                      <div className="bg-white p-4 rounded-lg border border-gray-200">
                        <div className="flex items-center justify-between">
                          <span className="text-gray-700">Systolic/Diastolic</span>
                          <span className="font-medium">{ap_hi}/{ap_lo} mmHg</span>
                        </div>
                        <div className="mt-2">
                          <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div 
                              className={`h-2.5 rounded-full ${
                                bpCategory === 'High' ? 'bg-red-500' : 
                                bpCategory === 'Elevated' ? 'bg-yellow-500' : 'bg-green-500'
                              }`} 
                              style={{ width: `${Math.min(100, (ap_hi / 200) * 100)}%` }}
                            ></div>
                          </div>
                          <div className="flex justify-between text-xs text-gray-500 mt-1">
                            <span>Low</span>
                            <span>Normal</span>
                            <span>Elevated</span>
                            <span>High</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-gray-500 mb-2">Cholesterol</h4>
                      <div className="bg-white p-4 rounded-lg border border-gray-200">
                        <div className="flex items-center justify-between">
                          <span className="text-gray-700">Level</span>
                          <span className="font-medium">{getCholesterolLabel(cholesterol)}</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-gray-500 mb-2">Glucose</h4>
                      <div className="bg-white p-4 rounded-lg border border-gray-200">
                        <div className="flex items-center justify-between">
                          <span className="text-gray-700">Level</span>
                          <span className="font-medium">{getGlucoseLabel(glucose)}</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-gray-500 mb-2">Body Mass Index (BMI)</h4>
                      <div className="bg-white p-4 rounded-lg border border-gray-200">
                        <div className="flex items-center justify-between">
                          <span className="text-gray-700">Your BMI</span>
                          <span className="font-medium">{bmi}</span>
                        </div>
                        <div className="mt-2">
                          <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div 
                              className={`h-2.5 rounded-full ${
                                bmi < 18.5 ? 'bg-blue-500' : 
                                bmi < 25 ? 'bg-green-500' : 
                                bmi < 30 ? 'bg-yellow-500' : 'bg-red-500'
                              }`} 
                              style={{ 
                                width: `${Math.min(100, (Math.min(40, parseFloat(bmi)) / 40) * 100)}%` 
                              }}
                            ></div>
                          </div>
                          <div className="flex justify-between text-xs text-gray-500 mt-1">
                            <span>Underweight</span>
                            <span>Normal</span>
                            <span>Overweight</span>
                            <span>Obese</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'recommendations' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Personalized Recommendations</h3>
                  <p className="text-gray-600 mb-6">Based on your assessment results, here are some recommendations to help improve your cardiovascular health:</p>
                  
                  <div className="space-y-4">
                    {recommendations.map((rec, index) => (
                      <div key={index} className="flex p-4 bg-gray-50 rounded-lg">
                        <div className="flex-shrink-0">
                          <div className="flex items-center justify-center h-10 w-10 rounded-full bg-white shadow-sm">
                            {rec.icon}
                          </div>
                        </div>
                        <div className="ml-4">
                          <h4 className="text-base font-medium text-gray-900">{rec.title}</h4>
                          <p className="mt-1 text-sm text-gray-600">{rec.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-blue-50 p-6 rounded-lg">
                  <h4 className="text-lg font-medium text-blue-800 mb-3">Next Steps</h4>
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-700">Schedule a follow-up with your healthcare provider to discuss these results</span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-700">Consider lifestyle changes based on the recommendations above</span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-700">Retake this assessment in 3-6 months to track your progress</span>
                    </li>
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex flex-col sm:flex-row justify-between items-center mt-8 space-y-4 sm:space-y-0">
          <button
            onClick={() => navigate('/predict')}
            className="inline-flex items-center px-6 py-3 border border-gray-300 shadow-sm text-base font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <ArrowLeft className="h-5 w-5 mr-2" />
            Back to Assessment
          </button>
          
          <div className="flex space-x-3">
            <button
              onClick={handlePrint}
              className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              <Printer className="h-4 w-4 mr-2" />
              Print
            </button>
            <button
              onClick={handleShare}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              <Share2 className="h-4 w-4 mr-2" />
              Share Results
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;
