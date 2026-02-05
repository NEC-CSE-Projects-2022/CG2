import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { HeartPulse, ArrowLeft, ArrowRight, Info, User, Activity } from 'lucide-react';
import RiskMeter from '../components/RiskMeter';
import { toast } from 'react-hot-toast';

const formSections = [
  {
    title: 'Personal Information',
    icon: <User className="h-5 w-5" />,
    description: 'Tell us a bit about yourself to get started.',
    fields: [
      { 
        name: 'age', 
        label: 'Age', 
        type: 'number', 
        min: 18, 
        max: 100,
        placeholder: 'Enter your age',
        required: true,
        validation: (value) => (value >= 18 && value <= 100) || 'Age must be between 18 and 100'
      },
      {
        name: 'gender',
        label: 'Gender',
        type: 'select',
        options: [
          { value: '1', label: 'Male' },
          { value: '2', label: 'Female' }
        ],
        placeholder: 'Select your gender',
        required: true
      },
      { 
        name: 'height', 
        label: 'Height (cm)', 
        type: 'number', 
        min: 100, 
        max: 250,
        placeholder: 'Enter height in cm',
        required: true,
        validation: (value) => (value >= 100 && value <= 250) || 'Height must be between 100cm and 250cm'
      },
      { 
        name: 'weight', 
        label: 'Weight (kg)', 
        type: 'number', 
        min: 30, 
        max: 200, 
        step: 0.1,
        placeholder: 'Enter weight in kg',
        required: true,
        validation: (value) => (value >= 30 && value <= 200) || 'Weight must be between 30kg and 200kg'
      }
    ]
  },
  {
    title: 'Health Metrics',
    icon: <Activity className="h-5 w-5" />,
    description: 'Please provide your health information.',
    fields: [
      { 
        name: 'ap_hi', 
        label: 'Systolic Blood Pressure (mmHg)', 
        type: 'number', 
        min: 90, 
        max: 200,
        placeholder: 'e.g., 120',
        required: true,
        validation: (value) => (value >= 90 && value <= 200) || 'Please enter a valid systolic pressure'
      },
      { 
        name: 'ap_lo', 
        label: 'Diastolic Blood Pressure (mmHg)', 
        type: 'number', 
        min: 60, 
        max: 120,
        placeholder: 'e.g., 80',
        required: true,
        validation: (value) => (value >= 60 && value <= 120) || 'Please enter a valid diastolic pressure'
      },
      {
        name: 'cholesterol',
        label: 'Cholesterol Level',
        type: 'select',
        options: [
          { value: '1', label: 'Normal' },
          { value: '2', label: 'Above Normal' },
          { value: '3', label: 'Well Above Normal' }
        ],
        placeholder: 'Select cholesterol level',
        required: true
      },
      {
        name: 'glucose',
        label: 'Glucose Level',
        type: 'select',
        options: [
          { value: '1', label: 'Normal' },
          { value: '2', label: 'Above Normal' },
          { value: '3', label: 'Well Above Normal' }
        ],
        placeholder: 'Select glucose level',
        required: true
      }
    ]
  },
  {
    title: 'Lifestyle Factors',
    icon: <HeartPulse className="h-5 w-5" />,
    description: 'Tell us about your lifestyle habits.',
    fields: [
      {
        name: 'smoke',
        label: 'Do you smoke?',
        type: 'radio',
        options: [
          { value: '1', label: 'Yes' },
          { value: '0', label: 'No' }
        ],
        required: true
      },
      {
        name: 'alco',
        label: 'Do you consume alcohol?',
        type: 'radio',
        options: [
          { value: '1', label: 'Yes' },
          { value: '0', label: 'No' }
        ],
        required: true
      },
      {
        name: 'active',
        label: 'Are you physically active?',
        type: 'radio',
        options: [
          { value: '1', label: 'Yes' },
          { value: '0', label: 'No' }
        ],
        required: true
      }
    ]
  }
];

const Predict = ({ setPredictionResult }) => {
  const [currentSection, setCurrentSection] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    height: '',
    weight: '',
    ap_hi: '',
    ap_lo: '',
    cholesterol: '',
    glucose: '',
    smoke: '',
    alco: '',
    active: ''
  });

  const navigate = useNavigate();
  const currentSectionData = formSections[currentSection];
  const isLastSection = currentSection === formSections.length - 1;

  // Reset errors when section changes
  useEffect(() => {
    setErrors({});
  }, [currentSection]);

  const validateField = (name, value) => {
    const field = formSections
      .flatMap(section => section.fields)
      .find(field => field.name === name);
    
    if (!field) return '';
    
    if (field.required && !value) {
      return 'This field is required';
    }
    
    if (field.validation) {
      const validationResult = field.validation(value);
      if (typeof validationResult === 'string') {
        return validationResult;
      }
    }
    
    return '';
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const fieldValue = type === 'radio' ? value : (type === 'checkbox' ? checked : value);
    
    setFormData(prev => ({
      ...prev,
      [name]: fieldValue
    }));
    
    // Validate on change only if the field has been touched
    if (touched[name]) {
      const error = validateField(name, fieldValue);
      setErrors(prev => ({
        ...prev,
        [name]: error
      }));
    }
  };

  const handleBlur = (e) => {
    const { name, value } = e.target;
    setTouched(prev => ({ ...prev, [name]: true }));
    
    const error = validateField(name, value);
    setErrors(prev => ({
      ...prev,
      [name]: error
    }));
  };

  const validateSection = () => {
    const currentSectionFields = formSections[currentSection].fields;
    const newErrors = {};
    let isValid = true;
    
    currentSectionFields.forEach(field => {
      const error = validateField(field.name, formData[field.name]);
      if (error) {
        newErrors[field.name] = error;
        isValid = false;
      }
    });
    
    setErrors(newErrors);
    return isValid;
  };

  const nextSection = () => {
    if (validateSection()) {
      setCurrentSection(prev => Math.min(prev + 1, formSections.length - 1));
    }
  };

  const prevSection = () => {
    setCurrentSection(prev => Math.max(prev - 1, 0));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateSection()) {
      toast.error('Please fill in all required fields correctly');
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      // Prepare data for API with proper type conversion
      const requestData = {
        ...formData,
        // Convert string numbers to integers where needed
        age: parseInt(formData.age, 10),
        height: parseFloat(formData.height),
        weight: parseFloat(formData.weight),
        ap_hi: parseInt(formData.ap_hi, 10),
        ap_lo: parseInt(formData.ap_lo, 10),
        cholesterol: parseInt(formData.cholesterol, 10),
        glucose: parseInt(formData.glucose, 10),
        // Ensure boolean values are properly converted to numbers (1 or 0)
        smoke: formData.smoke === '1' || formData.smoke === 1 ? 1 : 0,
        alco: formData.alco === '1' || formData.alco === 1 ? 1 : 0,
        active: formData.active === '1' || formData.active === 1 ? 1 : 0
      };

      // Call your prediction API here
      // const response = await fetch('/api/predict', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(requestData)
      // });
      // const result = await response.json();
      
      // For now, simulate API call with timeout
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Calculate BMI
      const heightInMeters = requestData.height / 100;
      const bmi = requestData.weight / (heightInMeters * heightInMeters);
      
      // Calculate individual risk factors (0-100 scale)
      const ageRisk = Math.min(30, (requestData.age - 30) * 0.8); // 0-30 points (ages 30-90)
      const bmiRisk = bmi > 35 ? 25 : bmi > 30 ? 20 : bmi > 25 ? 10 : 0;
      const cholesterolRisk = (requestData.cholesterol - 1) * 15; // 0-30 points
      const smokeRisk = requestData.smoke === 1 ? 35 : 0; // Smoking is a major risk factor
      const activityRisk = requestData.active === 0 ? 25 : 0;
      const bpRisk = requestData.ap_hi > 160 || requestData.ap_lo > 100 ? 40 : 
                    requestData.ap_hi > 140 || requestData.ap_lo > 90 ? 25 : 0;
      const glucoseRisk = (requestData.glucose - 1) * 12; // 0-24 points
      
      // Calculate base risk with more emphasis on critical factors
      let riskScore = (
        ageRisk * 1.5 +     // Age is a significant factor
        bmiRisk * 1.2 +     // Weight status impacts risk
        cholesterolRisk * 1.3 + // Cholesterol is important
        smokeRisk * 2.0 +    // Smoking is a major risk factor
        activityRisk * 1.2 + // Physical activity level
        bpRisk * 1.7 +       // Blood pressure is critical
        glucoseRisk * 1.3    // Glucose levels are important
      ) / 1.8; // Adjusted normalization factor for better distribution
      
      // Ensure score is within 0-100 range with minimum of 1%
      riskScore = Math.min(100, Math.max(1, Math.round(riskScore)));
      
      console.log('Risk factors:', {
        ageRisk, bmiRisk, cholesterolRisk, smokeRisk, activityRisk, bpRisk, glucoseRisk, riskScore
      });
      
      const mockResponse = {
        riskScore: riskScore.toFixed(1),
        recommendations: []
      };
      
      // Navigate to results with the prediction data
      const resultData = {
        age: requestData.age,
        gender: requestData.gender,
        height: requestData.height,
        weight: requestData.weight,
        ap_hi: requestData.ap_hi,
        ap_lo: requestData.ap_lo,
        cholesterol: requestData.cholesterol.toString(), // Ensure string for switch statements
        glucose: requestData.glucose.toString(),         // Ensure string for switch statements
        smoke: requestData.smoke,
        alco: requestData.alco,
        active: requestData.active,
        bmi: bmi.toFixed(1), // Add BMI to results
        riskScore: parseFloat(riskScore.toFixed(1)),
        timestamp: new Date().toISOString()
      };
      
      // Update the parent state
      setPredictionResult(resultData);
      
      // Navigate with the result data in the location state
      navigate('/results', { state: resultData });
      
    } catch (error) {
      console.error('Prediction error:', error);
      toast.error('Failed to process your request. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderField = (field) => {
    const { name, label, type, options, placeholder, min, max, step, info } = field;
    const error = errors[name];
    const value = formData[name];
    const isTouched = touched[name];
    const showError = error && isTouched;
    
    return (
      <div key={name} className="mb-6">
        <label htmlFor={name} className="block text-sm font-medium text-gray-700 mb-1">
          {label} {field.required && <span className="text-red-500">*</span>}
        </label>
        
        {type === 'select' ? (
          <select
            id={name}
            name={name}
            value={value}
            onChange={handleChange}
            onBlur={handleBlur}
            className={`mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm ${
              showError ? 'border-red-300' : 'border-gray-300'
            }`}
          >
            <option value="">{placeholder || 'Select an option'}</option>
            {options.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        ) : type === 'radio' ? (
          <div className="mt-2 space-y-2">
            {options.map(option => (
              <div key={option.value} className="flex items-center">
                <input
                  id={`${name}-${option.value}`}
                  name={name}
                  type="radio"
                  value={option.value}
                  checked={value === option.value}
                  onChange={handleChange}
                  onBlur={handleBlur}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                />
                <label htmlFor={`${name}-${option.value}`} className="ml-3 block text-sm font-medium text-gray-700">
                  {option.label}
                </label>
              </div>
            ))}
          </div>
        ) : (
          <div className="relative rounded-md shadow-sm">
            <input
              type={type}
              id={name}
              name={name}
              value={value}
              onChange={handleChange}
              onBlur={handleBlur}
              min={min}
              max={max}
              step={step}
              placeholder={placeholder}
              className={`block w-full rounded-md ${showError ? 'border-red-300' : 'border-gray-300'} pr-10 focus:border-blue-500 focus:ring-blue-500 sm:text-sm`}
            />
          </div>
        )}
        
        {showError && (
          <p className="mt-1 text-sm text-red-600">{error}</p>
        )}
        
        {info && !showError && (
          <p className="mt-1 text-xs text-gray-500">{info}</p>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white py-8 sm:py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-center text-gray-900 mb-8">Cardiovascular Risk Assessment</h1>
        
        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between mb-2">
            {formSections.map((section, index) => (
              <div key={index} className="flex flex-col items-center flex-1">
                <button
                  onClick={() => setCurrentSection(index)}
                  className={`flex items-center justify-center w-10 h-10 rounded-full ${
                    currentSection === index
                      ? 'bg-blue-600 text-white'
                      : index < currentSection
                      ? 'bg-green-100 text-green-600'
                      : 'bg-gray-200 text-gray-600'
                  }`}
                >
                  {index + 1}
                </button>
                <span className="mt-2 text-xs text-center text-gray-600">{section.title}</span>
              </div>
            ))}
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-600 transition-all duration-300"
              style={{
                width: `${((currentSection + 1) / formSections.length) * 100}%`,
              }}
            />
          </div>
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={currentSection}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="bg-white rounded-xl shadow-lg overflow-hidden"
          >
            {/* Form Header */}
            <div className="bg-gradient-to-r from-blue-600 to-blue-800 px-6 py-8 text-center sm:px-10">
              <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 mb-4">
                {React.cloneElement(currentSectionData.icon, { className: 'h-6 w-6 text-blue-600' })}
              </div>
              <h1 className="text-2xl font-bold text-white">
                {currentSectionData.title}
              </h1>
              <p className="mt-2 text-blue-100">
                {currentSectionData.description}
              </p>
            </div>
            
            {/* Form Content */}
            <div className="px-6 py-8 sm:px-10">
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 gap-6">
                  {currentSectionData.fields.map(field => renderField(field))}
                </div>
                
                {/* Navigation Buttons */}
                <div className="flex justify-between pt-6 border-t border-gray-200">
                  <button
                    type="button"
                    onClick={prevSection}
                    disabled={currentSection === 0 || isSubmitting}
                    className={`inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm ${
                      currentSection === 0
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                        : 'bg-white text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    <ArrowLeft className="h-4 w-4 mr-2" />
                    Previous
                  </button>
                  
                  {!isLastSection ? (
                    <button
                      type="button"
                      onClick={nextSection}
                      className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                    >
                      Next
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </button>
                  ) : (
                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isSubmitting ? (
                        <>
                          <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Processing...
                        </>
                      ) : (
                        'Get My Risk Assessment'
                      )}
                    </button>
                  )}
                </div>
              </form>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Predict;
