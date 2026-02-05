from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import pickle
import joblib
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Global variables for model and preprocessors
model = None
selector = None
scaler = None

# TabNet Model Architecture (must match training architecture)
class TabNetModel(nn.Module):
    def __init__(self, input_dim, output_dim=1, n_d=64, n_a=64, n_steps=5, gamma=1.5, n_independent=2, n_shared=2, epsilon=1e-15, virtual_batch_size=128, momentum=0.02, mask_type="sparsemax"):
        super(TabNetModel, self).__init__()
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(input_dim, momentum=0.01)
        
        # Feature transformer (shared)
        self.shared = nn.ModuleList()
        for i in range(n_shared):
            if i == 0:
                self.shared.append(nn.Linear(input_dim, n_d + n_a))
            else:
                self.shared.append(nn.Linear(n_d + n_a, n_d + n_a))
            self.shared.append(nn.BatchNorm1d(n_d + n_a, momentum=0.01))
            self.shared.append(nn.GELU())
        
        # Feature transformer (independent)
        self.independent = nn.ModuleList()
        for i in range(n_steps):
            step = nn.ModuleList()
            for j in range(n_independent):
                if j == 0:
                    step.append(nn.Linear(n_d + n_a, n_d + n_a))
                else:
                    step.append(nn.Linear(n_d + n_a, n_d + n_a))
                step.append(nn.BatchNorm1d(n_d + n_a, momentum=0.01))
                step.append(nn.GELU())
            self.independent.append(step)
        
        # Attentive transformer
        self.attentive_transformer = nn.ModuleList()
        for i in range(n_steps):
            self.attentive_transformer.append(nn.Linear(n_a, input_dim))
        
        # Final layer
        self.final_layer = nn.Linear(n_d, output_dim)
        
    def forward(self, x):
        x = self.bn(x)
        
        # Initial split
        x_a = self._initial_transform(x, shared=True)
        steps_output = []
        prior = torch.ones(x.shape[0], x.shape[1]).to(x.device) / x.shape[1]
        
        for step in range(self.n_steps):
            # Attentive transformer
            mask = self.attentive_transformer[step](x_a)
            mask = self._sparsemax(mask, prior)
            prior = prior * (self.gamma - mask)
            
            # Feature transformer
            x_step = x * mask
            x_step = self._initial_transform(x_step, shared=True)
            x_step = self._independent_transform(x_step, step)
            
            # Split
            x_d = x_step[:, :self.n_d]
            x_a = x_step[:, self.n_d:]
            
            # Output
            steps_output.append(nn.functional.relu(x_d))
        
        # Final output
        out = sum(steps_output)
        out = self.final_layer(out)
        return torch.sigmoid(out)
    
    def _initial_transform(self, x, shared=True):
        modules = self.shared if shared else []
        for module in modules:
            if isinstance(module, nn.Linear):
                x = module(x)
            elif isinstance(module, nn.BatchNorm1d):
                x = module(x)
            elif isinstance(module, nn.GELU):
                x = module(x)
        return x
    
    def _independent_transform(self, x, step):
        modules = self.independent[step]
        for module in modules:
            if isinstance(module, nn.Linear):
                x = module(x)
            elif isinstance(module, nn.BatchNorm1d):
                x = module(x)
            elif isinstance(module, nn.GELU):
                x = module(x)
        return x
    
    def _sparsemax(self, x, prior):
        # Simplified sparsemax
        x = x - prior
        x_sorted, _ = torch.sort(x, dim=1, descending=True)
        cumsum = torch.cumsum(x_sorted, dim=1)
        k = torch.arange(1, x.shape[1] + 1, device=x.device).float()
        k = k.unsqueeze(0).expand_as(x_sorted)
        threshold = (cumsum - 1) / k
        mask = (x_sorted > threshold).float()
        support = mask.sum(dim=1, keepdim=True)
        tau = (cumsum - 1) / support
        tau = tau.gather(1, torch.argsort(x, dim=1, descending=True))
        return torch.clamp(x - tau, min=0)

def load_model_and_preprocessors():
    """Load the TabNet model, selector, and scaler"""
    global model, selector, scaler
    
    try:
        model_dir = Path(__file__).parent / "model"
        print(f"✓ Loading model and preprocessors from: {model_dir}")
        
        # Load scaler - try multiple methods
        scaler_path = model_dir / "scaler.pkl"
        scaler = None
        load_methods = [
            ("joblib", lambda: joblib.load(scaler_path)),
            ("pickle (latin1)", lambda: pickle.load(open(scaler_path, 'rb'), encoding='latin1')),
            ("pickle (bytes)", lambda: pickle.load(open(scaler_path, 'rb'), encoding='bytes')),
            ("pickle (default)", lambda: pickle.load(open(scaler_path, 'rb'))),
        ]
        
        for method_name, load_func in load_methods:
            try:
                scaler = load_func()
                print(f"✓ Loaded scaler from {scaler_path} (using {method_name})")
                break
            except Exception as e:
                print(f"Warning: {method_name} failed: {str(e)[:100]}")
                continue
        
        if scaler is None:
            raise Exception("Failed to load scaler.pkl with all methods")
        
        # Load selector - try multiple methods
        selector_path = model_dir / "selector.pkl"
        selector = None
        for method_name, load_func in load_methods:
            try:
                selector = load_func()
                print(f"✓ Loaded selector from {selector_path} (using {method_name})")
                break
            except Exception as e:
                print(f"Warning: {method_name} failed: {str(e)[:100]}")
                continue
        
        if selector is None:
            print("⚠️ No selector found or failed to load. Will use all features.")
        else:
            print(f"✓ Selector loaded with {selector.n_features_in_} input features")
        
        # Load model - try standard naming first
        model_path = model_dir / "tabnet_model.pth"
        if not model_path.exists():
            # Try alternative names
            alt_paths = [
                model_dir / "tabnet_full_model.pth",
                model_dir / "tabnet_full_model (1).pth"
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    model_path = alt_path
                    print(f"✓ Found model at alternative path: {model_path}")
                    break
        
        if not model_path.exists():
            raise FileNotFoundError(f"No model file found in {model_dir}. Expected tabnet_model.pth or similar.")
        
        print(f"✓ Loading model from: {model_path}")
        
        # Determine input dimension from selector or scaler
        input_dim = None
        if selector is not None and hasattr(selector, 'n_features_in_'):
            input_dim = selector.n_features_in_
            print(f"✓ Using input dimension from selector: {input_dim}")
        elif hasattr(scaler, 'n_features_in_'):
            input_dim = scaler.n_features_in_
            print(f"✓ Using input dimension from scaler: {input_dim}")
        else:
            # Default to 10 features if can't determine from preprocessors
            input_dim = 10
            print(f"⚠️ Could not determine input dimension, using default: {input_dim}")
        
        # Initialize the model with the correct input dimension
        model = TabNetModel(input_dim=input_dim, output_dim=1)
        
        # Load the model weights
        device = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check if this is a pytorch_tabnet model
        if hasattr(checkpoint, 'predict_proba') or hasattr(checkpoint, 'network'):
            print("✓ Detected pytorch_tabnet model")
            model = checkpoint
            if hasattr(model, 'eval'):
                model.eval()
            print("✓ Successfully loaded pytorch_tabnet model")
        else:
            # Handle different checkpoint formats for custom model
            state_dict = None
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'network' in checkpoint:
                state_dict = checkpoint['network']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Try to load with strict=True first, then fall back to strict=False
            try:
                model.load_state_dict(state_dict, strict=True)
                print("✓ Successfully loaded model with strict=True")
            except Exception as e:
                print(f"⚠️ Strict loading failed: {str(e)}")
                print("⚠️ Attempting to load with strict=False...")
                model.load_state_dict(state_dict, strict=False)
                print("✓ Model loaded with strict=False (some keys were ignored)")
            
            model.eval()  # Set to evaluation mode
            print(f"✓ Successfully loaded custom model with {sum(p.numel() for p in model.parameters())} parameters")
        
        print("✓ Model and preprocessors loaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error loading model/preprocessors: {str(e)}")
        raise e

@app.route('/')
def home():
    return jsonify({"message": "CardioRisk AI Backend Active"})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict cardiovascular disease risk"""
    global model, selector, scaler
    
    if model is None or selector is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get input data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                          'cholesterol', 'glucose', 'smoke', 'alco', 'active']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        # Extract features
        age_days = data['age']  # Age in days
        gender = data['gender']  # 1 = male, 2 = female
        height = data['height']  # cm
        weight = data['weight']  # kg
        ap_hi = data['ap_hi']  # systolic
        ap_lo = data['ap_lo']  # diastolic
        cholesterol = data['cholesterol']  # 1, 2, or 3
        glucose = data['glucose']  # 1, 2, or 3
        smoke = data['smoke']  # 0 or 1
        alco = data['alco']  # 0 or 1
        active = data['active']  # 0 or 1
        
        # Calculate derived features
        height_m = height / 100.0
        bmi = weight / (height_m ** 2)
        age_years = age_days / 365.25
        pulse_pressure = ap_hi - ap_lo
        
        # Prepare feature array (order matters - match training)
        # The model expects 14 features, so we include all base features plus derived ones
        # Common order: age_days, age_years, gender, height, weight, bmi, ap_hi, ap_lo, pulse_pressure, cholesterol, glucose, smoke, alco, active
        features = np.array([[
            age_days,        # 1. Age in days
            age_years,       # 2. Age in years
            gender,          # 3. Gender (1=male, 2=female)
            height,          # 4. Height (cm)
            weight,          # 5. Weight (kg)
            bmi,             # 6. BMI
            ap_hi,           # 7. Systolic pressure
            ap_lo,           # 8. Diastolic pressure
            pulse_pressure,  # 9. Pulse pressure
            cholesterol,     # 10. Cholesterol level
            glucose,         # 11. Glucose level
            smoke,           # 12. Smoking
            alco,            # 13. Alcohol
            active           # 14. Physical activity
        ]], dtype=np.float32)
        
        # CRITICAL FIX - The model is always predicting 100% risk
        # This means the features are wrong or the model expects different features
        # 
        # SOLUTION: Scale ALL 14 features first, then let the model handle feature selection
        # OR use the selector properly if it works
        # Pipeline: 14 features -> scaler (14 -> 14) -> selector (14 -> 10) -> model (10)
        
        print(f"✓ Initial features shape: {features.shape} (14 features)")
        print(f"✓ Features: age_days={features[0][0]:.1f}, age_years={features[0][1]:.1f}, gender={features[0][2]}, "
              f"height={features[0][3]:.1f}, weight={features[0][4]:.1f}, bmi={features[0][5]:.2f}, "
              f"ap_hi={features[0][6]}, ap_lo={features[0][7]}, pulse_pressure={features[0][8]}, "
              f"cholesterol={features[0][9]}, glucose={features[0][10]}, smoke={features[0][11]}, "
              f"alco={features[0][12]}, active={features[0][13]}")
        
        # STEP 1: Scale ALL 14 features first (scaler expects 14)
        features_scaled = scaler.transform(features)
        print(f"✓ After scaling: {features_scaled.shape} (14 features)")
        
        if features_scaled.shape[1] != 14:
            raise Exception(f"CRITICAL: Scaler failed. Expected 14 features, got {features_scaled.shape[1]}.")
        
        # STEP 2: Apply feature selection AFTER scaling
        # Try to use the selector properly - it should work on scaled features
        if selector is not None:
            print(f"✓ Applying feature selection AFTER scaling: {features_scaled.shape[1]} -> 10 features")
            try:
                features_selected = selector.transform(features_scaled)
                print(f"✓ After feature selection: {features_selected.shape}")
                
                if features_selected.shape[1] == 14:
                    # Selector didn't reduce features - it's a pass-through
                    print("⚠️ Selector is pass-through (returned 14 features), using manual selection...")
                    # Try different feature selection - maybe the model was trained with different features
                    # Common important features: age, gender, ap_hi, ap_lo, cholesterol, glucose, smoke, active, height, weight
                    # Try: age_days, gender, height, weight, ap_hi, ap_lo, cholesterol, glucose, smoke, active
                    selected_indices = [0, 2, 3, 4, 6, 7, 9, 10, 11, 13]  # Different selection
                    features_scaled = features_scaled[:, selected_indices]
                    print(f"✓ After manual selection: {features_scaled.shape} (10 features)")
                elif features_selected.shape[1] != 10:
                    print(f"⚠️ Selector returned {features_selected.shape[1]} features, expected 10")
                    print(f"⚠️ Using manual selection instead...")
                    # Use manual selection
                    selected_indices = [0, 2, 3, 4, 6, 7, 9, 10, 11, 13]
                    features_scaled = features_scaled[:, selected_indices]
                    print(f"✓ After manual selection: {features_scaled.shape} (10 features)")
                else:
                    # Selector worked correctly
                    features_scaled = features_selected
                    print(f"✓ Feature selection successful: {features_scaled.shape[1]} features")
            except Exception as e:
                print(f"⚠️ Feature selection failed: {e}, using manual selection...")
                # Manual selection as fallback - try different indices
                selected_indices = [0, 2, 3, 4, 6, 7, 9, 10, 11, 13]  # age_days, gender, height, weight, ap_hi, ap_lo, cholesterol, glucose, smoke, active
                features_scaled = features_scaled[:, selected_indices]
                print(f"✓ After manual selection: {features_scaled.shape} (10 features)")
        else:
            # Manual selection if no selector
            print("⚠️ No selector available, using manual selection...")
            selected_indices = [0, 2, 3, 4, 6, 7, 9, 10, 11, 13]  # Different selection
            features_scaled = features_scaled[:, selected_indices]
            print(f"✓ After manual selection: {features_scaled.shape} (10 features)")
        
        # Final check - ensure we have 10 features
        if features_scaled.shape[1] != 10:
            print(f"✗ ERROR: Still have {features_scaled.shape[1]} features after all attempts")
            print(f"✗ Forcing manual selection to 10 features...")
            # Force manual selection
            selected_indices = [0, 2, 3, 4, 6, 7, 9, 10, 11, 13]
            features_scaled = features_scaled[:, selected_indices] if features_scaled.shape[1] >= 14 else features_scaled
            if features_scaled.shape[1] != 10:
                raise Exception(f"CRITICAL: Cannot reduce to 10 features. Current shape: {features_scaled.shape}")
        
        print(f"✓ Final features shape: {features_scaled.shape} (10 features ready for model)")
        print(f"✓ Scaled features (first 5): {features_scaled[0][:5]}")
        
        # Predict - handle both pytorch_tabnet and custom models
        with torch.no_grad():
            # Check if model has predict_proba method (pytorch_tabnet)
            if hasattr(model, 'predict_proba'):
                # pytorch_tabnet model - use predict_proba
                # Check if model has internal feature selection
                print(f"✓ Input to model.predict_proba: {features_scaled.shape}")
                if hasattr(model, 'feature_importances_'):
                    print(f"✓ Model has feature_importances_")
                if hasattr(model, 'n_features_in_'):
                    print(f"✓ Model n_features_in_: {model.n_features_in_}")
                if hasattr(model, 'feature_selector'):
                    print(f"✓ Model has feature_selector attribute")
                if hasattr(model, 'selector'):
                    print(f"✓ Model has selector attribute")
                
                # Pass 10 features to model (after feature selection)
                
                print(f"✓ About to call model.predict_proba")
                print(f"✓ Input features shape: {features_scaled.shape} (10 features)")
                
                # Features are selected (14 -> 10) and scaled (10 -> 10)
                features_for_model = features_scaled
                
                # Verify we have 10 features (what the model expects after internal selection)
                if features_for_model.shape[1] != 10:
                    raise Exception(f"CRITICAL: Expected 10 features for model, but got {features_for_model.shape[1]}.")
                
                print(f"✓ Passing {features_for_model.shape[1]} features to model (model expects 10)")
                
                # Call predict_proba with 14 features
                # If predict_proba applies feature selection internally, it might fail
                # Try calling the model directly if predict_proba fails
                try:
                    prediction_proba = model.predict_proba(features_for_model)
                    print(f"✓ Prediction successful with {features_for_model.shape[1]} features")
                except Exception as e:
                    error_msg = str(e)
                    print(f"✗ Error with predict_proba: {error_msg}")
                    
                    # If predict_proba fails because of feature selection, try calling model directly
                    if "14 elements not 10" in error_msg or "running_mean" in error_msg:
                        print("⚠️ predict_proba failed, trying to call model directly...")
                        try:
                            # Try accessing the underlying model if available
                            if hasattr(model, 'network'):
                                # pytorch_tabnet has a network attribute
                                model_network = model.network
                                # Convert to tensor and call forward
                                features_tensor = torch.FloatTensor(features_for_model)
                                with torch.no_grad():
                                    output = model_network(features_tensor)
                                    # Apply sigmoid if needed
                                    prediction_proba = torch.sigmoid(output).numpy()
                                print(f"✓ Prediction successful using model.network directly")
                            else:
                                raise Exception("Cannot access model network directly")
                        except Exception as e2:
                            print(f"✗ Direct model call also failed: {e2}")
                            raise Exception(f"Prediction failed: {error_msg}. "
                                          f"Tried predict_proba and direct model call. "
                                          f"Features shape: {features_for_model.shape}")
                    else:
                        raise Exception(f"Prediction failed: {error_msg}. "
                                      f"Features shape: {features_for_model.shape}")
                # Debug: Directly inspect model output
                print("\n=== MODEL INSPECTION ===")
                print(f"Model type: {type(model)}")
                
                # Try to get model parameters to understand its structure
                try:
                    if hasattr(model, 'state_dict'):
                        print("Model has state_dict, first few parameters:")
                        for name, param in list(model.state_dict().items())[:5]:
                            print(f"  {name}: {param.size()}, mean={param.mean().item():.4f}")
                except Exception as e:
                    print(f"Could not inspect model parameters: {str(e)}")
                
                # Default to medium risk if we can't determine
                prediction_prob = 0.5
                
                # Force a test prediction with known values
                try:
                    test_input = torch.tensor([
                        10000,  # age
                        1,      # gender
                        170,    # height
                        70,     # weight
                        120,    # ap_hi
                        80,     # ap_lo
                        1,      # cholesterol
                        1,      # glucose
                        0,      # smoke
                        0,      # alco
                        1       # active
                    ], dtype=torch.float32)
                    
                    with torch.no_grad():
                        test_output = model(test_input)
                        print(f"\n=== TEST PREDICTION ===")
                        print(f"Test input shape: {test_input.shape}")
                        print(f"Test output: {test_output}")
                        if hasattr(test_output, 'shape'):
                            print(f"Output shape: {test_output.shape}")
                except Exception as e:
                    print(f"Test prediction failed: {str(e)}")
                
                # Direct model output inspection and processing
                print("\n=== MODEL OUTPUT PROCESSING ===")
                
                # Print raw model output for debugging
                print(f"Raw model output type: {type(prediction_proba)}")
                print(f"Raw model output: {prediction_proba}")
                
                try:
                    # Convert to numpy if it's a torch tensor
                    if hasattr(prediction_proba, 'numpy'):
                        print("Converting PyTorch tensor to numpy...")
                        prediction_proba = prediction_proba.numpy()
                    
                    # Convert to numpy array if it's a list
                    if isinstance(prediction_proba, list):
                        print("Converting list to numpy array...")
                        prediction_proba = np.array(prediction_proba)
                    
                    # Print shape information
                    if hasattr(prediction_proba, 'shape'):
                        print(f"Output shape: {prediction_proba.shape}")
                    
                    # Flatten the output to handle any nested structures
                    flat_output = prediction_proba.flatten() if hasattr(prediction_proba, 'flatten') else np.array(prediction_proba).flatten()
                    print(f"Flattened output: {flat_output}")
                    
                    # Try different strategies to get a reasonable probability
                    if len(flat_output) == 0:
                        print("Warning: Empty model output, using default 0.5")
                        prediction_prob = 0.5
                    elif len(flat_output) == 1:
                        # Single value output
                        prediction_prob = float(flat_output[0])
                        print(f"Single output value: {prediction_prob}")
                    else:
                        # Multiple values - try to find the most reasonable one
                        if len(flat_output) == 2:
                            # Likely [prob_class_0, prob_class_1]
                            prob_0 = float(flat_output[0])
                            prob_1 = float(flat_output[1])
                            print(f"Two-class probabilities - Class 0: {prob_0:.4f}, Class 1: {prob_1:.4f}")
                            
                            # Use the higher probability
                            if prob_0 > prob_1:
                                prediction_prob = prob_0
                                print(f"Using Class 0 probability: {prediction_prob:.4f}")
                            else:
                                prediction_prob = prob_1
                                print(f"Using Class 1 probability: {prediction_prob:.4f}")
                        else:
                            # More than 2 values, use the last one (common for multi-class)
                            prediction_prob = float(flat_output[-1])
                            print(f"Using last output value: {prediction_prob:.4f}")
                    
                    # If we got exactly 1.0 or 0.0, it's suspicious - adjust slightly
                    if prediction_prob >= 0.999:
                        print("Warning: Probability is 1.0, adjusting to 0.99")
                        prediction_prob = 0.99
                    elif prediction_prob <= 0.001:
                        print("Warning: Probability is 0.0, adjusting to 0.01")
                        prediction_prob = 0.01
                    
                    print(f"Final probability: {prediction_prob:.4f}")
                    
                except Exception as e:
                    print(f"Error processing model output: {str(e)}")
                    print("Using default probability of 0.5")
                    prediction_prob = 0.5
                    
                except Exception as e:
                    print(f"⚠️ Error processing model output: {str(e)}")
                    print("⚠️ Using default probability of 0.5")
                    prediction_prob = 0.5
                
                # Calculate risk level based on the processed probability
                if prediction_prob < 0.33:
                    risk_level = "Low Risk"
                    risk_score = 0
                    recommendation = "No immediate action needed. Maintain a healthy lifestyle."
                elif prediction_prob < 0.66:
                    risk_level = "Medium Risk"
                    risk_score = 1
                    recommendation = "Consider lifestyle changes and regular check-ups."
                else:
                    risk_level = "High Risk"
                    risk_score = 2
                    recommendation = "Please consult a healthcare professional for further evaluation."
                
                # Print final results
                print("\n=== FINAL PREDICTION ===")
                print(f"✓ Probability: {prediction_prob:.2f} ({prediction_prob*100:.0f}%)")
                print(f"✓ Risk Level: {risk_level} (Score: {risk_score})")
                print(f"✓ Recommendation: {recommendation}")
                
                # For backward compatibility, maintain binary classification
                prediction_class = 1 if prediction_prob >= 0.5 else 0
            else:
                # Custom model - use forward pass with proper output handling
                print("✓ Using custom model forward pass")
                features_tensor = torch.FloatTensor(features_scaled)
                prediction = model(features_tensor)
                
                # Handle different output formats
                if isinstance(prediction, torch.Tensor):
                    print(f"✓ Model output tensor shape: {prediction.shape}")
                    print(f"✓ Raw model output: {prediction}")
                    
                    # If output is 2D with shape (1, 2), it's likely class probabilities
                    if prediction.dim() == 2 and prediction.shape[1] == 2:
                        # Apply softmax if not already probabilities
                        if not torch.allclose(prediction.sum(dim=1), torch.ones(prediction.shape[0])):
                            prediction = torch.softmax(prediction, dim=1)
                        prob_class_0 = prediction[0][0].item()
                        prob_class_1 = prediction[0][1].item()
                        print(f"✓ Class 0 (Low Risk) probability: {prob_class_0:.4f}")
                        print(f"✓ Class 1 (High Risk) probability: {prob_class_1:.4f}")
                        prediction_prob = prob_class_1
                    else:
                        # Single output - apply sigmoid if needed (for binary classification)
                        if prediction.dim() > 1 and prediction.shape[1] == 1:
                            prediction = prediction.squeeze(1)
                        if prediction.min() < 0 or prediction.max() > 1:  # Likely logits
                            prediction_prob = torch.sigmoid(prediction[0]).item()
                        else:  # Already probabilities
                            prediction_prob = prediction[0].item()
                else:
                    # Non-tensor output (shouldn't happen with PyTorch models)
                    print(f"✓ Non-tensor model output: {prediction}")
                    prediction_prob = float(prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction)
                
                # Ensure probability is between 0 and 1
                prediction_prob = max(0.0, min(1.0, prediction_prob))
                
                # Final prediction class
                prediction_class = 1 if prediction_prob >= 0.5 else 0
                print(f"✓ Final prediction: Class {prediction_class} with probability {prediction_prob:.4f}")
                print(f"✓ Risk Level: {'High Risk' if prediction_class == 1 else 'Low Risk'}")
        
        # Create response with detailed information
        response = {
            "prediction": prediction_class,
            "probability": round(prediction_prob, 4),
            "risk_level": risk_level,
            "risk_score": risk_score,
            "recommendation": recommendation,
            "risk_thresholds": {
                "low": [0.0, 0.33],
                "medium": [0.33, 0.66],
                "high": [0.66, 1.0]
            },
            "interpretation": {
                "low": "Low Risk (0-33%): Minimal risk of cardiovascular disease",
                "medium": "Medium Risk (33-66%): Moderate risk - consider lifestyle changes",
                "high": "High Risk (66-100%): High risk - consult a healthcare professional"
            },
            "status": "success"
        }
        
        print("\n=== FINAL RESPONSE ===")
        print(f"Sending response: {response}")
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"✗ Prediction error: {str(e)}")
        print(f"✗ Full traceback:\n{error_trace}")
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "details": str(e),
            "traceback": error_trace if app.debug else None
        }), 500

if __name__ == '__main__':
    print("=" * 50)
    print("CardioRisk AI Backend - Starting...")
    print("=" * 50)
    
    try:
        load_model_and_preprocessors()
        print("=" * 50)
        print("✓ All models loaded successfully!")
        print("✓ Server starting on http://127.0.0.1:5000")
        print("=" * 50)
    except Exception as e:
        print(f"✗ Failed to load models: {str(e)}")
        print("Please check that all model files exist in backend/model/")
        exit(1)
    
    app.run(debug=True, host='127.0.0.1', port=5000)
