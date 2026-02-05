# CG2 – Hybrid Resampling-Driven Deep Learning Architecture for Cardiovascular Risk Stratification


## Team Info
- **22471A05E2 — Thogati Adi Lakshmi** ([LinkedIn](https://www.linkedin.com/in/adi-lakshmi-thogati-211648276))  
  _Work Done:_ Data preprocessing, feature engineering, TabNet implementation, SHAP analysis.

- **22471A05J8 — Sunkara Harini** ([LinkedIn](https://www.linkedin.com/in/harini2213))  
  _Work Done:_ CNN–BiLSTM and Attention-LSTM model development, training, and evaluation.

- **22471A05I7 — Sattenapalli Sadarani** ([LinkedIn](https://www.linkedin.com/in/sadarani-sattenapalli-779151282))  
  _Work Done:_ MLP and 1D-CNN implementation, performance comparison, visualization.

---

## Abstract
cardiovascular diseases (CVD) are a significant health issue around the world, which gives the reason why an accurate and interpretable predictive system should be developed. This is a study that proposes a deep learning method that considers a compound resampling technique to consider data skew and enhance the risk prediction of cardiovascular disease. The framework contains five neural network models: TabNet, Attention-LSTM, a mix of CNN and BiLSTM, multilayer perceptron (MLP), and 1D CNN. SHAP explainability was applied to the TabNet, The MLP, and the CNN+BiLSTM models to increase the explained Ness of the model, and thus it revealed many new things regarding feature prominence. TabNet has been the model to perform the best out of all models that have been tested with an accuracy of 96.19%, but it provided good results on F1 and AUC as well. Albeit the results of the Attention-LSTM and 1D CNN performing slightly worse, the MLP and CNN+BiLSTM models also gave comparative results. Each of the models was trained on a common preprocessing pipeline. The findings indicate that aggregating multiple models of deep learning with focused explainability and balanced data strategies are capable of generating predictive tools of cardiovascular risks that are reliable and interpretable. This approach has a possibility to become a part of clinical support systems.

---

## Paper Reference (Inspiration)
👉 **Improving Cardiovascular Disease Prediction With Deep Learning and Correlation-Aware SMOTE**  
Original IEEE / conference research paper used as inspiration for the methodology.

---

## Our Improvement Over Existing Paper
This work improves upon existing studies by incorporating advanced deep learning architectures, hybrid resampling using SMOTEENN, mutual information–based feature selection, and SHAP-based explainability for all major models. Unlike prior approaches, this project focuses on both high accuracy and interpretability, which is essential for healthcare applications.

---

## About the Project
**What the project does:**  
Predicts the presence of cardiovascular disease based on patient clinical attributes such as age, blood pressure, BMI, cholesterol, glucose, and lifestyle indicators.

**Why it is useful:**  
Early detection of cardiovascular disease helps clinicians take preventive actions and improves patient outcomes through data-driven decision support.

**General Project Workflow:**  
Input → Data Cleaning → Feature Engineering → SMOTEENN Balancing → Deep Learning Models → SHAP Explainability → Prediction Output

---

## Dataset Used
**[Cardiovascular Disease Dataset (Kaggle – Sulianova)](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)**

**Dataset Details:**  
- Records: ~70,000 patient samples  
- Features: Age, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol intake, physical activity  
- Target variable: `cardio` (0 – No disease, 1 – Disease)

---

## Dependencies Used
- Python  
- pandas  
- numpy  
- scikit-learn  
- imbalanced-learn  
- PyTorch  
- pytorch-tabnet  
- SHAP  
- matplotlib  
- seaborn  

---

## EDA & Preprocessing
- Removal of physiologically invalid records  
- Outlier filtering using domain-based thresholds  
- Feature engineering:
  - Age (years)
  - Body Mass Index (BMI)
  - Pulse pressure  
- Standard scaling of numerical features  
- Class imbalance handled using SMOTEENN  
- Feature selection using Mutual Information  

---

## Model Training Info
The following models were trained independently:
- TabNet Classifier  
- CNN + BiLSTM  
- Multilayer Perceptron (MLP)  
- Attention-based LSTM  
- 1D Convolutional Neural Network  

Training setup:
- Train-test split: 80–20 (stratified)  
- Optimizer: Adam  
- Loss function: Binary Cross-Entropy  
- Epochs: 20–200 (model-dependent)  

---

## Model Testing / Evaluation
Evaluation metrics used:
- Accuracy  
- F1 Score  
- ROC-AUC  
- Confusion Matrix  
- Classification Report  

Model performance comparison plots were generated for analysis.

---

## Results
- TabNet achieved the best balance of accuracy and interpretability  
- CNN-BiLSTM and Attention-LSTM captured complex feature relationships effectively  
- SHAP analysis identified age, blood pressure, BMI, and cholesterol as key risk factors  
- All deep learning models achieved accuracy above 90%

---

## Limitations & Future Work
- Dataset limited to structured clinical attributes  
- No ECG or imaging data included  

Future improvements:
- Larger multi-hospital datasets  
- Multi-modal learning (ECG + clinical data)  
- Real-time clinical deployment  
- Ensemble learning for further performance gains  

---

## Deployment Info
- Models saved using `torch.save()`  
- Scaler and feature selector stored using `joblib`  
- Can be deployed as a Flask-based REST API or integrated into clinical decision-support systems

---
