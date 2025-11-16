# CO2 Emission Prediction Using Machine Learning and Deep Learning

This project implements an end-to-end machine learning pipeline to predict vehicle CO2 emissions using multiple regression models and a deep learning neural network. The workflow includes preprocessing, feature engineering, model training, evaluation, and deployment via a Gradio web application.

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- TensorFlow / Keras
- Gradio

## Project Workflow
1. Data Cleaning and Preprocessing
2. Feature Engineering
3. Model Training (ML and DL Models)
4. Model Evaluation and Comparison
5. Deployment using Gradio

## Results
- Best performing ensemble model achieved R² ≈ 0.997  
- Deep learning model achieved R² ≈ 0.95  
- Random Forest model performed the strongest overall

## How to Run the Application

### Install dependencies
```
pip install -r requirements.txt
```

### Run the Gradio application
```
python app.py
```

This will launch a browser interface to input vehicle details and generate CO2 emission predictions.

## Repository Structure
```
Carbon_Accounting_Source_code.ipynb   # Full ML and DL workflow
app.py                                # Gradio deployment script
requirements.txt                       # Required packages
README.md                              # Project documentation
```

## Future Improvements
- Additional hyperparameter tuning for neural networks
- Deployment on cloud platforms (HuggingFace Spaces, Streamlit Cloud)
- Testing additional models such as LightGBM and CatBoost