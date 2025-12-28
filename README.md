# Telecom Customer Churn Prediction
Live App: https://telecom-customerchurn-prediction-iz6ue9xsctk3csvuqdr47i.streamlit.app/

This project predicts whether a telecom customer is likely to churn (leave the service) based on customer details, service usage, and billing information.  


## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Project Structure
- churn.csv – Dataset used for training
- Customer Churn Prediction.ipynb – Model training notebook
- churn_model_bundle.pkl – Trained model bundle
- lg_app.py – Streamlit web application
- requirements.txt – Project dependencies
- README.md – Project documentation

## How to Run the Project

1. Clone the repository  
-git clone <your-github-repo-url>

2. Navigate to the project folder  
-cd Customer-Churn-Prediction

3. Install dependencies  
-pip install -r requirements.txt

4. Run the Streamlit app  
-streamlit run lg_app.py

## Output
The app predicts whether a customer has:
- High Churn Risk
- Low Churn Risk

## Learning Outcome
- Built a classification model using Logistic Regression  
- Performed feature encoding and scaling  
- Saved and reused trained ML models  
- Built a Streamlit web app  
- Used Git and GitHub for version control
