import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
import base64





# Function to load data and train the model with cross-validation
def load_and_train_model():
    # Load the dataset
    data = pd.read_csv('data/titanic.csv')
    
    # Preprocess the data
    data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    data = data.dropna()
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    data['Embarked'] = label_encoder.fit_transform(data['Embarked'])
    
    # Define features and target
    X = data.drop(columns=['Survived'])
    y = data['Survived']
    
    # Train the model with cross-validation
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Perform cross-validated predictions
    predictions = cross_val_predict(model, X, y, cv=5, method='predict_proba')
    
    # Save the model
    model.fit(X, y)  # Fit on the whole dataset
    joblib.dump((model, predictions, y), 'titanic_survival_model.pkl')
    
    return model, predictions, y


# Initialize session state if not already done
if 'generated_passengers' not in st.session_state:
    st.session_state.generated_passengers = {}

# Load the model
model, predictions, y = load_and_train_model()

# ******************************************************************************************************************
# Load the CSS file
# Main title
st.markdown('<h1 class="main-title">Titanic Survival Predictor</h1>', unsafe_allow_html=True)

# Inputs
st.markdown('<div class="input-container"><label class="input-names">Sex</label>', unsafe_allow_html=True)
sex = st.selectbox("", ["male", "female"], key="sex")

st.markdown('<div class="input-container"><label class="input-names">Age</label>', unsafe_allow_html=True)
age = st.number_input("", min_value=0, max_value=100, value=30, key="age")

st.markdown('<div class="input-container"><label class="input-names">Passenger class</label>', unsafe_allow_html=True)
pclass = st.selectbox("", [1, 2, 3], key="pclass")

st.markdown('<div class="input-container"><label class="input-names">Number of Siblings/Spouses Aboard</label>', unsafe_allow_html=True)
sibsp = st.number_input("", min_value=0, max_value=10, key="sibsp")

st.markdown('<div class="input-container"><label class="input-names">Number of Parents/Children Aboard</label>', unsafe_allow_html=True)
parch = st.number_input("", min_value=0, max_value=10, key="parch")

st.markdown('<div class="input-container"><label class="input-names">Passenger Fare</label>', unsafe_allow_html=True)
fare = st.number_input("", min_value=0.0, value=20.0, key="fare")

st.markdown('<div class="input-container"><label class="input-names">Port of Embarkation</label>', unsafe_allow_html=True)
embarked = st.selectbox("", ["C", "Q", "S"], key="embarked")

# Load CSS file at the end to prevent whitespace issues
with open("styles/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)




# ******************************************************************************************************************


# Preprocess the inputs to match the training data
sex = 1 if sex == "male" else 0
embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

# Create a DataFrame for the input
input_data = pd.DataFrame(
    [[pclass, sex, age, sibsp, parch, fare, embarked]],
    columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
)

# Convert the input data to a tuple (hashable type) for storing in session state
input_tuple = tuple(input_data.iloc[0])

# Predict the survival
if st.button("Predict"):
    if input_tuple in st.session_state.generated_passengers:
        prediction_proba = st.session_state.generated_passengers[input_tuple]
    else:
        prediction_proba = model.predict_proba(input_data)[0]  # Get the prediction probabilities
        st.session_state.generated_passengers[input_tuple] = prediction_proba

    if prediction_proba[1] > 0.5:
        st.markdown(
            f'<div class="success-message">The passenger that you generated would have survived. '
            f'A probability of the prediction is {prediction_proba[1]*100:.2f}%.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="error-message">The passenger that you generated would not have survived. '
            f'A probability of the prediction is {prediction_proba[0]*100:.2f}%.</div>',
            unsafe_allow_html=True
        )


def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExOXVzcjM3a2pwOW0yMmVhc3ZhZ3BqaTNkeTBtMnVtc2M3dnk2eWNoNCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/QWudnPjdeCSqJfAs29/giphy.gif");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    

set_bg_hack_url() 