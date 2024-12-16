import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf

# Load pre-trained model and encoders
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('ohe.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

# Streamlit UI
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0.0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode geography using OneHotEncoder
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine input data and encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Make predictions
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

# Display results
st.write('Churn Probability: {:.2f}'.format(prediction_prob))

if prediction_prob > 0.5:
    st.write('The Customer is Likely to Churn')
else:
    st.write('The Customer is Not Likely to Churn')




