import streamlit as st
import pandas as pd

import pandas as pd
import numpy as np

st.write("""
# Churn Prediction App
choose your parameters from the sidebar and know if customer will **churn**
""")


st.sidebar.header('User Input Parameters ')

def user_input_features():
    age =st.sidebar.slider('Age of customer',5,200,50)
    Total_Purchase =st.sidebar.slider('Total_Purchase of customer',0,20000,10000)
    Account_Manager =st.sidebar.slider('Account_Manager of customer',0,1,0)
    Years =st.sidebar.slider('Years of customer',0,15,5)
    Num_Sites =st.sidebar.slider('Num_Sites of customer',1,20,8)

    u_data={'age':age,
            'Total_Purchase':Total_Purchase,
            'Account_Manager':Account_Manager,
            'Years':Years,
            'Num_Sites':Num_Sites
            }
    feature=pd.DataFrame(u_data,index=['valus'])
    return feature

df=user_input_features()

st.subheader('Predicted Parameters')
st.write(df)
    
customers_data = pd.read_csv('customer_churn.csv')
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

# Scale only the input data
input_data = customers_data[['Age','Total_Purchase','Account_Manager', 'Years', 'Num_Sites']].to_numpy()
data_scaled = scaler.fit_transform(input_data)
data_scaled_df = pd.DataFrame (data_scaled, columns = ['Age','Total_Purchase','Account_Manager', 'Years', 'Num_Sites'])

X = data_scaled_df
y = customers_data['Churn']

from sklearn.ensemble import AdaBoostClassifier
ada_model = AdaBoostClassifier(n_estimators=100,random_state=0)   # n_estimators=100, max_depth=10, random_state = 0)
ada_model.fit(X, y)

u_value=scaler.transform(df)
pred= ada_model.predict(u_value)
pred_prob= ada_model.predict_proba(u_value)

st.subheader('Probability Display')
st.write(pd.DataFrame({'won\'t churn':pred_prob[0][0],'churn':pred_prob[0][1]},index=['probability']))

classes={0:'won\'t churn',1:'churn'}
st.subheader('Predicted Action')
st.write('**{}**'.format(classes[pred[0]]))

