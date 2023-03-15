import pandas as pd
import numpy as np
import gzip
import dill

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

with gzip.open('churn_model.dill.gz', 'wb') as f:
    model =dill.dump(ada_model,f,recurse=True)

with gzip.open('rescale.dill.gz', 'wb') as f:
    model =dill.dump(scaler,f,recurse=True)
