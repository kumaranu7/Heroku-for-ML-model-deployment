import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
data = pd.read_csv('deploy.csv')
y = data.iloc[:, -1]
X = data.iloc[:, 0:3]

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)

#saving model to disk
pickle.dump(model, open('model.pkl','wb'))  