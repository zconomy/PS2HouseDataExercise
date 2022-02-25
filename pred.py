import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import linear_model

#import data
data = pd.read_csv('train.csv')

#kick out all categorical independent variables
numeric = data.select_dtypes(include=[np.number])
corr = numeric.corr()
cols = corr['SalePrice'].sort_values(ascending=False)[:7].index
print(cols)
# print(data['SalePrice'].describe())
# plt.hist(data['SalePrice'])
# plt.show()
data_for_model = data[cols]

#split to training and test data sets
np.random.seed(65535)
msk = np.random.rand(len(data_for_model)) < 0.8
data_train = data_for_model[msk]
data_test = data_for_model[~msk]

price_train = data_train['SalePrice']
data_train = data_train.drop(['SalePrice'], axis=1)

price_test = data_test['SalePrice']
data_test = data_test.drop(['SalePrice'], axis=1)

#build linear model with logorithm dependent variable
#poor model with NO categorical variables
lr = linear_model.LinearRegression()
model = lr.fit(data_train, np.log(price_train))
print(f"R^2 is: {model.score(data_test,np.log(price_test))}")

#generate prediction for the test.csv file, kick out categorical
test_csv_input = pd.read_csv('test.csv')
data_for_model_test_csv = test_csv_input[cols.drop('SalePrice')]
#back to salesprice with exp()
pred_test_csv = model.predict(data_for_model_test_csv)
pred_price = np.exp(pred_test_csv)

#output csv
pred_output_df = pd.DataFrame(pred_price, columns=['SalesPrice'])
pred_output_df.insert(0,'Id', test_csv_input['Id'])
# print(pred_output_df)
pred_output_df.to_csv('ps2_pred.csv', index=-False)