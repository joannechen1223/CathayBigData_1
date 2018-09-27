'''
Cathay Big Data
Xgboost
create date:2018.09.15
update date:2018.09.17
author:Joanne Chen b04701232
reference:
https://www.kaggle.com/babatee/intro-xgboost-classification
'''
import math
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def load_data():
	X = pd.read_csv("../../data/X.csv")
	Y = pd.read_csv("../../data/Y_train.csv")
	X = X.values
	Y = Y.values
	CustId = X[:,0:1]
	X = X[:,1:]
	y = Y[:,1:]

	X, X_test = X[:-10000], X[-10000:]
	trainCustId, testCustId = CustId[:-10000], CustId[-10000:]
	y = y.reshape((Y.shape[0],))
	y_pred = pd.read_csv("../../original_data/Submmit_Sample_testing_Set.csv")
	
	return X_test, X, y, y_pred, testCustId

def fillPredictValue(pred, y_pred):
	pred = pred.set_index("CUST_ID")
	for i in y_pred.index:
		index = y_pred.loc[i, "CUST_ID"]
		y_pred.loc[i, "BUY_TYPE"] = chr(pred.loc[index, "BUY_TYPE"]+ord('a'))

	return y_pred

def toCsv(data, file_path):
	data.to_csv(file_path, index=None)
	return

def main():
	X_test, X, y, y_pred, testCustId = load_data()
	model = xgb.XGBClassifier(max_depth=10)
	train_model = model.fit(X, y)
	pred = train_model.predict(X_test)
	pred = pred.reshape((pred.shape[0], 1))
	testCustId = testCustId.reshape((testCustId.shape[0], 1))
	pred = np.concatenate((testCustId, pred), axis=1)
	pred = pred.astype(int)
	pred = pd.DataFrame(pred, columns=["CUST_ID", "BUY_TYPE"])
	toCsv(pred, "../../data/try.csv")
	y_pred = fillPredictValue(pred, y_pred)
	toCsv(y_pred, "../../data/ans.csv")
	return


if __name__ == '__main__':
	main()