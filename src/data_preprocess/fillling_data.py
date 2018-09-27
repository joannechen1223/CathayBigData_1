'''
Cathay Big Data
data preprocess
1. kick outliers
2. concatenate features, split label
3. fill non availible data
create date:2018.09.07
update date:2018.09.14
author:Joanne Chen b04701232
reference:
https://www.kaggle.com/yogi045/preprocess-and-predicting-using-random-forest
'''
import math
import random
import numpy as np
import pandas as pd
from sklearn import linear_model

def load_data():
	# read original data
	train_buy_info = pd.read_csv("../../original_data/train_buy_info.csv")
	train_cust_info = pd.read_csv("../../original_data/train_cust_info.csv")
	train_tpy_info = pd.read_csv("../../original_data/train_tpy_info.csv")
	test_buy_info = pd.read_csv("../../original_data/test_buy_x_info.csv")
	test_cust_info = pd.read_csv("../../original_data/test_cust_x_info.csv")
	test_tpy_info = pd.read_csv("../../original_data/test_tpy_x_info.csv")

	# sort all data by 'CUST_ID' as a foreign key in order to concatenate data
	train_buy_info = train_buy_info.sort_values(by=['CUST_ID'])
	train_cust_info = train_cust_info.sort_values(by=['CUST_ID'])
	train_tpy_info = train_tpy_info.sort_values(by=['CUST_ID'])
	test_buy_info = test_buy_info.sort_values(by=['CUST_ID'])
	test_cust_info = test_cust_info.sort_values(by=['CUST_ID'])
	test_tpy_info = test_tpy_info.sort_values(by=['CUST_ID'])

	# count N/A datasets
	print("buy_info")
	print("--TRAIN--"+"--"*10)
	print(train_buy_info.isnull().sum())
	print("--TEST--"+"--"*10)
	print(test_buy_info.isnull().sum())
	print("---"*10)
	print("cust_info")
	print("--TRAIN--"+"--"*10)
	print(train_cust_info.isnull().sum())
	print("--TEST--"+"--"*10)
	print(test_cust_info.isnull().sum())
	print("---"*10)
	print("tpy_info")
	print("--TRAIN--"+"--"*10)
	print(train_tpy_info.isnull().sum())
	print("--TEST--"+"--"*10)
	print(test_tpy_info.isnull().sum())
	print("---"*10)



	# x column name
	buyColumnNames = list(test_buy_info.head())
	custColumnNames = list(test_cust_info.head())
	tpyColumnNames = list(test_tpy_info.head())
	columnNames = buyColumnNames + custColumnNames[1:4] + custColumnNames[8:9] + custColumnNames[10:13] + custColumnNames[23:] + tpyColumnNames[1:]
	columnNames = np.array(columnNames)
	#columnNames = np.reshape(columnNames, (1,columnNames.shape[0]))
	
	# generate y label
	Y = train_buy_info[['CUST_ID', 'BUY_TYPE']]
	Y.index = range(len(Y))
	
	# to np array
	train_buy_info = train_buy_info.values
	train_cust_info = train_cust_info.values
	train_tpy_info = train_tpy_info.values
	test_buy_info = test_buy_info.values
	test_cust_info = test_cust_info.values
	test_tpy_info = test_tpy_info.values

	# concatenate X datasets
	X_train = np.concatenate((train_buy_info[:,0:1], train_buy_info[:, 2:], train_cust_info[:,1:4], train_cust_info[:,8:9], train_cust_info[:,10:13], train_cust_info[:,23:],train_tpy_info[:,1:]), axis=1)
	X_test = np.concatenate((test_buy_info, test_cust_info[:,1:4], test_cust_info[:,8:9], test_cust_info[:,10:13], test_cust_info[:,23:],test_tpy_info[:,1:]), axis=1)
	X = np.concatenate((X_train, X_test), axis=0)
	X = pd.DataFrame(X, columns=columnNames)
	return X, Y

def kickOutliers(features, label):
	print(features)
	print(label)
	outliers = features.index[features["IS_APP"].isnull()].tolist()
	print(outliers)
	features = features.drop(outliers)
	label = label.drop(outliers)
	return features, label

def fillHeight(data, index):
	# split notnull data
	heightNotNull = data[data["HEIGHT"].notnull()]
	# count the averge and standard deviation group by age and sex
	avg = []
	std = []
	ageRange = pd.unique(heightNotNull["AGE"]).shape[0]
	for i in range(ageRange):
		avgInAgeRange = []
		stdInAgeRange = []
		for j in range(2):
			populations = heightNotNull[heightNotNull["AGE"] == chr(i+97)]
			populations = populations[populations["SEX"] == chr(j+97)]
			avgInAgeRange.append(populations["HEIGHT"].mean())
			stdInAgeRange.append(populations["HEIGHT"].std())
		avg.append(avgInAgeRange)
		std.append(stdInAgeRange)
	
	# fill in the non available data
	for i in index:
		if(math.isnan(data.loc[i,"HEIGHT"])):
			getAvg = avg[ord(data.loc[i, "AGE"])-97][ord(data.loc[i, "SEX"])-97]
			getStd = std[ord(data.loc[i, "AGE"])-97][ord(data.loc[i, "SEX"])-97]
			data.loc[i, "HEIGHT"] = random.uniform(getAvg - getStd, getAvg + getStd)
			
	return data

def fillWeight(data, index):
	# split nonnull data
	weightNotNull = data[data["WEIGHT"].notnull()]
	# use linear regression of height and weight to predict the non available weight
	height = np.array(weightNotNull["HEIGHT"]).reshape(-1, 1)
	weight = np.array(weightNotNull["WEIGHT"])
		#build the linear regression model
	lm = linear_model.LinearRegression()
	lm.fit(height, weight)
	print('Coeff of determination:',lm.score(height,weight))
	print('correlation is:',math.sqrt(lm.score(height,weight)))
		# fill in the non available data by predicting with linear regression model
	for i in index:
		if(math.isnan(data.loc[i,"WEIGHT"])):
			data.loc[i, "WEIGHT"] = float(lm.predict(np.array(data.loc[i,"HEIGHT"]).reshape(-1,1)))
	return data

def random_index(rate):
    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index


def randomByRate(data, index):
	notNull = list(data[data.notnull()])
	arr = ['a', 'b', 'c']
	rate = [notNull.count('a'), notNull.count('b'), notNull.count('c')]
	for i in index:
		if(data.loc[i]!=data.loc[i]):
			data.loc[i] = arr[random_index(rate)]
	return data

def fillBlank(data, index):
	data["BEHAVIOR_1"] = randomByRate(data["BEHAVIOR_1"], index)
	data["BEHAVIOR_2"] = randomByRate(data["BEHAVIOR_2"], index)
	data["BEHAVIOR_3"] = randomByRate(data["BEHAVIOR_3"], index)
	data["CHARGE_WAY"] = randomByRate(data["CHARGE_WAY"], index)
	data["IS_SPECIALMEMBER"] = randomByRate(data["IS_SPECIALMEMBER"], index)	
	return data

def toCsv(data, file_path):
	data.to_csv(file_path, index=None)
	return

def main():
	X, Y = load_data()
	index = X.index[X["IS_APP"].notnull()].tolist()
	
	X, Y = kickOutliers(X, Y)
	X = fillHeight(X, index)
	X = fillWeight(X, index)
	X = fillBlank(X, index)
	toCsv(X, "../../data/X_fillNA.csv")
	toCsv(Y, "../../data/Y_fillNA.csv")
	
	print(X.isnull().sum())
	
if __name__ == '__main__':
	main()