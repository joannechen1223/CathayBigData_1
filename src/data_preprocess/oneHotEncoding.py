'''
Cathay Big Data
data preprocess
alphabat to digit and one hot encoding the nominal data
create date:2018.09.12
update date:2018.09.14
author:Joanne Chen
reference:

'''
import math
import random
import numpy as np
import pandas as pd
import keras as keras

def load_data():
	X = pd.read_csv("../../data/X_fillNA.csv")
	Y = pd.read_csv("../../data/Y_fillNA.csv")
	columnNames = np.array(list(X.head()))
	X = X.values
	Y = Y.values
	return X, Y, columnNames

def toDigit(data):
	data = data.reshape((data.shape[0],))
	digit = np.zeros((data.shape[0],))
	#ord() is to get a char's ascii code
	for i in range(0, data.shape[0]):
		if((ord(data[i])>=ord('a') and ord(data[i])<=ord('z'))):
			digit[i] = ord(data[i])-ord('a')
		elif((ord(data[i])>=ord('A') and ord(data[i])<=ord('Z'))):
			digit[i] = ord(data[i])-ord('A')
	digit = digit.reshape((data.shape[0],1))
	return digit

def oneHotEncoding(data):
	data = data.reshape((data.shape[0],))
	categories = int(np.amax(data)+1)
	encodedData = keras.utils.to_categorical(data, categories)
	return encodedData

def extractFirstChar(data):
	occ = []
	for i in range(0, data.shape[0]):
		occ.append(data[i][0])
	return np.array(occ)

def X_preprocess(data):
	# CUST_ID
	X = data[:, 0:1]
	# AGE
	X = np.concatenate((X, toDigit(data[:, 1])), axis=1)
	# SEX
	X = np.concatenate((X, toDigit(data[:, 2])), axis=1)
	# HEIGHT&WEIGHT
	X = np.concatenate((X, data[:, 3:5]), axis=1)
	# OCCUPATION
	occupation = extractFirstChar(data[:, 5])
	X = np.concatenate((X, oneHotEncoding(toDigit(occupation))), axis=1)
	# CHILD_NUM
	X = np.concatenate((X, data[:, 6:7]), axis=1)
	# BUY_MONTH
	buyMonth = data[:, 7]-1
	print(oneHotEncoding(buyMonth))
	X = np.concatenate((X, oneHotEncoding(buyMonth)), axis=1)
	# CITY_CODE
	print(np.unique(data[:, 9]))
	X = np.concatenate((X, oneHotEncoding(toDigit(data[:, 9]))), axis=1)
	# BUDGET
	X = np.concatenate((X, data[:, 10:11]), axis=1)
	# MARRIGE
	X = np.concatenate((X, oneHotEncoding(toDigit(data[:, 11]))), axis=1)
	# BEHAVIOR
	X = np.concatenate((X, oneHotEncoding(toDigit(data[:, 12])), oneHotEncoding(toDigit(data[:, 13])), oneHotEncoding(toDigit(data[:, 14]))), axis=1)
	# EDUCATION
	X = np.concatenate((X, toDigit(data[:, 15])), axis=1)
	# CHARGE_WAY
	X = np.concatenate((X, toDigit(data[:, 16])), axis=1)
	# IS_EMAIL
	X = np.concatenate((X, toDigit(data[:, 17])), axis=1)
	# IS_PHONE
	X = np.concatenate((X, toDigit(data[:, 18])), axis=1)
	# IS_APP
	X = np.concatenate((X, toDigit(data[:, 19])), axis=1)
	# IS_SPECIALMEMBER
	X = np.concatenate((X, toDigit(data[:, 20])), axis=1)
	# PARENTS_DEAD
	X = np.concatenate((X, toDigit(data[:, 21])), axis=1)
	# REAL_ESTATE_HAVE
	X = np.concatenate((X, toDigit(data[:, 22])), axis=1)
	# IS_MAJOR_INCOME
	X = np.concatenate((X, toDigit(data[:, 23])), axis=1)
	# BUY_TYP_NUM_CLASS
	for i in range(24, 31):
		X = np.concatenate((X, oneHotEncoding(toDigit(data[:, i]))), axis=1)
	print(X)
	print(X.shape)
	return X

def main():
	X, Y, X_col = load_data()
	# label
	Y = np.concatenate((Y[:, 0:1], toDigit(Y[:, 1:])), axis=1)
	Y_train = pd.DataFrame(Y, columns=['CUST_ID', 'BUY_TYPE'])
	Y_train.to_csv("../../data/Y_train.csv", index=None)
	
	# feature
	X = X_preprocess(X)
	X = pd.DataFrame(X)
	X = X.to_csv("../../data/X.csv", index=None)
	return



if __name__ == '__main__':
	main()