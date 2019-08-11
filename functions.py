import numpy as np
import math
import pandas_datareader as pdr
import pandas as pd
import random

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the sigmoid
def sigmoid(x):
	#好像可以增加學習效率
	return 1 / (1 + math.exp(-x))

# returns an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	#資料量大於window前的處理
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))
	#state是window內，每天跟前一天差值取sigmoid
	return np.array([res])


