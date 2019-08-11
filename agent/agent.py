from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random, os
from collections import deque

class Agent:
	def __init__(self, ticker, state_size, m_path):
		self.state_size = state_size # normalized previous days
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=1000) #後來的1000筆記憶
		self.inventory = []
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.checkpoint_path = m_path
		self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
		self.check_index = self.checkpoint_path + '.index'   #checkpoint裡面的檔案多加了一個.index
		self.model = self._model()

	def _model(self):
		self.optimizer = keras.optimizers.Adam(lr=0.001)
		model = keras.Sequential()
		model.add(keras.layers.Dense(units=64, input_dim=self.state_size, activation="relu"))
		model.add(keras.layers.Dense(units=32, activation="relu"))
		model.add(keras.layers.Dense(units=8, activation="relu"))
		model.add(keras.layers.Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=self.optimizer)
		#output為各action的機率(要轉換)
		if os.path.exists(self.check_index):
			#如果已經有訓練過，就接著load權重
			print('-'*32+'Weights loaded!!'+'-'*32)
			model.load_weights(self.checkpoint_path)
		else:
			print('-'*31+'Create new model!!'+'-'*31)
		
		return model

	def act(self, state):
		if os.path.exists(self.check_index):
			options = self.model.predict(state)
		else:
			return random.randrange(self.action_size)
		#起始是亂數，後來就是靠神經網路的預測結果決定
		
		return np.argmax(options[0]) #array裡面最大值的位置號

	def expReplay(self, batch_size): #用memory來訓練神經網路
		mini_batch = []
		l = len(self.memory)
		mini_batch = random.sample(self.memory, batch_size)
		#for i in range(l - batch_size + 1, l):
		#	mini_batch.append(self.memory[i]) #後面的memory會被拿出來
			
		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				#Deep Q Network的靈魂公式
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
				
			target_f = self.model.predict(state)
			target_f[0][action] = target
			#設檢查點
		
			cp_callback = tf.keras.callbacks.ModelCheckpoint(
			filepath=self.checkpoint_path,
			save_weights_only=True,
			verbose=0)
				
			self.model.fit(state, target_f, epochs=1,
			 verbose=0, callbacks = [cp_callback])

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay 
