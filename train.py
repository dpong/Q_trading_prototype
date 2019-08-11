import sys
from agent.agent import Agent
from functions import *

ticker, window_size, episode_count = 'AAPL', 10, 1
#要給checkpoint個路徑
m_path = "models/{}/training.ckpt".format(ticker)
#丟給agent初始化
agent = Agent(ticker, window_size, m_path)

#取得歷史資料，沒給時間就是從有資料到最近
data = []
df = pdr.DataReader('{}'.format(ticker),'yahoo',start='2018-1-1',end='2018-12-31')
for close in df['Close']:
	data.append(close)

l = len(data) - 1
batch_size = 32

for e in range(1, episode_count + 1):
	#print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			print("Buy : " + formatPrice(data[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price)+ " | Total Profit: " + formatPrice(total_profit))

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print("-"*80)
			print("Episode " + str(e) + "/" + str(episode_count)+ " | Total Profit: " + formatPrice(total_profit))
			print("-"*80)

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)
		
		#呈現進度
		#sys.stdout.write("\r"+"Episode " + str(e) + "/" + str(episode_count)+" Training: "+"%.2f%%" % round(t*(100/l),2))
		#sys.stdout.flush()
		

