
from agent.agent import Agent
from functions import *
import sys


ticker, window_size = 'AAPL', 10
#要給checkpoint個路徑
m_path = "models/{}/training.ckpt".format(ticker)
#丟給agent初始化
agent = Agent(ticker, window_size, m_path, is_eval=True)

#取得歷史資料，沒給時間就是從有資料到最近
data = []
df = pdr.DataReader('{}'.format(ticker),'yahoo')
for close in df['Close']:
	data.append(close)

l = len(data) - 1
batch_size = 32

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
		total_profit += data[t] - bought_price
		print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

	done = True if t == l - 1 else False
	state = next_state

	if done:
		print("-"*80)
		print("Trade " + ticker + "With Total Profit: " + formatPrice(total_profit))
		print("-"*80)
