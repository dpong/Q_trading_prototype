# Q_trading_prototype

啟發自：
https://github.com/edwardhdlu/q-trader
https://github.com/keon/deep-q-learning


目的：利用Q-learning來訓練交易機器人

版本：Tensorflow 2.0 beta

根據輸入的前幾天收盤價來訓練機器人的action。
機器人有三種action：買、賣、不動作。

2019/08/12 update:
再回去看了一下 Q-learning，原本的 action 決定不能一開始就吃 model 的預測，
要利用貪婪度去做冒險。然後 Q-learning 本身取得 reward 的時候會歸功到之前的
action 我上一次的結論明顯沒有了解 Q-table 的用途。

雖然如此，我對於原型機到底有沒有歸功採懷疑的態度，也因此參考了一些範例後，
修改在 expReplay 內的結構，讓邏輯更貼合 Q-learning。結果明顯的就好多了。
