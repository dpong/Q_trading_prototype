# Q_trading_prototype

啟發自：https://github.com/edwardhdlu/q-trader

目的：利用Q-learning來訓練交易機器人

版本：Tensorflow 2.0 beta

根據輸入的前幾天收盤價來訓練機器人的action。
機器人有三種action：買、賣、不動作。
原型機的問題就是賣出股票後的收益當作reward，買、不動作的reward都是0。
訓練後的結果就會是只賣...

如圖所示：
![image](https://github.com/dpong/Q_trading_prototype/blob/master/fig_01.png)


接下來要針對這個問題來解決一下，讓機器人的行為更符合現實。

