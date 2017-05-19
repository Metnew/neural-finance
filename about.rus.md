Идея: создание масштабной сети
Входящий вектор:
    - rolling_window (x10-15):
        - pseudo_log - std
        - log - std
        - close - std + centering or to range
        - average - std + centering - to range
        - std (либо высчитывать std уже нормализованных цен)
        - price_growth_percent (prev)
Выходящий вектор: рост цены в процентах на следующем временном промежутке
Необходимо:
    - код
    - инстанс
    - брокер
    - информер

Сеть:
    - LSTM
    - Dense
    - linear last layer

https://www.tradingview.com/chart/?symbol=BITSTAMP:BTCUSD
------------
https://quant.stackexchange.com/questions/20675/how-to-forecast-high-frequency-data :
Personal feedback:
From my own research based on hft data I can write that generally only few models are better than naive forecasting (but only in some sub time series - few minutes etc.)

Kalman Filtering (calibrated accordingly)
ARIMA, ARMA models
Generally worse than naive forecasting are:

Holts linear model (ARIMA (0,2,2)
Linear Regression
Exponential Smoothing
Moving averages (different types)

------------
Online algorithm for selecting smoothing parameter?
https://quant.stackexchange.com/questions/16565/online-algorithm-for-selecting-smoothing-parameter
------------
https://quant.stackexchange.com/questions/16954/hft-architecture
------------
https://quant.stackexchange.com/questions/1274/how-can-we-reverse-engineer-a-market-making-algorithm-hft
http://www.math.nyu.edu/faculty/avellane/HighFrequencyTrading.pdf
------------
https://quant.stackexchange.com/questions/310/how-high-is-the-frequency-in-hft
http://www.dbresearch.com/PROD/DBR_INTERNET_EN-PROD/PROD0000000000269468.pdf
------------
According to the **International Financial Reporting Standards (IFRS)**, a financial asset can be:

Cash or cash equivalent,
Equity instruments of another entity,
Contractual right to receive cash or another financial asset from another entity or to exchange financial assets or financial liabilities with another entity under conditions that are potentially favourable to the entity,
A contract that will or may be settled in the entity's own equity instruments and is either a non-derivative for which the entity is or may be obliged to receive a variable number of the entity's own equity instruments, or a derivative that will or may be settled other than by exchange of a fixed amount of cash or another financial asset for a fixed number of the entity's own equity instruments.
------------
------------
------------
------------
------------
------------
------------
------------
------------
