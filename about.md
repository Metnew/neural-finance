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
