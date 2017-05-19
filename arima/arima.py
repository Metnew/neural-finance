from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from matplotlib import pyplot
import numpy as np
# from fbprophet import Prophet


def parser(x):
    return datetime.utcfromtimestamp(float(x)).strftime('%Y-%m-%dT%H:%M:%SZ')

df = read_csv("./data/GSPC.csv", header=0, parse_dates=[5], index_col=False, squeeze=True, date_parser=parser)
df['close_price'] = np.log(df['close_price'])
df['open_price'] = np.log(df['open_price'])
df['low_price'] = np.log(df['low_price'])
df['high_price'] = np.log(df['high_price'])
dff = DataFrame({"y": df["close_price"] - df["close_price"].mean(), "ds": df["timestamp"]})
print(dff.head())
# m = Prophet()
# m.fit(dff)
# future = m.make_future_dataframe(periods=365)
# future.tail()
# fit model

# # plot residual errors
df.plot()
# series.show()

# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# pyplot.show()
# residuals.plot(kind='kde')
# pyplot.show()
# print(residuals.describe())
