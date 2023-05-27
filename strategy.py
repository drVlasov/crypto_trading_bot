import sqlalchemy
import pandas as pd
import numpy as np
import time
import statsmodels.api as sml
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier
import scipy
import scipy.stats as stats
from binance.client import Client
import requests
import pandas as pd

api_key = ''
api_secret = ''

client = Client(api_key, api_secret)

model1 = CatBoostClassifier()
model1.load_model("catboost_file5_59_rets.json")


### FUNCTIONS
# Tick bars
def get_tick_bars(prices: np.ndarray, vols: np.ndarray,
                  times: np.ndarray, freq: int) -> np.ndarray:
    bars = np.zeros(shape=(len(range(freq, len(prices), freq)), 6), dtype=object)
    ind = 0
    for i in range(freq, len(prices), freq):
        bars[ind][0] = pd.Timestamp(times[i - 1])  # time
        bars[ind][1] = prices[i - freq]  # open
        bars[ind][2] = np.max(prices[i - freq: i])  # high
        bars[ind][3] = np.min(prices[i - freq: i])  # low
        bars[ind][4] = prices[i - 1]  # close
        bars[ind][5] = np.sum(vols[i - freq: i])  # volume
        ind += 1
    return bars


# Get lagged "close" prices
def lagged(df, lookback, stride):
    df_list = []
    df_columns = []
    for i in range(0, lookback + stride, stride):
        shifted = df.shift(i)
        df_list.append(shifted)
        df_columns.append(str.format("Close_" + "lag_" + str(i)))
    s = pd.concat(df_list, axis=1)
    s.columns = df_columns
    return s


# get returns
def returns(df):
    df_list = []
    df_columns = []
    for i in range(len(df.columns) - 1):
        ret = np.log((df[df.columns[i]] / df[df.columns[i + 1]]).astype(float))  # -1
        df_list.append(ret)
        df_columns.append(str.format("ret_" + str(i)))
    k = pd.concat(df_list, axis=1)
    k.columns = df_columns
    return k


## trend - strong
## t -value of a trend through the lagged period - the sign of t-value - is the direction of a trend, the module of t-value - is the significance of the regression coefficient
def tValLinR(c):
    x = np.arange(0, c.shape[1])
    y = c.values[0][::-1].astype('float64')
    x = sml.add_constant(x)
    ols = sml.OLS(y, x).fit()
    return ols.tvalues[1]


lags = list(range(5, 21, 5)) + list(range(30, 51, 10))


def trend_value(s_cut):
    trends = []
    trends_inds = []
    for i in range(len(s_cut)):
        obs = s_cut.iloc[i:i + 1, :]
        trend_ind = s_cut.index[i]
        trends_inds.append(trend_ind)
        trends_sequence = []
        for j in lags:
            c = obs.iloc[:, :-j]
            trend = tValLinR(c)
            trends_sequence.append(trend)
        trends.append(trends_sequence)
    t_columns = [str.format("t_%-c_%d" % (str(0), i + 1)) for i in lags]
    out = pd.DataFrame(data=trends, index=trends_inds, columns=t_columns)
    return out


# volatilities of returns
def volatils_value(rets_cut):
    volatils = []
    vol_inds = []
    for i in range(len(rets_cut)):
        obs = rets_cut.iloc[i:i + 1, :]
        vol_ind = rets_cut.index[i]
        vol_inds.append(vol_ind)
        vol_sequence = []
        for j in lags:
            c = obs.iloc[:, :-j]
            volatil = c.std(axis=1)
            vol_sequence.append(float(volatil))
        volatils.append(vol_sequence)
    v_columns = [str.format("v_%-c_%d" % (str(0), i + 1)) for i in lags]
    out = pd.DataFrame(data=volatils, index=vol_inds, columns=v_columns)
    return out


### what percentile is Close0

def percentile_value(s_short, s_long):
    p_s = []
    p_s_inds = []
    for i in range(len(s_long)):
        p_short = stats.percentileofscore(s_short.iloc[i], s_short.iloc[i][0]) / 100
        p_long = stats.percentileofscore(s_long.iloc[i], s_long.iloc[i][0]) / 100
        p_s_ind = s_short.index[i]
        p_s_inds.append(p_s_ind)
        p_s.append([p_short, p_long])
    p_columns = ['perc_short', 'perc_long']
    out = pd.DataFrame(data=p_s, index=p_s_inds, columns=p_columns)
    return out


### get signal
# select last 300 ticks and make new tick bar
def get_signal(tick_bars_df, model1):
    sql_query = "SELECT * FROM BTCUSDT ORDER BY rowid DESC LIMIT 301"
    df1 = pd.read_sql(sql_query, engine)
    df1 = df1.iloc[::-1]
    df1['Datetime'] = pd.to_datetime(df1['Time'])
    df1 = df1.drop(['symbol', 'Time'], axis=1)
    tick_bars = get_tick_bars(df1['Price'].values, df1['Qty'].values, df1['Datetime'].values, 300)
    tick_bars_df1 = pd.DataFrame(data=tick_bars[:, 1:], index=tick_bars[:, 0],
                                 columns=['open', 'high', 'low', 'close', 'volume'])

    # stack tick bars (0) with tick_bars (1)
    tick_bars_df = pd.concat([tick_bars_df, tick_bars_df1], axis=0)
    tick_bars_df = tick_bars_df.iloc[1:, :]
    tick_bars_df_short = tick_bars_df.iloc[-51:, :]

    # extract close prices
    close = tick_bars_df["close"]
    price = close.iloc[-1]
    price_lag = close.iloc[-2]
    close = pd.DataFrame(data=close, index=close.index, columns=["close"])
    close_short = tick_bars_df_short["close"]
    close_short = pd.DataFrame(data=close_short, index=close_short.index, columns=["close"])
    # get lagged prices
    s_long = lagged(close, 10000, 200)
    s_short = lagged(close_short, 50, 1)
    s_long = s_long.dropna(axis=0)
    s_short = s_short.dropna(axis=0)

    # get lagged volumes
    volumes = tick_bars_df.volume
    volumes_short = tick_bars_df_short.volume
    volumes_lagged_short = lagged(volumes_short, 50, 1)
    volumes_lagged_long = lagged(volumes, 10000, 200)
    volumes_lagged_short = volumes_lagged_short.dropna(axis=0)
    volumes_lagged_long = volumes_lagged_long.dropna(axis=0)

    # Volume weighted Price - close price
    vwap_l = np.sum(volumes_lagged_long * s_long, axis=1) / np.sum(volumes_lagged_long, axis=1)
    vol_weighted_price_long = (vwap_l - s_long["Close_lag_0"])  # /vwap_l
    vol_weighted_price_long = pd.DataFrame(data=vol_weighted_price_long.values, index=vol_weighted_price_long.index,
                                           columns=["VWP_long"])

    vwap_s = np.sum(volumes_lagged_short * s_short, axis=1) / np.sum(volumes_lagged_short, axis=1)
    vol_weighted_price_short = (vwap_s - s_short["Close_lag_0"])  # /vwap_s
    vol_weighted_price_short = pd.DataFrame(data=vol_weighted_price_short.values, index=vol_weighted_price_short.index,
                                            columns=["VWP_short"])

    # get returns
    rets_short = returns(s_short)
    rets_long = returns(s_long)

    # get trends
    trend_long = trend_value(s_long)
    trend_long = trend_long.dropna(axis=1)
    trend_short = trend_value(s_short)
    trend_short = trend_short.dropna(axis=1)

    # get volatilities of returns
    volatils_short = volatils_value(rets_short)
    volatils_long = volatils_value(rets_long)
    volatils_long = volatils_long.dropna(axis=1)
    volatils_short = volatils_short.dropna(axis=1)

    # get percentiles
    p_s = percentile_value(s_short, s_long)
    p_s_rets = percentile_value(rets_short, rets_long)

    # final df
    frames = [trend_long, volatils_long, vol_weighted_price_long, rets_long,  # trend_rets_long, rets_buckets_long,
              trend_short, volatils_short, vol_weighted_price_short, rets_short, p_s,
              p_s_rets]  # trend_rets_short, rets_buckets_short, #, trend_rets_short, rets_buckets_short,

    col_names = []
    frames_str = ["l", "l", "l", "l", "s", "s", "s", "s", "_prices", "_rets"]
    for i in range(len(frames)):
        for j in frames[i].columns:
            j = str(j) + str(frames_str[i])
            col_names.append(j)
    data = pd.concat(frames, axis=1)
    data.columns = col_names

    signal = model1.predict(data)
    return signal, price, price_lag, tick_bars_df


# telgram send message
def telegram_bot_sendtext(bot_message):
    bot_token = ''
    bot_chatID = ''
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
    response = requests.get(send_text)
    return response.json()



# INITIAL SELECT
start_time = time.time()
engine = sqlalchemy.create_engine('sqlite:///BTCUSDTstream2.db')

sql_query = "SELECT * FROM BTCUSDT ORDER BY rowid DESC LIMIT 3000301"
df = pd.read_sql(sql_query, engine)
df = df.iloc[::-1]

sql_query = "SELECT MAX (rowid) FROM BTCUSDT"
df2 = pd.read_sql(sql_query, engine)
last_index = int(df2.iloc[0][0])

print(last_index)
df['Datetime'] = pd.to_datetime(df['Time'])
df = df.drop(['symbol', 'Time'], axis = 1)

# GET TICK BARS FOR INITIAL SELECT
tick_bars = get_tick_bars(df['Price'].values, df['Qty'].values, df['Datetime'].values, 300)
tick_bars_df = pd.DataFrame(data=tick_bars[:, 1:], index=tick_bars[:, 0], columns=['open', 'high', 'low', 'close', 'volume'])
print("--- %s seconds ---" % (time.time() - start_time))
tick_bars_df.tail(3)


def write_stats(stat_row):
    # TODO write to db
    print(stat_row)


def strategy(last_index, tick_bars_df, qty, s_l=0.5, t_p=0.005, alpha=0.95):
    strategy_return = []
    position = 0
    print(last_index)
    stat_row = {}
    while True:
        engine = sqlalchemy.create_engine('sqlite:///BTCUSDTstream2.db')
        sql_query = "SELECT MAX (rowid) FROM BTCUSDT"
        df2 = pd.read_sql(sql_query, engine)
        last_row = int(df2.iloc[0][0])
        # print(last_row)
        if last_row == (last_index + 300):
            start_time = time.time()
            last_index = last_row
            signal, price, price_lag, tick_bars_df = get_signal(tick_bars_df, model1)
            signal = signal[0]
            #signal = int(round(np.random.uniform(), 0))
            print(signal)
            # price1 = float(client.get_ticker(symbol = "BTCUSDT")['lastPrice'])
            print("current price", price, "lag price", price_lag)
            # print(price1)

            if (signal == 1) and (position == 0):
                qty_buy = qty
                order_buy = client.create_order(symbol='BTCUSDT',
                                                side="BUY",
                                                type="MARKET",
                                                quantity=qty_buy)
                position_buy = 0
                for i in order_buy['fills']:
                    position_buy += float(i['price']) * float(i["qty"])
                price_buy = position_buy / qty_buy
                ts_buy = pd.to_datetime(order_buy['transactTime'], unit='ms')
                print("BUY", " ", ts_buy, " ", qty, "Price ", price_buy)
                text = str("BUY ") + \
                       str(ts_buy) + \
                       str(" ") + str(round(qty_buy, 3)) + str(" Price ") + \
                       str(round(price_buy, 2))
                text = telegram_bot_sendtext(str(text))
                position = 1
                stop_loss_price = (1 - s_l * t_p) * price_buy
                print("SL = ", stop_loss_price)

                stat_row['ts_buy'] = ts_buy
                stat_row['position_buy'] = position_buy
                stat_row['price_buy'] = price_buy
                stat_row['qty_buy'] = qty_buy
                stat_row['stop_losses'] = [{'price': stop_loss_price, 'ts': ts_buy}]
            if position == 1:

                if price <= stop_loss_price:
                    qty_sell = 0.999 * qty_buy
                    order_sell = client.create_order(symbol='BTCUSDT',
                                                     side="SELL",
                                                     type="MARKET",
                                                     quantity=qty_sell)
                    position_sell = 0
                    for i in order_sell['fills']:
                        position_sell += float(i['price']) * float(i["qty"])
                    price_sell = position_sell / qty_sell
                    ts_sell = pd.to_datetime(order_sell['transactTime'], unit='ms')
                    print("SELL", " ", ts_sell, " ", qty_sell, "Price ",
                          price_sell)
                    text = str("SELL ") + \
                           str(ts_sell) + \
                           str(" ") + str(round(qty_sell, 3)) + str(" Price ") + \
                           str(round(price_sell, 2))
                    text = telegram_bot_sendtext(str(text))
                    position = 0
                    result = float(float(position_sell) - float(position_buy))

                    print("deal: ", result)
                    text = str("Result ") + str(round(result, 4))
                    text = telegram_bot_sendtext(str(text))

                    stat_row['ts_sell'] = ts_sell
                    stat_row['position_sell'] = position_sell
                    stat_row['price_sell'] = price_sell
                    stat_row['qty_sell'] = qty_sell
                    stat_row['result'] = result
                    write_stats(stat_row)
                    stat_row = {}

                if price > stop_loss_price:
                    if price > price_lag:
                        stop_loss_price = (1 - alpha) * price + alpha * stop_loss_price
                    else:
                        stop_loss_price = stop_loss_price
                    if stat_row['stop_losses']:
                        stat_row['stop_losses'].append({'price': stop_loss_price, 'ts': start_time})
                    else:
                        stat_row['stop_losses'] = [{'price': stop_loss_price, 'ts': start_time}]
                print("SL = ", stop_loss_price)

            print("--- %s seconds ---" % (time.time() - start_time))

strategy(last_index, tick_bars_df, qty=0.01, s_l = 0.5, t_p = 0.005, alpha = 0.95)
