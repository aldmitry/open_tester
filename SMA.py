import MySQLdb
import MetaTrader5 as mt5
import time
import os
import datetime
import pandas as pd
import numpy as np
import pytz
import json
import time
import timer
from tqdm import tqdm
import itertools
import talib
import pickle
import multiprocessing
from sklearn.preprocessing import MinMaxScaler
from metatrader_commands import *
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)
import warnings
warnings.filterwarnings("ignore")

class Position:
    use_comission = True

    def __init__(self, enter_index, enter_time, enter_price, position_type, comission = 7):
        self.enter_price = enter_price
        self.enter_index = enter_index
        self.enter_time = enter_time
        self.position_type = position_type
        self.close_price = None
        self.close_time = None
        self.close_index = None
        self.comission = comission

    def set_close_position(self, indx, time, price):
        self.close_time = time
        self.close_price = price
        self.close_index = indx

    def calculate_profit(self):
        if self.use_comission:
            return self.position_type * (self.close_price - self.enter_price) - self.comission
        else:
            return self.position_type * (self.close_price - self.enter_price)

    def get_position_type(self):
        return self.position_type

    def get_close_price(self):
        return self.close_price

    def get_close_time(self):
        return self.close_time

    def get_enter_time(self):
        return self.enter_time

    def get_enter_index(self):
        return self.enter_index

    def get_close_index(self):
        return self.close_index

    def set_close_index(self, indx):
        self.close_index = indx

class StatReport:
    def __init__(self, GO,  params):

        self.GO = GO
        self.positions = []
        self.params = params

    def add_close_position(self, position):
            self.positions.append(position)

    def trades_table(self):
        table = pd.DataFrame({'position_type': [i.get_position_type() for i in self.positions],
                              'profit':[i.calculate_profit() for i in self.positions],
                              'profit_percent':[i.calculate_profit()/self.GO for i in self.positions],
                              'sum_profit': np.array([i.calculate_profit() for i in self.positions]).cumsum(),
                              'sum_percent_profit': np.array([i.calculate_profit()/self.GO for i in self.positions]).cumsum(),
                              'enter_time': [i.get_enter_time() for i in self.positions],
                              'close_time': [i.get_close_time() for i in self.positions]})

        table['deal_duration_bars'] = [i.get_close_index()-i.get_enter_index() for i in self.positions]
        table['deal_duration_hours'] = table.close_time - table.enter_time
        table['deal_duration_hours'] = table['deal_duration_hours'].apply(lambda x: x.seconds/60/60)

        return table

    def visualising_ekviti(self, df):
        deals = self.trades_table()
        self.visualising_ekviti2(df, deals)

    def statistics(self):

        def drawdown_analyse(df):
                max_drawdown = 0
                max_drawdown_duration_days = 0
                df['deposit'] = df.profit
                df['deposit'] = np.array(df['deposit']).cumsum() + self.GO
                profit = df.iloc[0, list(df.columns).index('deposit')]
                t0 = df.iloc[0, list(df.columns).index('close_time')]
                for i in df.index:
                    if df.loc[i, 'deposit'] > profit:
                        profit = df.loc[i, 'deposit']
                        t0 = df.loc[i, 'close_time']
                    else:
                        drawdown = (profit - df.loc[i, 'deposit'])/profit
                        t1 = df.loc[i, 'close_time']
                        drawdown_duration_days = t1-t0
                        drawdown_duration_days = drawdown_duration_days.days
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown
                        if drawdown_duration_days > max_drawdown_duration_days:
                            max_drawdown_duration_days = drawdown_duration_days
                return max_drawdown, max_drawdown_duration_days

        # Расчитаем коэффициент шарпа
        # sharp_df = pd.DataFrame({'time': self.__close_times, 'profit':self.__profits})
        # sharp_df['year'] = sharp_df.time.apply(lambda x: x.year)
        # sharp_df['month'] = sharp_df.time.apply(lambda x: x.month)
        # sharp_df['profit'] = sharp_df.profit / 5000
        # sharp_df = sharp_df.groupby(['year', 'month'], as_index=False).aggregate({'profit': 'sum'})
        # if sharp_df.profit.mean() ==0:
        #     sharp_rate = 0
        # elif sharp_df.profit.std() == 0:
        #     std_dev_profit = 0.01
        #     avg_profit = sharp_df.profit.mean()
        #     sharp_rate = (avg_profit - 8/12/100) / std_dev_profit
        # else:
        #     avg_profit = sharp_df.profit.mean()
        #     std_dev_profit = sharp_df.profit.std()
        #     sharp_rate = (avg_profit - 8/12/100) / std_dev_profit

        # Расчитаем коэффициент Сортино
        #
        # avg_profit = sharp_df.profit.mean()
        # std_dev_profit = sharp_df.profit[sharp_df.profit < 0].std()
        # sortino_rate = (avg_profit - 8/12/100) / std_dev_profit
        # print(sharp_df)

        if len(self.positions) > 0:
            df = self.trades_table()
            n_trades = df.shape[0]
            sum_profit_percent = df.sum_percent_profit.values[-1]
            mean_profit_percent = df.profit_percent.mean()
            part_plus_deals = df.profit_percent[df.profit_percent >= 0].shape[0] / (n_trades)
            if df.profit_percent[df.profit_percent >= 0].shape[0] >0:
                mean_plus_profit = df.profit_percent[df.profit_percent >= 0].mean()
            else:
                mean_plus_profit = 0
            if df.profit_percent[df.profit_percent < 0].shape[0] >0:
                mean_minus_profit = df.profit_percent[df.profit_percent < 0].mean()
            else:
                mean_minus_profit = 0
            if mean_minus_profit == 0:
                risk_reward = -1
            else:
                risk_reward = abs(mean_plus_profit/mean_minus_profit)
            max_drawdown, max_drawdown_duration_days = drawdown_analyse(df)
        else:
            n_trades = 0
            sum_profit_percent = 0
            mean_profit = 0
            part_plus_deals = 0
            mean_plus_profit = 0
            mean_minus_profit = 0
            risk_reward = 0
            max_drawdown, max_drawdown_duration_days = 0,0
            mean_profit_percent = 0

        result = {'n_trades': n_trades, 'part_plus_deals': part_plus_deals,  'sum_profit_percent': sum_profit_percent, 'mean_profit_percent': mean_profit_percent,  'mean_plus_profit': mean_plus_profit, 'mean_minus_profit': mean_minus_profit, 'risk_reward': risk_reward, 'max_drawdown': max_drawdown, 'max_drawdown_duration_days': max_drawdown_duration_days}
        for i in self.params:
            result[i] = self.params[i]
        return result

    @staticmethod
    def visualising_ekviti2(df, deals, trade_params = None):
        deals.index = deals.close_time
        for i in range(deals.shape[0]):
            if deals.index[i] not in df.index:
                temp = df.index - deals.index[i]
                temp = pd.DataFrame({'time_': df.index.values, 'divers': temp.values})
                temp['divers'] = temp.divers.apply(lambda x: abs(x.seconds + x.days * 24 * 60 * 60))
                temp = temp.sort_values('divers')
                ind = deals.index.values
                ind[i] = temp.iloc[0, 0]
                deals.index = ind

        df['real_ekviti_percent'] = 0
        df['real_ekviti_long_percent'] = 0
        df['real_ekviti_short_percent'] = 0
        mask = deals.close_time
        df.loc[mask,'real_ekviti_percent'] = deals.profit_percent

        mask = deals.loc[deals.position_type == 1, 'close_time']
        vals = deals.loc[deals.position_type == 1, 'profit_percent']
        df.loc[mask,'real_ekviti_long_percent'] = vals

        mask = deals.loc[deals.position_type == -1, 'close_time']
        vals = deals.loc[deals.position_type == -1, 'profit_percent']
        df.loc[mask,'real_ekviti_short_percent'] = vals

        df['real_ekviti_percent'] = df['real_ekviti_percent'].values.cumsum() + 1
        df['real_ekviti_long_percent'] = df['real_ekviti_long_percent'].values.cumsum() + 1
        df['real_ekviti_short_percent'] = df['real_ekviti_short_percent'].values.cumsum() + 1

        plt.figure(figsize=(20,10))
        plt.subplot(211)
        plt.plot(df.close)
        plt.title('График стоимости бумаги')
        plt.xlabel('Время')
        plt.ylabel('Цена')
        plt.subplot(212)
        plt.plot(df['real_ekviti_percent'], label = 'ALL')
        plt.plot(df['real_ekviti_long_percent'], label = 'LONG')
        plt.plot(df['real_ekviti_short_percent'], label = 'SHORT')
        plt.title('Эквити')
        plt.xlabel('Время')
        plt.ylabel('Доходность')
        plt.legend()

        if trade_params != None:
            name = str(trade_params)
            name = name.replace("}", "_")
            name = name.replace("{", "_")
            name = name.replace(")", "_")
            name = name.replace("(", "_")
            name = name.replace("]", "_")
            name = name.replace("[", "_")
            name = name.replace(":", "_")
            name = name.replace(":", "_")
            plt.savefig(name + '.png')
        else:
            # Сохраняем картинку с нумерацией 1,2,3,4,5
            num = 1
            while True:
                if os.path.exists('result_' + str(num) + '.png'):
                    num += 1
                else:
                    break

            plt.savefig('result_' + str(num) + '.png')

        plt.show()


class DataTransformDoubleSMA:
    def __init__(self, path):
        with open(path, mode='rb') as f:
            self.df = pickle.load(f)

    def resample_data(self, timeframe):

        """
        Ресемплируем датафрейм. Аргумент таймфрейм согласно документации пандаса
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html
        :param df: Pandas Data Frame
        :param timeframe: period
        :return: Pandas Data Frame
        """
        df = self.df.copy()
        df['total_volume'] = df.adj_price * df.volume
        ohlcv_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'buy':'sum', 'sell':'sum', 'total_volume':'sum'}
        df = df.resample(timeframe).agg(ohlcv_dict).dropna()
        df['adj_price'] = df.total_volume / df.volume
        del df['total_volume']
        return df

    def cut_data(self, df,  date_from = None, date_to = None):
        # Если нужно прогнать не на всей выборке, то обрезаем ее
        if date_from:
            mask1 = df.index >= date_from
        if date_to:
            mask2 = df.index < date_to
        if date_from and date_to:
            mask = mask1 * mask2
        elif date_from:
            mask = mask1
        elif date_to:
            mask = mask2

        if date_from or date_to:
            df = df.loc[mask, :]
        return df

    def make_indicators(self, df, params):


        df['ma_fast'] = talib.MA(df.close, timeperiod = params['fastma'])
        df['ma_slow'] = talib.MA(df.close, timeperiod= params['slowma'])
        df['make_trade'] = 0

        df = df.dropna()

        # Номера индексов столбцов
        ind_ma_fast = list(df.columns).index('ma_fast')
        ind_ma_slow = list(df.columns).index('ma_slow')
        ind_make_trade = list(df.columns).index('make_trade')

        for i in range(1, df.shape[0]):

            if df.iloc[i, ind_ma_fast] > df.iloc[i, ind_ma_slow] and df.iloc[i-1, ind_ma_fast] < df.iloc[i-1, ind_ma_slow]:
                df.iloc[i, ind_make_trade] = 1
            if df.iloc[i, ind_ma_fast] < df.iloc[i, ind_ma_slow] and df.iloc[i-1, ind_ma_fast] > df.iloc[i-1, ind_ma_slow]:
                df.iloc[i, ind_make_trade] = -1

        return df

class SimulatorDoubleMoovingAverage:
    def __init__(self, params=None, GO = 5000):
        self.position = None
        self.GO = GO
        self.report = StatReport(self.GO, params)
        self.params = params

    def simulation(self, df):

        # Номера индексов столбцов
        ind_open = list(df.columns).index('open')
        ind_high = list(df.columns).index('high')
        ind_low = list(df.columns).index('low')
        ind_close = list(df.columns).index('close')
        ind_volume = list(df.columns).index('volume')
        ind_adj_price = list(df.columns).index('adj_price')
        ind_make_trade = list(df.columns).index('make_trade')

        for i in range(1, df.shape[0]-1):

            # Если конец дня, закрываем позиции на последнем баре
            if self.params['close_position_in_evening']:
                if df.index[i].day != df.index[i+1].day:
                    if self.position:
                        self.position.set_close_position(i, df.index[i], df.iloc[i,ind_close])
                        self.report.add_close_position(self.position)
                        self.position = None
                    continue

            if df.iloc[i-1, ind_make_trade]:
                if self.position:
                    self.position.set_close_position(i, df.index[i], df.iloc[i,ind_close])
                    self.report.add_close_position(self.position)
                    self.position = None
                    if self.params['min_hour_treshold'] < df.index[i].hour < self.params['max_hour_treshold']:
                        self.position = Position(i, df.index[i], df.iloc[i,ind_open] ,df.iloc[i-1, ind_make_trade])
                else:
                    if  self.params['min_hour_treshold'] < df.index[i].hour < self.params['max_hour_treshold']:
                        self.position = Position(i, df.index[i], df.iloc[i,ind_open] ,df.iloc[i-1, ind_make_trade])

            if self.params['deal_period'] in self.params:
                # Если ограничение на длительность сделки
                if self.position:
                    if self.position.get_close_index() - self.position.get_enter_index() > self.params['deal_period']:
                        self.position.set_close_position(i, df.index[i], df.iloc[i,ind_close])
                        self.report.add_close_position(self.position)
                        self.position = None

    def set_params(self, params):
        self.params = params
        self.report = StatReport(self.GO, params)



if __name__ == '__main__':

    datapath = "D:\УИИ диплом\lion_tester\datas\SBRFF_1MIN.pickle"
    data =  DataTransformDoubleSMA(datapath)
    params = {'fastma': 5, 'slowma': 90, 'min_hour_treshold': 7,
              'max_hour_treshold': 22, 'close_position_in_evening': True, 'timeframe': '10min',
              'deal_period': 0}
    new_df = data.make_indicators(data.resample_data(params['timeframe']), params)
    simulator = SimulatorDoubleMoovingAverage(params=params, GO=5000)
    simulator.simulation(data.cut_data(new_df, date_from=datetime.datetime(2021, 7, 1, 0, 0, 0),
                                       date_to=datetime.datetime(2022, 1, 1, 0, 0, 0)))
    print(simulator.report.statistics())
    print(simulator.report.trades_table())
