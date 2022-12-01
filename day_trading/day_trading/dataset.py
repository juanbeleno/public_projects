#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:44:46 2022
@author: Juan Beleño
"""
from .files import DayTradingFiles
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import os
import requests
# Division by zero
np.seterr(divide='ignore', invalid='ignore')


class DayTradingDataset:
    def __init__(self) -> None:
        self.range = '60d'
        self.granularity = '5m'
        self.window = 6
        self.files = DayTradingFiles()
        self.p_stop_loss = 0.004
        self.slippage = 0.002
        self.p_take_profit = 0.005

        self.candlestick_patterns = [
            'BEARISH_ENGULFING_PATTERN',
            'BULLISH_ENGULFING_PATTERN',
            'BEARISH_EVENING_STAR',
            'BEARISH_HARAMI',
            'BULLISH_HARAMI',
            'BULLISH_RISING_THREE',
            'BEARISH_RISING_THREE'
        ]

    def download_ticket_candidates(self):
        # S&P 500
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'
        }
        request = requests.get(
            'https://finviz.com/maps/sec.json?rev=317',
            headers=headers
        )
        response = request.json()

        data = []
        for category_data in response['children']:
            for subcategory_data in category_data['children']:
                for company in subcategory_data['children']:
                    data.append({
                        'category': category_data['name'],
                        'subcategory': subcategory_data['name'],
                        'company_name': company['description'],
                        'company_code': company['name'],
                        'company_value': company['value']
                    })
        tickets_data = pd.DataFrame(data)
        tickets_data.to_csv(self.files.ticket_candidates, index=False)

    def get_recent_data(self, ticket):
        # Collect the most recent data using the Yahoo Finances API
        params = {
            'region': 'US',
            'lang': 'en-US',
            'includePrePost': 'false',
            'interval': self.granularity,
            'useYfid': 'true',
            'range': self.range,
            'corsDomain': 'finance.yahoo.com',
            '.tsrc': 'finance'
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'
        }
        request = requests.get(
            f'https://query1.finance.yahoo.com/v8/finance/chart/{ticket}',
            params=params,
            headers=headers
        )
        recent_data = request.json()
        return recent_data

    def get_raw_data(self, ticket):
        response = []
        print(f'Getting data for {ticket}')
        # Get the data from the past stored manually by me (Juan Beleño)
        # collecting the data on certain dates.
        ticket_directory = os.path.join(
            self.files.input_directory, f'{ticket}_interval_{self.granularity}_range_{self.range}')
        filepaths = next(os.walk(ticket_directory), (None, None, []))[
            2]  # [] if no file

        for filepath in filepaths:
            with open(os.path.join(ticket_directory, filepath)) as json_file:
                response.append(json.load(json_file))

        recent_data = self.get_recent_data(ticket)
        response.append(recent_data)
        """
        # Save the data in a folder for future training
        ticket_folder = Path(ticket_directory)
        ticket_folder.mkdir(exist_ok=True)
        ticket_filepath = os.path.join(
            ticket_directory, f"{datetime.utcnow().strftime('%Y-%m-%d')}.json")
        with open(ticket_filepath, 'w') as f:
            json.dump(recent_data, f)
        """

        return response

    def get_support_slope(self, lows):
        lows = lows.tolist()
        num_samples = len(lows)
        if num_samples < 2:
            return np.nan
        first_point = lows[0]
        last_point = lows[-1]
        slope = (last_point - first_point) / (num_samples - 1)
        intercept = first_point
        new_first_index = 0
        max_error = 0
        for index in range(1, num_samples - 1):
            y = slope * index + intercept
            error = y - lows[index]
            if error > max_error:
                error = max_error
                new_first_index = index
        num_samples = num_samples - new_first_index
        slope = (last_point - lows[new_first_index]) / (num_samples - 1)
        return slope

    def get_resistance_slope(self, highs):
        highs = highs.tolist()
        num_samples = len(highs)
        if num_samples < 2:
            return np.nan
        first_point = highs[0]
        last_point = highs[-1]
        slope = (last_point - first_point) / (num_samples - 1)
        intercept = first_point
        new_first_index = 0
        max_error = 0
        for index in range(1, num_samples - 1):
            y = slope * index + intercept
            error = highs[index] - y
            if error > max_error:
                error = max_error
                new_first_index = index
        num_samples = num_samples - new_first_index
        slope = (last_point - highs[new_first_index]) / (num_samples - 1)
        return slope

    def get_candlestick_score(self, item):
        # Source: https://blog.quantinsti.com/candlestick-patterns-meaning/
        close = item['close']
        open = item['open']
        high = item['high']
        low = item['low']
        score = 0
        hl = high - low

        if hl == 0:
            score = 0
        elif close - open > 0:
            # Bullish candlestick
            hc = high - close
            ol = open - low
            co = close - open
            score = 0.5 * ((3 * ol + 5 * co - 10 * hc) / (18 * hl) + 1)
        else:
            # Bearish candlestick
            oc = open - close
            ho = high - open
            cl = close - low
            score = - 0.5 * ((3 * ho + 5 * oc - 10 * cl) / (18 * hl) + 1)
        return score

    def get_candlestick_patterns(self, data):
        # Source: https://www.investopedia.com/trading/candlestick-charting-what-is-it/
        patterns = []

        # Bearish engulfing pattern
        if (
            len(data) >= 4
            and (data[-4]['close'] - data[-4]['open']) > 0
            and (data[-3]['close'] - data[-3]['open']) > 0
            and (data[-2]['close'] - data[-2]['open']) > 0
            and (data[-1]['close'] - data[-1]['open']) < 0
            and data[-1]['close'] < data[-2]['open']
            and data[-1]['open'] > data[-2]['close']
        ):
            patterns.append('BEARISH_ENGULFING_PATTERN')

        # Bullish engulfing pattern
        if (
            len(data) >= 4
            and (data[-4]['close'] - data[-4]['open']) < 0
            and (data[-3]['close'] - data[-3]['open']) < 0
            and (data[-2]['close'] - data[-2]['open']) < 0
            and (data[-1]['close'] - data[-1]['open']) > 0
            and data[-1]['close'] > data[-2]['open']
            and data[-1]['open'] < data[-2]['close']
        ):
            patterns.append('BULLISH_ENGULFING_PATTERN')

        # Bearish evening start pattern
        if (
            len(data) >= 4
            and (data[-4]['close'] - data[-4]['open']) > 0
            and (data[-3]['close'] - data[-3]['open']) > 0
            and (data[-1]['close'] - data[-1]['open']) < 0
            and abs(data[-2]['close'] - data[-2]['open']) < (data[-2]['open'] - data[-2]['close'])
            and (abs((data[-2]['open'] - data[-2]['close']) - (data[-3]['close'] - data[-3]['open'])) / (data[-3]['close'] - data[-3]['open'])) < 0.1
        ):
            patterns.append('BEARISH_EVENING_STAR')

        # Bearish Harami
        if (
            len(data) >= 4
            and (data[-4]['close'] - data[-4]['open']) > 0
            and (data[-3]['close'] - data[-3]['open']) > 0
            and (data[-2]['close'] - data[-2]['open']) > 0
            and (data[-1]['close'] - data[-1]['open']) < 0
            and data[-1]['close'] > data[-2]['open']
            and data[-1]['open'] > data[-2]['close']
        ):
            # TODO: Bearish Harami Cross
            patterns.append('BEARISH_HARAMI')

        # Bullish Harami
        if (
            len(data) >= 4
            and (data[-4]['close'] - data[-4]['open']) < 0
            and (data[-3]['close'] - data[-3]['open']) < 0
            and (data[-2]['close'] - data[-2]['open']) < 0
            and (data[-1]['close'] - data[-1]['open']) > 0
            and data[-1]['close'] < data[-2]['open']
            and data[-1]['open'] < data[-2]['close']
        ):
            # TODO: Bullish Harami Cross
            patterns.append('BULLISH_HARAMI')

        # Bullish Rising Three
        if (
            len(data) >= 5
            and (data[-5]['close'] - data[-5]['open']) > 0
            and (data[-4]['close'] - data[-4]['open']) < 0
            and (data[-3]['close'] - data[-3]['open']) < 0
            and (data[-2]['close'] - data[-2]['open']) < 0
            and (data[-1]['close'] - data[-1]['open']) > 0
            and abs(data[-5]['close'] - data[-4]['open']) / (data[-5]['close'] - data[-5]['open']) < 0.05
            and abs(data[-5]['open'] - data[-2]['close']) / (data[-5]['close'] - data[-5]['open']) < 0.05
            and (abs((data[-1]['close'] - data[-1]['open']) - (data[-5]['close'] - data[-5]['open'])) / (data[-5]['close'] - data[-5]['open'])) < 0.1
        ):
            patterns.append('BULLISH_RISING_THREE')

        # Bearish Rising Three
        if (
            len(data) >= 5
            and (data[-5]['close'] - data[-5]['open']) < 0
            and (data[-4]['close'] - data[-4]['open']) > 0
            and (data[-3]['close'] - data[-3]['open']) > 0
            and (data[-2]['close'] - data[-2]['open']) > 0
            and (data[-1]['close'] - data[-1]['open']) < 0
            and abs(data[-5]['open'] - data[-4]['close']) / (data[-5]['open'] - data[-5]['close']) < 0.05
            and abs(data[-5]['close'] - data[-2]['open']) / (data[-5]['open'] - data[-5]['close']) < 0.05
            and (abs((data[-1]['open'] - data[-1]['close']) - (data[-5]['open'] - data[-5]['close'])) / (data[-5]['open'] - data[-5]['close'])) < 0.1
        ):
            patterns.append('BEARISH_RISING_THREE')
        return patterns

    def transform_data(self, ticket, raw_data, bet_type='long'):
        print('Convert the HTTP response into a pandas DataFrame')
        dataset = pd.DataFrame(
            columns=['timestamp', 'high', 'low', 'close', 'open', 'volume'])
        for partial_data in raw_data:
            print(f'Processing data for {ticket}')
            timestamps = partial_data['chart']['result'][0]['timestamp']
            data = partial_data['chart']['result'][0]['indicators']['quote'][0]
            partial_dataset = pd.DataFrame({
                'timestamp': timestamps,
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'open': data['open'],
                'volume': data['volume']
            })
            dataset = pd.concat([partial_dataset, dataset])

        # Remove duplicated data
        dataset.drop_duplicates(inplace=True)

        # Convert int to timestamp
        dataset['timestamp'] = dataset['timestamp'].apply(
            lambda timestamp: pd.to_datetime(timestamp, utc=False, unit='s')
        )
        dataset['day_of_week'] = dataset['timestamp'].dt.dayofweek
        dataset['hour'] = dataset['timestamp'].dt.hour
        dataset['minute'] = dataset['timestamp'].dt.minute
        dataset['date'] = dataset['timestamp'].dt
        print(f'There are {dataset.shape[0]} data points.')

        # Verify that the dataset is ordered by timestamp
        dataset.sort_values(by='timestamp', ascending=True,
                            inplace=True, ignore_index=True)

        for window in [20, 50, 100, 200]:
            dataset.loc[:, f'EMA_{window}'] = (dataset['close'].ewm(
                span=window, adjust=False).mean().values - dataset['close'])

        # Support and Resistance
        num_data_points = 3
        support_slope = dataset['low'].rolling(
            window=self.window * num_data_points).apply(self.get_support_slope)
        resistance_slope = dataset['high'].rolling(
            window=self.window * num_data_points).apply(self.get_resistance_slope)

        # Find the theoretical gains/losses from the trendlines
        dataset[f'Support_pct_change'] = self.window * \
            support_slope / dataset['close']
        dataset[f'Resistance_pct_change'] = self.window * \
            resistance_slope / dataset['close']

        # Past values for the grow
        delta_close = (dataset['close'] -
                       dataset['open']) / dataset['open']
        delta_volume = (dataset['volume'] -
                        dataset['volume'].shift(1))
        for step in range(1, self.window * num_data_points + 1):
            dataset[f'P_Delta_Close_{step}'] = delta_close.shift(
                step)
            dataset[f'P_Delta_Volume_{step}'] = delta_volume.shift(step)
        for index in range(num_data_points):
            dataset[f'Delta_Close_Sum_{index}'] = 0
            #dataset[f'Delta_Volume_Sum_{index}'] = 0
            for step in range(1, self.window + 1):
                dataset[f'Delta_Close_Sum_{index}'] += dataset[f'P_Delta_Close_{index * self.window + step}']
                #dataset[f'Delta_Volume_Sum_{index}'] += dataset[f'P_Delta_Volume_{index * self.window + step}']

        columns = [col for col in dataset.columns if 'P_Delta_Close_' not in col]
        columns = [col for col in columns if 'P_Delta_Volume_' not in col]
        dataset = dataset[columns].copy()

        # RSI
        rsi_period = 14
        delta = dataset['close'] - dataset['open']
        gains, loses = delta.copy(), delta.copy()
        gains[gains < 0] = 0
        loses[loses > 0] = 0
        rolling_mean_gains = gains.ewm(
            span=rsi_period, adjust=False).mean().values
        rolling_mean_loses = np.absolute(loses.ewm(
            span=rsi_period, adjust=False).mean().values)
        rs = rolling_mean_gains / rolling_mean_loses
        dataset['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_26 = dataset['close'].ewm(
            span=26, adjust=False).mean().values
        ema_12 = dataset['close'].ewm(
            span=12, adjust=False).mean().values
        dataset['MACD'] = ema_26 - ema_12
        dataset['MACD_Signal'] = dataset['MACD'].ewm(
            span=9, adjust=False).mean().values
        # dataset['MACD_Histogram'] = dataset['MACD'] - dataset['MACD_Signal']

        # Default values for the patterns
        '''
        for pattern in self.candlestick_patterns:
            dataset[pattern] = 0
        '''

        # Highs and Lows
        for point in range(1, num_data_points + 1):
            dataset[f'p_Delta_High_{point}'] = 0
            dataset[f'p_Delta_Low_{point}'] = 0

        # VWAP and patterns
        dataset_list = dataset.to_dict('records')
        new_dataset_list = []
        last_date = None
        cummulative_pv = 0
        cummulative_volume = 0
        volume_anchor = 0
        previous_close = dataset_list[0]['close']
        previous_high = dataset_list[0]['high']
        previous_low = dataset_list[0]['low']
        closes = []
        for index, item in enumerate(dataset_list):
            # Patterns
            '''
            patterns = self.get_candlestick_patterns(
                dataset_list[max(0, index - 4):index + 1])
            for pattern in patterns:
                item[pattern] = 1
            '''

            # TR
            item['TR'] = max(item['high'] - item['low'], abs(item['high'] -
                             previous_close), abs(item['low'] - previous_close))
            previous_close = item['close']

            # Directional Movement
            item['positive_DM'] = item['high'] - previous_high
            item['negative_DM'] = item['low'] - previous_low
            previous_high = item['high']
            previous_low = item['low']

            # Historical data of lows and high changes in percentage related to close
            closes.append(item['close'])
            for point in range(1, num_data_points + 1):
                data_size = len(closes)
                start_index = data_size - self.window * point
                end_index = start_index + self.window
                if start_index > 0:
                    lowest_point = min(closes[start_index:end_index])
                    highest_point = max(closes[start_index:end_index])
                    close_point = closes[start_index - 1]
                    item[f'p_Delta_High_{point}'] = (
                        highest_point - close_point) / close_point
                    item[f'p_Delta_Low_{point}'] = (
                        close_point - lowest_point) / close_point

            # VWAP
            current_date = item['date']
            if current_date != last_date:
                cummulative_pv = 0
                cummulative_volume = 0
                volume_anchor = item['volume']
            item['relative_volume'] = 0
            if volume_anchor > 0:
                item['relative_volume'] = (
                    item['volume'] - volume_anchor) / volume_anchor
            cummulative_pv += (item['close'] + item['high'] + item['low']) / 3
            cummulative_volume += item['volume']
            item['VWAP'] = 0
            if cummulative_volume > 0:
                item['VWAP'] = cummulative_pv / cummulative_volume
            new_dataset_list.append(item)
            last_date = current_date

        # Convert back to dataframe
        dataset = pd.DataFrame(new_dataset_list)

        # ATR
        dataset['ATR'] = dataset['TR'].rolling(window=14).mean().values

        # ADX
        positive_DI = 100 * dataset['positive_DM'].ewm(
            span=14, adjust=False).mean().values / dataset['ATR']
        negative_DI = 100 * dataset['negative_DM'].ewm(
            span=14, adjust=False).mean().values / dataset['ATR']
        DX = 100 * (abs(positive_DI - negative_DI) /
                    abs(positive_DI + negative_DI))
        previous_DX = DX.shift(1)
        dataset['ADX'] = ((previous_DX * 13) + DX) / 14

        # Define target variables
        print('Defining the targets.')
        dataset['target_close'] = dataset['close'].shift(-self.window)

        target_high = dataset['close'].rolling(
            window=self.window).max().shift(-self.window)
        target_low = dataset['close'].rolling(
            window=self.window).min().shift(-self.window)
        high_delta = (target_high - dataset['close']) / dataset['close']
        low_delta = (dataset['close'] - target_low) / dataset['close']

        if bet_type == 'long':
            label_close = []
            for index in range(dataset.shape[0]):
                if (high_delta[index] > self.p_take_profit):
                    label_close.append(1)
                else:
                    label_close.append(0)
            dataset['label_close'] = label_close
        else:
            label_close = []
            for index in range(dataset.shape[0]):
                if (low_delta[index] > self.p_take_profit):
                    label_close.append(1)
                else:
                    label_close.append(0)
            dataset['label_close'] = label_close

        # Remove some features because they have high correlation with other features
        correlated_columns = [
            'high',  'low', 'open', 'TR', 'positive_DM', 'negative_DM']
        new_features = [
            col for col in dataset.columns if col not in correlated_columns]

        return dataset[new_features].copy()

    def get_all_dataset(self, ticket, bet_type='long'):
        print('Getting the data for the ticket')
        raw_data = self.get_raw_data(ticket)
        dataset = self.transform_data(ticket, raw_data, bet_type)
        # Drop NA for Linear Regression
        dataset.dropna(inplace=True)

        features_columns = [col for col in dataset.columns if col not in [
            'timestamp', 'label_close', 'target_close', 'date']]
        dataset = dataset.sample(frac=1, ignore_index=True)
        label_close = dataset['label_close'].tolist()
        features_df = dataset[features_columns].copy()
        return (features_df, label_close)

    def get_prediction_features(self, ticket, bet_type='long'):
        print('Getting the data for the ticket')
        raw_data = [self.get_recent_data(ticket)]
        dataset = self.transform_data(ticket, raw_data)
        dataset = dataset[dataset['minute'] % 5 == 0].copy()
        # print(dataset.tail(1).to_dict('records'))

        features_columns = [col for col in dataset.columns if col not in [
            'timestamp', 'label_close', 'target_close', 'date']]
        features_df = dataset[features_columns].copy()
        features_df = features_df.tail(1)
        return features_df

    def test_train_split(self, ticket, bet_type='long'):
        print('Getting the data for the ticket')
        raw_data = self.get_raw_data(ticket)
        dataset = self.transform_data(ticket, raw_data, bet_type)
        print(f'# Samples with NaN: {dataset.shape[0]}')

        # Drop NA for Linear Regression
        dataset.dropna(inplace=True)
        print(f'# Samples without NaN: {dataset.shape[0]}')

        # Define timestamp ranges for datasets
        two_weeks_ago = datetime.utcnow() - timedelta(days=14)
        # week_ago = datetime(2022, 10, 18)
        print(f'Two weeks ago: {two_weeks_ago}')

        # Split the datasets
        features = [col for col in dataset.columns if col not in [
            'timestamp', 'label_close', 'target_close', 'date']]

        print('Defining the training data.')
        train_df = dataset[dataset['timestamp'] <= two_weeks_ago].copy()
        # Sample the dataset to add a little bit of randomness before training
        train_df = train_df.sample(frac=1, ignore_index=True)
        label_close_train = train_df['label_close'].tolist()
        features_train_df = train_df[features].copy()

        print('Defining the test data.')
        test_df = dataset[dataset['timestamp'] >= two_weeks_ago].copy()
        # test_df.to_csv(self.files.features_sample, index=False)
        test_df.dropna(subset=['label_close'], inplace=True)
        label_close_test = test_df['label_close'].tolist()
        features_test_df = test_df[features].copy()

        return (
            features_train_df, label_close_train,
            features_test_df, label_close_test
        )

    def test_val_train_split(self, ticket, bet_type='long'):
        print('Getting the data for the ticket')
        raw_data = self.get_raw_data(ticket)
        dataset = self.transform_data(ticket, raw_data, bet_type)
        # Drop NA for Linear Regression
        dataset.dropna(inplace=True)

        # Define timestamp ranges for datasets
        week_ago = datetime.utcnow() - timedelta(days=7)
        # week_ago = datetime(2022, 10, 18)
        three_weeks_ago = datetime.utcnow() - timedelta(days=21)
        # two_weeks_ago = datetime(2022, 10, 11)
        print(f'A week ago: {week_ago}')

        # Split the datasets
        features = [col for col in dataset.columns if col not in [
            'timestamp', 'label_close', 'target_close', 'date']]

        print('Defining the training data.')
        train_df = dataset[dataset['timestamp'] <= three_weeks_ago].copy()
        # Sample the dataset to add a little bit of randomness before training
        train_df = train_df.sample(frac=1, ignore_index=True)
        label_close_train = train_df['label_close'].tolist()
        features_train_df = train_df[features].copy()

        print('Defining the validation data.')
        val_df = dataset[((dataset['timestamp'] > three_weeks_ago) & (
            dataset['timestamp'] < week_ago))].copy()
        label_close_val = val_df['label_close'].tolist()
        features_val_df = val_df[features].copy()

        print('Defining the test data.')
        test_df = dataset[dataset['timestamp'] >= week_ago].copy()
        # test_df.to_csv(self.files.features_sample, index=False)
        test_df.dropna(subset=['label_close'], inplace=True)
        label_close_test = test_df['label_close'].tolist()
        target_close_test = test_df['target_close'].tolist()
        features_test_df = test_df[features].copy()

        return (
            features_train_df, label_close_train,
            features_val_df, label_close_val,
            features_test_df, label_close_test, target_close_test
        )
