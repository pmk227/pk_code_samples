from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class Order(ABC):
    DEFAULT_EXPIRATION_DAYS = 180

    def __init__(self, signal: dict):
        self.signal = self._validate_signal(signal.copy())
        self.entry_time = self.signal['submission_time']
        self.expiration = self.signal['expiration']

    def _validate_signal(self, signal: dict) -> dict:
        for field in ['trade_id', 'trade_group_id', 'instrument_id']:
            if field not in signal:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(signal[field], int):
                raise TypeError(f"{field} must be type int")

        for field in ["submission_time"]:
            if field not in signal:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(signal[field], datetime):
                raise TypeError(f"{field} must be type datetime")

        if 'expiration' not in signal or pd.isna(signal['expiration']):
            signal['expiration'] = signal['submission_time'] + timedelta(days=self.DEFAULT_EXPIRATION_DAYS)

        signal['action'] = signal['action'].lower()
        return signal

    @abstractmethod
    def evaluate(self, row):
        raise NotImplementedError("Subclasses must implement evaluate().")

    def create_trade_effect(self, execution_time, price, liquidity_type='remove'):
        effect = self.trade_effect_template()
        effect.update(self.signal)

        effect.update({
            'execution_time': execution_time,
            'price': price,
            'action': 'buy' if 'buy' in self.signal['action'] else 'sell',
            'liquidity_type': liquidity_type,
        })
        return effect

    @staticmethod
    def trade_effect_template():
        return {
            'execution_time':None,
            'action':None,
            'qty':np.nan,
            'price':np.nan,
            'position':np.nan,
            'cash':np.nan,
            'position_value':np.nan,
            'portfolio_value':np.nan,
            'submission_time':None,
            'trade_id':np.nan,
            'trade_group_id':np.nan,
            'instrument_id':np.nan,
            'limit_price':np.nan,
            'entry_time':None,
            'expiration':None,
            'liquidity_type':None,
            'commission_fees':np.nan,
            'clearing_fees':np.nan,
            'nyse_pass_through_fees':np.nan,
            'finra_pass_through_fees':np.nan,
            'exchange_fees':np.nan,
            'sec_fees':np.nan,
            'finra_trading_fees':np.nan,
            'cat_fees':np.nan,
            'total_fees':np.nan
        }


class MarketOrder(Order):
    def evaluate(self, row):
        if row['datetime'] < self.entry_time or row['datetime'] > self.expiration:
            return None
        return self.create_trade_effect(row['datetime'], row['open'])


class LimitOrder(Order):
    def evaluate(self, row):
        if row['datetime'] < self.entry_time or row['datetime'] > self.expiration:
            return None

        price = self.signal.get('limit_price')
        action = self.signal['action'].lower()

        if action == 'buy_limit' and row['low'] <= price:
            return self.create_trade_effect(row['datetime'], price, liquidity_type='add')
        elif action == 'sell_limit' and row['high'] >= price:
            return self.create_trade_effect(row['datetime'], price, liquidity_type='add')
        return None


class StopOrder(Order):
    def evaluate(self, row):
        if row['datetime'] < self.entry_time or row['datetime'] > self.expiration:
            return None

        price = self.signal.get('stop_price')
        action = self.signal['action'].lower()

        if row['low'] <= price <= row['high']:
            return self.create_trade_effect(row['datetime'], price)
        return None


class StopLimitOrder(Order):
    def __init__(self, signal):
        super().__init__(signal)
        self.triggered = False

    def evaluate(self, row):
        if row['datetime'] < self.entry_time or row['datetime'] > self.expiration:
            return None

        stop_price = self.signal.get('stop_price')
        action = self.signal['action'].lower()

        if not self.triggered:
            if row['low'] <= stop_price <= row['high']:
                self.triggered = True

        if self.triggered:
            limit_price = self.signal.get('limit_price')
            if action == 'buy_stop_limit' and row['low'] <= limit_price:
                return self.create_trade_effect(row['datetime'], limit_price)
            elif action == 'sell_stop_limit' and row['high'] >= limit_price:
                return self.create_trade_effect(row['datetime'], limit_price)
        return None


class OrderFactory:
    @staticmethod
    def create(signal):
        action = signal['action'].lower()

        if action in ['buy', 'sell']:
            return MarketOrder(signal)
        elif action in ['buy_limit', 'sell_limit']:
            return LimitOrder(signal)
        elif action in ['buy_stop', 'sell_stop']:
            return StopOrder(signal)
        elif action in ['buy_stop_limit', 'sell_stop_limit']:
            return StopLimitOrder(signal)
        else:
            raise ValueError(f"Unknown order action type: {action}")