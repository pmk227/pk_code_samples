import pandas as pd
import os
from src.core.backtesting.trade_cost_model import TradeCostModel
from src.core.backtesting.order_handler import OrderManager


class Backtester:
    def __init__(self, price_dict, starting_cash=10000.0, slippage=0.0, commission_type='fixed'):
        self.price_df = price_dict['price_df']
        self.price_df["datetime"] = pd.to_datetime(self.price_df["datetime"])
        self.validate_price_df(self.price_df)

        self.portfolio = {
            'position': 0,
            'cash': starting_cash,
            'history': []
        }

        self.pending_orders = {}
        self.signal_queue = []
        self.cost_model = TradeCostModel(commission_type=commission_type, slippage=slippage)
        self.order_manager = OrderManager(self.pending_orders)

    @staticmethod
    def validate_price_df(df: pd.DataFrame):
        required_cols = {
                        "datetime", "open", "high", "low", "close",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Price DataFrame missing required columns: {missing}")

    @staticmethod
    def validate_signal_df(df: pd.DataFrame):
        required_cols = {
            'submission_time', 'action', 'qty', 'trade_id', 'trade_group_id', 'instrument_id'
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Signal DataFrame missing required columns: {missing}")

    def run(self, signal_df: pd.DataFrame):
        signal_df = signal_df.dropna(subset=['action']).copy().sort_values('submission_time')
        signal_df["submission_time"] = pd.to_datetime(signal_df["submission_time"])
        self.validate_signal_df(signal_df)
        price_df = self.price_df.copy().reset_index(drop=True)
        signal_pointer = 0

        for idx, row in price_df.iterrows():
            current_time = row['datetime']

            # Process all signals scheduled for this time step
            while signal_pointer < len(signal_df) and signal_df.iloc[signal_pointer]['submission_time'] == current_time:
                signal = signal_df.iloc[signal_pointer].to_dict()
                self.order_manager.process_signal(signal)  # No current_time passed
                signal_pointer += 1

            # Evaluate pending orders and record any trades
            for trade_effect in self.order_manager.evaluate_orders(row):
                adjusted_trade_effect = self.cost_model.apply(trade_effect)
                self._record_trade(adjusted_trade_effect)

    def _record_trade(self, trade_record):
        # Rewrite this to accept the adjusted trade_effect
        qty = trade_record['qty']
        price = trade_record['price']
        action = trade_record['action']

        position_change = qty if action == 'buy' else -qty
        cost = qty * price
        total_fees = trade_record.get('total_fees', 0.0)

        self.portfolio['position'] += position_change
        cash_change = -(cost + total_fees) if action == 'buy' else (cost - total_fees)
        self.portfolio['cash'] += cash_change

        position_value = self.portfolio['position'] * price
        portfolio_value = self.portfolio['cash'] + position_value

        updated_fields = {
            'position': self.portfolio['position'],
            'cash': self.portfolio['cash'],
            'position_value': position_value,
            'portfolio_value': portfolio_value,
        }
        trade_record.update(updated_fields)

        # trade_record = {
        #     'execution_time': execution_time,
        #     'action': action,
        #     'qty': qty,
        #     'price': price,
        #     'position': self.portfolio['position'],
        #     'cash': self.portfolio['cash'],
        #     'position_value': position_value,
        #     'portfolio_value': portfolio_value,
        #     **kwargs  # flatten everything else in
        # }

        self.portfolio['history'].append(trade_record)

    def get_trade_history(self):
        return pd.DataFrame(self.portfolio['history'])

    def get_portfolio_state(self):
        return {
            'position': self.portfolio['position'],
            'cash': self.portfolio['cash'],
            'portfolio_value': self.portfolio['cash'] + self.portfolio['position'] * self.price_df.iloc[-1]['open']
        }

    def save_trade_history(self, output_dir:str, output_filename:str):
        os.makedirs(output_dir, exist_ok=True)
        trade_history = self.get_trade_history()
        trade_history.to_csv(os.path.join(output_dir, output_filename), index=False)

