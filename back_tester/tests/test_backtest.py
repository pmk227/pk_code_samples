import unittest
import pandas as pd
from datetime import datetime, timedelta
from src.core.backtesting.backtest import Backtester


class TestBacktester(unittest.TestCase):
    def setUp(self):
        base_time = datetime(2023, 1, 1, 9)
        self.base_time = base_time
        self.price_df = pd.DataFrame([
            {
                'datetime': base_time + timedelta(hours=i), 'open': 100 + i, 'high': 101 + i,
                'low': 99 + i, 'close': 100.5 + i
            }
            for i in range(10)
        ])

        self.price_dict = {
            'instrument_id': 1,
            'instrument_type': 'equity',
            'price_df': self.price_df
        }

        self.signal_df = pd.DataFrame([
            self.buy_signal(self.price_df.loc[0, 'datetime'], 5, trade_id=1, trade_group_id=1, instrument_id=1),
            self.sell_signal(self.price_df.loc[2, 'datetime'], 2, trade_id=2, trade_group_id=1, instrument_id=1)
        ])

    @staticmethod
    def buy_signal(dt, qty, trade_id, trade_group_id, instrument_id):
        return {'submission_time': dt, 'action': 'buy', 'qty': qty, 'trade_id': trade_id,
                'trade_group_id': trade_group_id, 'instrument_id': instrument_id}

    @staticmethod
    def sell_signal(dt, qty, trade_id, trade_group_id, instrument_id):
        return {'submission_time': dt, 'action': 'sell', 'qty': qty, 'trade_id': trade_id,
                'trade_group_id': trade_group_id, 'instrument_id': instrument_id}

    def test_portfolio_state_with_fees(self):
        bt = Backtester(self.price_dict, starting_cash=10000.0, slippage=0.0, commission_type='tiered')
        bt.run(self.signal_df)
        state = bt.get_portfolio_state()

        self.assertEqual(state['position'], 3)  # bought 5, sold 2

        buy_price = self.price_df.loc[0, 'open']
        sell_price = self.price_df.loc[2, 'open']
        approx_cash = 10000 - (5 * buy_price) + (2 * sell_price)
        self.assertLess(state['cash'], approx_cash)  # due to fees

    def test_no_action_when_no_signal(self):
        null_df = pd.DataFrame([
            {'submission_time': self.price_df.loc[0, 'datetime'], 'action': None, 'qty': None, 'trade_id':1, 'trade_group_id':1, 'instrument_id':1 },
        ])
        bt = Backtester(self.price_dict)
        bt.run(null_df)
        history = bt.get_trade_history()
        self.assertEqual(len(history), 0)
        self.assertEqual(bt.get_portfolio_state()['position'], 0)

    def test_custom_starting_cash(self):
        bt = Backtester(self.price_dict, starting_cash=1_000_000)
        state = bt.get_portfolio_state()
        self.assertEqual(state['cash'], 1_000_000)
        self.assertEqual(state['position'], 0)
        self.assertEqual(state['portfolio_value'], 1_000_000)


class TestBacktesterMarketOrders(unittest.TestCase):
    def setUp(self):
        base_time = datetime(2023, 1, 1, 9)
        self.price_df = pd.DataFrame([
            {
                'datetime': base_time + timedelta(hours=i), 'open': 100 + i, 'high': 101 + i,
                'low': 99 + i, 'close': 100.5 + i
            }
            for i in range(5)
        ])

        self.price_dict = {
            'instrument_id': 1,
            'instrument_type': 'equity',
            'price_df': self.price_df
        }

    @staticmethod
    def buy_signal(dt, qty, trade_id, trade_group_id, instrument_id):
        return {'submission_time': dt, 'action': 'buy', 'qty': qty, 'trade_id': trade_id,
                'trade_group_id': trade_group_id, 'instrument_id': instrument_id}

    @staticmethod
    def sell_signal(dt, qty, trade_id, trade_group_id, instrument_id):
        return {'submission_time': dt, 'action': 'sell', 'qty': qty, 'trade_id': trade_id,
                'trade_group_id': trade_group_id, 'instrument_id': instrument_id}

    def test_market_buy_executes_immediately(self):
        signal_df = pd.DataFrame([
            self.buy_signal(self.price_df.loc[0, 'datetime'], 1, 1, 101, 1)
        ])
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(signal_df)
        history = bt.get_trade_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]['action'], 'buy')
        self.assertEqual(history.iloc[0]['execution_time'], self.price_df.loc[0, 'datetime'])

    def test_market_sell_executes_immediately(self):
        signal_df = pd.DataFrame([
            self.sell_signal(self.price_df.loc[0, 'datetime'], 1, 2, 101, 1)
        ])
        bt = Backtester(self.price_dict, starting_cash=0.0)
        bt.portfolio['position'] = 1
        bt.run(signal_df)
        history = bt.get_trade_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]['action'], 'sell')
        self.assertEqual(history.iloc[0]['execution_time'], self.price_df.loc[0, 'datetime'])


class TestBacktesterLimitOrders(unittest.TestCase):
    def setUp(self):
        base_time = datetime(2023, 1, 1, 9)
        self.price_df = pd.DataFrame([
            {'datetime': base_time + timedelta(hours=i), 'open': 100 + i, 'high': 101 + i, 'low': 97 + i, 'close': 101+i}
            for i in range(10)
        ])
        self.price_dict = {
            'instrument_id': 1,
            'instrument_type': 'equity',
            'price_df': self.price_df
        }

    @staticmethod
    def limit_signal(dt, action, qty, limit_price, trade_id, trade_group_id, instrument_id, expiration=None):
        return {
            'submission_time': dt,
            'action': action,
            'qty': qty,
            'limit_price': limit_price,
            'trade_id': trade_id,
            'trade_group_id': trade_group_id,
            'instrument_id': instrument_id,
            'expiration': expiration
        }

    @staticmethod
    def buy_signal(dt, qty, trade_id, trade_group_id, instrument_id):
        return {'submission_time': dt, 'action': 'buy', 'qty': qty, 'trade_id': trade_id,
                'trade_group_id': trade_group_id, 'instrument_id': instrument_id}

    def test_market_limit_order_executes(self):
        signal_df = pd.DataFrame([
            self.buy_signal(self.price_df.loc[0, 'datetime'], 1, trade_id=2, trade_group_id=2, instrument_id=1),
            self.limit_signal(self.price_df.loc[1, 'datetime'], 'buy_limit', 1, 99.5, trade_id=1, trade_group_id=1, instrument_id=1),


        ])
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(signal_df)
        history = bt.get_trade_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history.iloc[0]['action'], 'buy')
        self.assertEqual(history.iloc[0]['execution_time'], self.price_df.loc[0, 'datetime'])
        self.assertEqual(history.iloc[1]['action'], 'buy')
        self.assertEqual(history.iloc[1]['liquidity_type'], 'add')
        pass


    def test_limit_order_executes_same_bar(self):
        signal_df = pd.DataFrame([
            self.limit_signal(self.price_df.loc[0, 'datetime'], 'buy_limit', 1, 99.5, trade_id=1, trade_group_id=1, instrument_id=1)
        ])
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(signal_df)
        history = bt.get_trade_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]['action'], 'buy')
        self.assertEqual(history.iloc[0]['execution_time'], self.price_df.loc[0, 'datetime'])

    def test_limit_order_does_not_fill(self):
        signal_df = pd.DataFrame([
            self.limit_signal(self.price_df.loc[0, 'datetime'], 'buy_limit', 1, 80.0, trade_id=2,trade_group_id=1, instrument_id=1)
        ])
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(signal_df)
        history = bt.get_trade_history()
        self.assertEqual(len(history), 0)

    def test_limit_order_respects_price_floor(self):
        signal_df = pd.DataFrame([
            self.limit_signal(self.price_df.loc[0, 'datetime'], 'buy_limit', 1, 99.5, trade_id=3, trade_group_id=1, instrument_id=1)
        ])
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(signal_df)
        history = bt.get_trade_history()
        self.assertEqual(len(history), 1)
        self.assertLessEqual(history.iloc[0]['price'], 99.5)


class TestBacktesterOrderManagement(unittest.TestCase):
    def setUp(self):
        base_time = datetime(2023, 1, 1, 9)
        self.price_df = pd.DataFrame([
            {'datetime': base_time + timedelta(hours=i), 'open': 100 + i, 'high': 101 + i, 'low': 99 + i, 'close': 101+i}
            for i in range(10)
        ])
        self.price_dict = {
            'instrument_id': 1,
            'instrument_type': 'equity',
            'price_df': self.price_df
        }

    @staticmethod
    def limit_signal(dt, action, qty, limit_price, trade_id, trade_group_id, instrument_id, expiration=None):
        if expiration is None:
            expiration = dt + timedelta(days=180)
        return {
            'submission_time': dt,
            'action': action,
            'qty': qty,
            'limit_price': limit_price,
            'trade_id': trade_id,
            'trade_group_id': trade_group_id,
            'instrument_id': instrument_id,
            'expiration': expiration
        }

    def test_cancel_before_execution(self):
        signals = pd.DataFrame([
            self.limit_signal(self.price_df.loc[0, 'datetime'], 'buy_limit', 1, 95, trade_id=1, trade_group_id=101,
                              instrument_id=1),
            {'submission_time': self.price_df.loc[1, 'datetime'], 'action': 'cancel_order', 'trade_id': 1,
             'trade_group_id': 101, 'instrument_id': 1}
        ])

        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(signals)
        self.assertNotIn(1, bt.pending_orders)


    def test_cancel_invalid_trade_id(self):
        signals = pd.DataFrame([
            {'submission_time': self.price_df.loc[0, 'datetime'], 'action': 'cancel_order', 'trade_id': 999, 'trade_group_id':102, 'instrument_id':1}
        ])
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        with self.assertRaises(ValueError):
            bt.run(signals)

    def test_modify_order_to_execute(self):
        signals = pd.DataFrame([
            self.limit_signal(self.price_df.loc[0, 'datetime'], 'buy_limit', 1, 98.0, trade_id=2, trade_group_id=102, instrument_id=1),
            {'submission_time': self.price_df.loc[0, 'datetime'], 'action': 'modify_order', 'trade_id': 2, 'trade_group_id':102, 'instrument_id':1, 'limit_price': 100.0}
        ])
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(signals)
        history = bt.get_trade_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]['action'], 'buy')

    def test_modify_invalid_trade_id(self):
        signals = pd.DataFrame([
            {'submission_time': self.price_df.loc[0, 'datetime'], 'action': 'modify_order', 'trade_id': 404,'trade_group_id':103, 'instrument_id':1, 'limit_price': 100.0}
        ])
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        with self.assertRaises(ValueError):
            bt.run(signals)

    def test_modify_order_quantity(self):
        signals = pd.DataFrame([
            self.limit_signal(self.price_df.loc[0, 'datetime'], 'buy_limit', 1, 102.0, trade_id=3, trade_group_id=103, instrument_id=1),
            {'submission_time': self.price_df.loc[0, 'datetime'], 'action': 'modify_order', 'trade_id': 3, 'trade_group_id':103, 'instrument_id':1, 'qty': 2, 'limit_price': 105.0}
        ])
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(signals)
        history = bt.get_trade_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]['qty'], 2)

    def test_modify_order_expiration(self):
        initial_exp = self.price_df.loc[1, 'datetime']
        extended_exp = self.price_df.loc[4, 'datetime'] + timedelta(hours=10)
        signals = pd.DataFrame([
            self.limit_signal(self.price_df.loc[0, 'datetime'], 'buy_limit', 1, 10.0, trade_id=4, trade_group_id=104, expiration=initial_exp, instrument_id=1),
            {'submission_time': self.price_df.loc[0, 'datetime'], 'action': 'modify_order', 'trade_id': 4, 'trade_group_id':104, 'instrument_id':1, 'expiration': extended_exp}
        ])
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(signals)
        self.assertIn(4, bt.order_manager.pending_orders)
        self.assertEqual(bt.order_manager.pending_orders[4].signal['expiration'], extended_exp)

    def test_order_expires_before_execution(self):
        dt = self.price_df.loc[0, 'datetime']
        expiration = datetime(2022, 12, 31, 23, 59)
        signal = {
            'submission_time': dt,
            'action': 'buy_limit',
            'qty': 1,
            'limit_price': 99.0,
            'trade_id': 99,
            'trade_group_id': 99,
            'instrument_id': 1,
            'expiration': expiration
        }
        bt = Backtester(self.price_dict)
        bt.run(pd.DataFrame([signal]))
        self.assertEqual(len(bt.get_trade_history()), 0)

    def test_replace_order(self):
        original = self.limit_signal(self.price_df.loc[0, 'datetime'], 'buy_limit', 1, 98.0, trade_id=5,
                                     trade_group_id=105, instrument_id=1)
        replacement = self.limit_signal(self.price_df.loc[0, 'datetime'], 'buy_limit', 2, 101.0, trade_id=5,
                                        trade_group_id=105, instrument_id=1)
        replacement['action'] = 'replace_order'

        signals = pd.DataFrame([original, replacement])
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(signals)

        history = bt.get_trade_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]['qty'], 2)
        self.assertEqual(history.iloc[0]['price'], 101.0)

    def test_has_order(self):
        signal = self.limit_signal(self.price_df.loc[0, 'datetime'], 'buy_limit', 1, 80.0, trade_id=6,
                                   trade_group_id=106, instrument_id=1)
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(pd.DataFrame([signal]))
        self.assertTrue(bt.order_manager.has_order(6))

    def test_get_pending_order(self):
        signal = self.limit_signal(self.price_df.loc[0, 'datetime'], 'buy_limit', 1, 80.0, trade_id=7,
                                   trade_group_id=107, instrument_id=1)

        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(pd.DataFrame([signal]))
        order = bt.order_manager.get_pending_order(7)
        self.assertIsNotNone(order)
        self.assertEqual(order.signal['qty'], 1)
        self.assertEqual(order.signal['limit_price'], 80.0)


class TestBacktesterStopOrders(unittest.TestCase):
    def setUp(self):
        base_time = datetime(2023, 1, 1, 9)
        self.price_df = pd.DataFrame([
            {'datetime': base_time + timedelta(hours=i), 'open': 100 + i, 'high': 101 + i, 'low': 99 + i, 'close': 101+i}
            for i in range(5)
        ])
        self.price_dict = {
            'instrument_id': 1,
            'instrument_type': 'equity',
            'price_df': self.price_df
        }

    @staticmethod
    def stop_signal(dt, action, qty, stop_price, trade_id, trade_group_id, instrument_id, limit_price=None, expiration=None):
        if expiration is None:
            expiration = dt + timedelta(days=180)
        signal = {
            'submission_time': dt,
            'action': action,
            'qty': qty,
            'stop_price': stop_price,
            'trade_id': trade_id,
            'trade_group_id': trade_group_id,
            'instrument_id': instrument_id,
            'expiration': expiration
        }
        if limit_price is not None:
            signal['limit_price'] = limit_price
        return signal

    def test_buy_stop_executes(self):
        signal_df = pd.DataFrame([
            self.stop_signal(self.price_df.loc[0, 'datetime'], 'buy_stop', 1, 100.5, 1, 1001, 1)
        ])
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(signal_df)
        history = bt.get_trade_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]['action'], 'buy')

    def test_sell_stop_executes(self):
        bt = Backtester(self.price_dict, starting_cash=0.0)
        bt.portfolio['position'] = 1

        signal_df = pd.DataFrame([
            self.stop_signal(self.price_df.loc[0, 'datetime'], 'sell_stop', 1, 100.5, 2, 1002, 1)
        ])
        bt.run(signal_df)
        history = bt.get_trade_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]['action'], 'sell')

    def test_buy_stop_limit_triggers_to_limit(self):
        signal_df = pd.DataFrame([
            self.stop_signal(self.price_df.loc[0, 'datetime'], 'buy_stop_limit', 1, 100.5, 3, 1003, 1, limit_price=101.0)
        ])
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(signal_df)
        history = bt.get_trade_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]['action'], 'buy')
        self.assertLessEqual(history.iloc[0]['price'], 101.0)

    def test_sell_stop_limit_triggers_to_limit(self):
        bt = Backtester(self.price_dict, starting_cash=0.0)
        bt.portfolio['position'] = 1

        signal_df = pd.DataFrame([
            self.stop_signal(self.price_df.loc[0, 'datetime'], 'sell_stop_limit', 1, 100.5, 4, 1004, 1, limit_price=100.0)
        ])
        bt.run(signal_df)
        history = bt.get_trade_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]['action'], 'sell')
        self.assertGreaterEqual(history.iloc[0]['price'], 100.0)


class TestBacktesterRequiredFields(unittest.TestCase):
    def setUp(self):
        base_time = datetime(2023, 1, 1, 9)
        self.price_df = pd.DataFrame([
            {'datetime': base_time + timedelta(hours=i), 'open': 100 + i, 'high': 101 + i, 'low': 99 + i, 'close': 101+i}
            for i in range(5)
        ])
        self.price_dict = {"instrument_id": 1, "instrument_type": "equity", "price_df": self.price_df}


    @staticmethod
    def buy_signal(dt, qty, trade_id, trade_group_id, instrument_id):
        return {"submission_time": dt, "action": "buy", "qty": qty, "trade_id": trade_id,
                "trade_group_id": trade_group_id, "instrument_id": instrument_id}

    @staticmethod
    def sell_signal(dt, qty, trade_id, trade_group_id, instrument_id):
        return {"submission_time": dt, "action": "sell", "qty": qty, "trade_id": trade_id,
                "trade_group_id": trade_group_id, "instrument_id": instrument_id}

    def test_trade_group_id_propagation(self):
        signal_df = pd.DataFrame([
            self.buy_signal(self.price_df.loc[0, 'datetime'], 1, trade_id=1, trade_group_id=42, instrument_id=1),
            self.sell_signal(self.price_df.loc[1, 'datetime'], 1, trade_id=2, trade_group_id=42, instrument_id=1)
        ])
        bt = Backtester(self.price_dict, starting_cash=1000.0)
        bt.run(signal_df)
        history = bt.get_trade_history()
        self.assertEqual(len(history), 2)
        self.assertTrue(all(history['trade_group_id'] == 42))

    def test_invalid_id_type(self):
        signal = {
            'submission_time': self.price_df.loc[0, 'datetime'],
            'action': 'buy',
            'qty': 1,
            'trade_id': 'not-an-int',
            'trade_group_id': 1,
            'instrument_id': 1
        }
        bt = Backtester(self.price_dict)
        with self.assertRaises(TypeError):
            bt.run(pd.DataFrame([signal]))

    def test_missing_required_field(self):
        signal = {
            'submission_time': self.price_df.loc[0, 'datetime'], 'action': 'buy', 'qty': None
        }

        bt = Backtester(self.price_dict)
        with self.assertRaises(ValueError):
            bt.run(pd.DataFrame([signal]))

    def test_duplicate_trade_id_raises(self):
        signal_df = pd.DataFrame([
            {
                'submission_time': self.price_df.loc[0, 'datetime'],
                'action': 'buy',
                'qty': 1,
                'trade_id': 1,
                'trade_group_id': 1,
                'instrument_id': 1
            },
            {
                'submission_time': self.price_df.loc[0, 'datetime'],
                'action': 'buy',
                'qty': 1,
                'trade_id': 1,  # Duplicate trade_id
                'trade_group_id': 1,
                'instrument_id': 1
            }
        ])
        bt = Backtester(self.price_dict)
        with self.assertRaises(ValueError):
            bt.run(signal_df)


if __name__ == '__main__':
    unittest.main()
