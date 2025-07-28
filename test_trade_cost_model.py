import unittest
from src.core.backtesting.trade_cost_model import TradeCostModel

class TestFeeCalculator(unittest.TestCase):
    def setUp(self):
        self.price = 100.0
        self.qty = 10
        self.trade_value = self.price * self.qty

        self.model_tiered = TradeCostModel(commission_type="tiered")
        self.model_fixed = TradeCostModel(commission_type="fixed")

    # Component-Level Tests
    def test_equity_commission_tiered(self):
        fee = self.model_tiered.broker_fee.commission(self.price, self.qty)
        expected = max(0.35, min(0.0035 * self.qty, 0.01 * self.trade_value))
        self.assertAlmostEqual(fee, expected)

    def test_equity_commission_fixed(self):
        fee = self.model_fixed.broker_fee.commission(self.price, self.qty)
        expected = max(1.00, min(0.005 * self.qty, 0.01 * self.trade_value))
        self.assertAlmostEqual(fee, expected)

    def test_equity_reg_fees(self):
        fee_sell = self.model_fixed.reg_fees.full_fee_breakdown(self.price, self.qty, action='sell')
        fee_buy = self.model_fixed.reg_fees.full_fee_breakdown(self.price, self.qty, action='buy')

        # Sells
        self.assertAlmostEqual(fee_sell['sec_fees'], 0.0000278 * self.trade_value)
        self.assertAlmostEqual(fee_sell['finra_trading_fees'], 0.000166 * self.qty) # below cap
        self.assertAlmostEqual(fee_sell['cat_fees'], 0.000035 * self.qty)

        # Buys
        self.assertAlmostEqual(fee_buy['sec_fees'], 0)
        self.assertAlmostEqual(fee_buy['finra_trading_fees'], 0) # below cap
        self.assertAlmostEqual(fee_buy['cat_fees'], 0.000035 * self.qty)

        # Cap
        fee_sell = self.model_fixed.reg_fees.full_fee_breakdown(self.price, 1_000_000, action='sell')
        self.assertAlmostEqual(fee_sell['finra_trading_fees'], 8.30)

    def test_equity_clearing_fee_below_cap(self):
        fee = self.model_fixed.broker_fee.exchange_fees.clearing_fee(self.price, self.qty)
        expected = min(0.00020 * self.qty, 0.005 * self.trade_value)
        self.assertAlmostEqual(fee, expected)

    def test_equity_pass_through_fees(self):
        commission = 10.0
        nyse, finra = self.model_fixed.broker_fee.exchange_fees.pass_through_fees(commission)
        self.assertAlmostEqual(nyse, 0.000175 * commission)
        self.assertEqual(finra, min(0.00056 * commission, 8.30))

    def test_equity_exchange_fee_remove(self):
        fee = self.model_fixed.broker_fee.exchange_fees.exchange_fee(self.qty, "remove")
        self.assertEqual(fee, 0.003 * self.qty)

    def test_equity_exchange_fee_add(self):
        fee = self.model_fixed.broker_fee.exchange_fees.exchange_fee(self.qty, "add")
        self.assertEqual(fee, 0.0)

    # End-to-End Tests
    def test_equity_total_fees_buy_tiered(self):
        fees = self.model_tiered.broker_fee.full_fee_breakdown(self.price, self.qty, action='buy', liquidity_type="remove")
        self.assertAlmostEqual(sum(fees.values()), 0.38225725)

    def test_equity_total_fees_sell_fixed(self):
        fees = self.model_fixed.reg_fees.full_fee_breakdown(self.price, self.qty, action="sell")
        self.assertIn("sec_fees", fees)
        self.assertIn("finra_trading_fees", fees)
        self.assertIn("cat_fees", fees)
        self.assertAlmostEqual(sum(fees.values()), 0.02981)

    def test_equity_total_fees_buy_fixed(self):
        fees = self.model_fixed.reg_fees.full_fee_breakdown(self.price, self.qty, action="buy")
        self.assertIn("sec_fees", fees)
        self.assertIn("finra_trading_fees", fees)
        self.assertIn("cat_fees", fees)
        self.assertAlmostEqual(sum(fees.values()), 0.00035)

    def test_equity_total_fees_tiered_keys(self):
        trade_effects = {"action": "buy", "liquidity_type": "remove", "price": self.price, "qty": self.qty}
        fees = self.model_tiered.apply(trade_effects)
        expected_keys = {
            "commission_fees", "cat_fees", "clearing_fees",
            "nyse_pass_through_fees", "finra_pass_through_fees",
            "exchange_fees", "total_fees"
        }
        self.assertTrue(expected_keys.issubset(fees.keys()))

    def test_equity_total_fees_fixed_keys(self):
        trade_effects = {"action": "buy", "liquidity_type": "remove", "price": self.price, "qty": self.qty}
        fees = self.model_fixed.apply(trade_effects)
        expected_keys = {
            "commission_fees", "sec_fees", "finra_trading_fees", "cat_fees", "total_fees"
        }
        unexpected_keys = {
            "clearing_fees", "nyse_pass_through_fees", "finra_pass_through_fees", "exchange_fees"
        }
        self.assertTrue(expected_keys.issubset(fees.keys()))
        for key in unexpected_keys:
            self.assertNotIn(key, fees)

class TestFeeCalculatorEndToEnd(unittest.TestCase):
    def setUp(self):
        self.price = 100.0
        self.qty = 10
        self.trade_value = self.price * self.qty

        self.model_tiered = TradeCostModel(commission_type="tiered")
        self.model_fixed = TradeCostModel(commission_type="fixed")

    def test_fixed_buy_remove(self):
        trade_effects = {"action": "buy", "liquidity_type": "remove", "price": self.price, "qty": self.qty}
        fees = self.model_fixed.apply(trade_effects)
        self.assertIn("total_fees", fees)
        self.assertAlmostEqual(fees["total_fees"], 1.00035)

    def test_fixed_buy_add(self):
        trade_effects = {"action": "buy", "liquidity_type": "add", "price": self.price, "qty": self.qty}
        fees = self.model_fixed.apply(trade_effects)
        self.assertIn("total_fees", fees)
        self.assertAlmostEqual(fees["total_fees"], 1.00035)

    def test_fixed_sell_remove(self):
        trade_effects = {"action": "sell", "liquidity_type": "remove", "price": self.price, "qty": self.qty}
        fees = self.model_fixed.apply(trade_effects)
        self.assertIn("total_fees", fees)
        self.assertAlmostEqual(fees["total_fees"], 1.02981)

    def test_fixed_sell_add(self):
        trade_effects = {"action": "sell", "liquidity_type": "add", "price": self.price, "qty": self.qty}
        fees = self.model_fixed.apply(trade_effects)
        self.assertIn("total_fees", fees)
        self.assertAlmostEqual(fees["total_fees"], 1.02981)

    def test_tiered_buy_remove(self):
        trade_effects = {"action": "buy", "liquidity_type": "remove", "price": self.price, "qty": self.qty}
        fees = self.model_tiered.apply(trade_effects)
        self.assertIn("total_fees", fees)
        self.assertAlmostEqual(fees["total_fees"], .382607)

    def test_tiered_buy_add(self):
        trade_effects = {"action": "buy", "liquidity_type": "add", "price": self.price, "qty": self.qty}
        fees = self.model_tiered.apply(trade_effects)
        self.assertIn("total_fees", fees)
        self.assertAlmostEqual(fees["total_fees"], .35260725)

    def test_tiered_sell_remove(self):
        trade_effects = {"action": "sell", "liquidity_type": "remove", "price": self.price, "qty": self.qty}
        fees = self.model_tiered.apply(trade_effects)
        self.assertIn("total_fees", fees)
        self.assertAlmostEqual(fees["total_fees"], .41206725)

    def test_tiered_sell_add(self):
        trade_effects = {"action": "sell", "liquidity_type": "add", "price": self.price, "qty": self.qty}
        fees = self.model_tiered.apply(trade_effects)
        self.assertIn("total_fees", fees)
        self.assertAlmostEqual(fees["total_fees"], .382067)

if __name__ == '__main__':
    unittest.main()
