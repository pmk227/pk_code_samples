from abc import ABC, abstractmethod
from datetime import timedelta


class RegulatoryFees(ABC):
    @abstractmethod
    def full_fee_breakdown(self, price, qty, action):
        pass


class EquityRegFees(RegulatoryFees):
    def sec_fee(self, price, qty):
        return 0.0000278 * price * qty

    def finra_trading_fee(self, qty):
        return min(0.000166 * qty, 8.30)

    def cat_fee(self, qty):
        return 0.000035 * qty

    def full_fee_breakdown(self, price, qty, action):
        sec_fee = self.sec_fee(price, qty) if action == 'sell' else 0.0
        finra_fee = self.finra_trading_fee(qty) if action == 'sell' else 0.0
        cat_fee = self.cat_fee(qty)

        return {
            'sec_fees': sec_fee,
            'finra_trading_fees': finra_fee,
            'cat_fees': cat_fee
        }


class ExchangeFees(ABC):
    @abstractmethod
    def clearing_fee(self, price, qty):
        pass

    @abstractmethod
    def pass_through_fees(self, commission):
        pass

    @abstractmethod
    def exchange_fee(self, qty, liquidity_type):
        pass


class NyseFees(ExchangeFees):
    def clearing_fee(self, price, qty):
        trade_value = price * qty
        return min(0.00020 * qty, 0.005 * trade_value)

    def pass_through_fees(self, commission):
        nyse = 0.000175 * commission
        finra = min(0.00056 * commission, 8.30)
        return nyse, finra

    def exchange_fee(self, qty, liquidity_type):
        if liquidity_type == 'remove':
            return 0.003 * qty
        elif liquidity_type == 'add':
            return 0.0
        else:
            raise ValueError(f"Unknown liquidity type: {liquidity_type}")


class BrokerFee(ABC):
    @abstractmethod
    def commission(self, price, qty):
        pass

    @abstractmethod
    def full_fee_breakdown(self, price, qty, action, liquidity_type):
        pass


class IBKRFees(BrokerFee):
    def __init__(self, commission_type):
        self.commission_type = commission_type
        self.exchange_fees = NyseFees()

    def commission(self, price, qty):
        trade_value = price * qty
        if self.commission_type == 'tiered':
            return max(0.35, min(0.0035 * qty, 0.01 * trade_value))
        elif self.commission_type == 'fixed':
            return max(1.00, min(0.005 * qty, 0.01 * trade_value))
        else:
            raise ValueError(f"Unsupported commission type: {self.commission_type}")

    def full_fee_breakdown(self, price, qty, action, liquidity_type):
        commission = self.commission(price, qty)
        fee_dict = {'commission_fees': commission}

        if self.commission_type == 'tiered':
            clearing_fee = self.exchange_fees.clearing_fee(price, qty)
            nyse_pass, finra_pass = self.exchange_fees.pass_through_fees(commission)
            exchange_fee = self.exchange_fees.exchange_fee(qty, liquidity_type=liquidity_type)

            fee_dict.update({
                'clearing_fees': clearing_fee,
                'nyse_pass_through_fees': nyse_pass,
                'finra_pass_through_fees': finra_pass,
                'exchange_fees': exchange_fee
            })

        return fee_dict


class SlippageModel:
    def __init__(self, slippage):
        self.slippage = slippage / 100

    def apply(self, action, price):
        direction = 1 if action == 'buy' else -1
        return price * (1 + direction * self.slippage)


class TradeCostModel:
    def __init__(self, commission_type='fixed', slippage=0.0):
        self.slippage_model = SlippageModel(slippage)
        self.broker_fee = IBKRFees(commission_type)
        self.reg_fees = EquityRegFees()

    def apply(self, trade_effect):
        action = trade_effect['action']
        liquidity_type = trade_effect['liquidity_type']
        price = trade_effect['price']
        qty = trade_effect['qty']

        slip_adjusted_price = self.slippage_model.apply(action, price)

        broker_fees = self.broker_fee.full_fee_breakdown(slip_adjusted_price, qty, action, liquidity_type)
        reg_fees = self.reg_fees.full_fee_breakdown(slip_adjusted_price, qty, action)

        fee_dict = {**broker_fees, **reg_fees}
        fee_dict['total_fees'] = sum(fee_dict.values())

        trade_effect['price'] = slip_adjusted_price
        trade_effect.update(fee_dict)
        return trade_effect
