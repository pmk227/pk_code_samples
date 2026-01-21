"""
Trading Cost Model - Comprehensive Fee Calculation System

This module provides a complete framework for calculating trading costs including:
- Regulatory fees (SEC, FINRA, CAT)
- Exchange fees (clearing, pass-through, liquidity-based)
- Broker commissions (tiered and fixed pricing)
- Slippage modeling

The system is designed to accurately model real-world trading costs for equity transactions,
particularly for Interactive Brokers (IBKR) fee structures and NYSE exchange fees.
"""

from abc import ABC, abstractmethod


class RegulatoryFees(ABC):
    """
    Abstract base class for regulatory fee calculations.

    Regulatory fees are government-mandated charges imposed on securities transactions
    to fund market oversight and regulation activities.
    """

    @abstractmethod
    def full_fee_breakdown(self, price, qty, action):
        pass


class EquityRegFees(RegulatoryFees):
    """
    Implementation of regulatory fees for equity transactions.

    Calculates SEC fees, FINRA trading activity fees, and CAT fees
    based on current regulatory fee schedules as of 2024.
    """

    def sec_fee(self, price, qty):
        """
         Calculate SEC (Securities and Exchange Commission) fee.

         SEC fees are charged on sell transactions only, calculated as a percentage
         of the transaction value to fund SEC operations.

         Args:
             price (float): Price per share
             qty (int): Number of shares

         Returns:
             float: SEC fee amount (currently $0.0000278 per dollar of transaction value)
         """

        return 0.0000278 * price * qty

    def finra_trading_fee(self, qty):
        """
        Calculate FINRA Trading Activity Fee (TAF).

        FINRA TAF is charged on sell transactions with a per-share rate
        and a maximum cap per trade.

        Args:
            qty (int): Number of shares

        Returns:
            float: FINRA TAF amount (currently $0.000166 per share, capped at $8.30)
        """

        return min(0.000166 * qty, 8.30)

    def cat_fee(self, qty):
        """
        Calculate CAT (Consolidated Audit Trail) fee.

        CAT fees fund the consolidated audit trail system for tracking
        all equity and options transactions across markets.

        Args:
            qty (int): Number of shares

        Returns:
            float: CAT fee amount (currently $0.000035 per share)
        """

        return 0.000035 * qty

    def full_fee_breakdown(self, price, qty, action):
        """
        Calculate complete regulatory fee breakdown for an equity trade.

        SEC and FINRA TAF fees only apply to sell transactions,
        while CAT fees apply to all transactions.

        Args:
            price (float): Price per share of the security
            qty (int): Number of shares traded
            action (str): Trade action ('buy' or 'sell')

        Returns:
            dict: Dictionary with keys:
                - 'sec_fees': SEC fee amount
                - 'finra_trading_fees': FINRA TAF amount
                - 'cat_fees': CAT fee amount
        """

        sec_fee = self.sec_fee(price, qty) if action == 'sell' else 0.0
        finra_fee = self.finra_trading_fee(qty) if action == 'sell' else 0.0
        cat_fee = self.cat_fee(qty)

        return {
            'sec_fees': sec_fee,
            'finra_trading_fees': finra_fee,
            'cat_fees': cat_fee
        }


class ExchangeFees(ABC):
    """
    Abstract base class for exchange-specific fee calculations.

    Exchange fees vary by trading venue and include clearing fees,
    pass-through regulatory fees, and liquidity-based pricing.
    """

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
    """
    Implementation of NYSE (New York Stock Exchange) fee structure.

    Includes NYSE-specific clearing fees, pass-through charges, and
    maker-taker pricing model for liquidity-based fees.
    """

    def clearing_fee(self, price, qty):
        """
        Calculate NYSE clearing fee with dual cap structure.

        NYSE clearing fees are charged per share with an additional
        cap based on trade value to prevent excessive fees on large trades.

        Args:
            price (float): Price per share
            qty (int): Number of shares

        Returns:
            float: NYSE clearing fee (lesser of $0.0002/share or 0.5% of trade value)
        """
        trade_value = price * qty
        return min(0.00020 * qty, 0.005 * trade_value)

    def pass_through_fees(self, commission):
        """
        Calculate NYSE pass-through regulatory fees.

        These are regulatory fees that NYSE passes through to customers
        based on the commission amount.

        Args:
            commission (float): Commission charged on the trade

        Returns:
            tuple: (nyse_passthrough, finra_passthrough) fee amounts
                - NYSE pass-through: 0.0175% of commission
                - FINRA pass-through: 0.056% of commission, capped at $8.30
        """
        nyse = 0.000175 * commission
        finra = min(0.00056 * commission, 8.30)
        return nyse, finra

    def exchange_fee(self, qty, liquidity_type):
        """
        Calculate NYSE liquidity-based trading fees.

        NYSE uses a maker-taker model where liquidity providers (makers)
        may receive rebates while liquidity takers are charged fees.

        Args:
            qty (int): Number of shares
            liquidity_type (str): 'add' for providing liquidity, 'remove' for taking liquidity

        Returns:
            float: Exchange fee amount
                - 'remove': $0.003 per share (taking liquidity)
                - 'add': $0.00 per share (providing liquidity - no fee/potential rebate)

        Raises:
            ValueError: If liquidity_type is not 'add' or 'remove'
        """
        if liquidity_type == 'remove':
            return 0.003 * qty
        elif liquidity_type == 'add':
            return 0.0
        else:
            raise ValueError(f"Unknown liquidity type: {liquidity_type}")


class BrokerFee(ABC):
    """
    Abstract base class for broker-specific fee calculations.

    Broker fees typically include commissions and may include
    additional exchange and regulatory pass-through charges.
    """

    @abstractmethod
    def commission(self, price, qty):
        pass

    @abstractmethod
    def full_fee_breakdown(self, price, qty, action, liquidity_type):
        pass


class IBKRFees(BrokerFee):
    """
    Interactive Brokers (IBKR) fee structure implementation.

    Supports both IBKR's tiered and fixed commission structures:
    - Tiered: Lower per-share rates with additional exchange fees
    - Fixed: Higher all-inclusive per-share rates

    Attributes:
        commission_type (str): Either 'tiered' or 'fixed'
        exchange_fees (NyseFees): Exchange fee calculator for tiered pricing
    """

    def __init__(self, commission_type):
        """
        Initialize IBKR fee calculator.

        Args:
            commission_type (str): Commission structure type ('tiered' or 'fixed')

        Raises:
            ValueError: If commission_type is not 'tiered' or 'fixed'
        """
        self.commission_type = commission_type
        self.exchange_fees = NyseFees()

    def commission(self, price, qty):
        """
        Calculate IBKR commission based on pricing tier.

        IBKR offers two commission structures:
        - Tiered: $0.0035/share (min $0.35, max 1% of trade value)
        - Fixed: $0.005/share (min $1.00, max 1% of trade value)

        Args:
            price (float): Price per share
            qty (int): Number of shares

        Returns:
            float: Commission amount based on selected tier

        Raises:
            ValueError: If commission_type is not supported
        """
        trade_value = price * qty
        if self.commission_type == 'tiered':
            return max(0.35, min(0.0035 * qty, 0.01 * trade_value))
        elif self.commission_type == 'fixed':
            return max(1.00, min(0.005 * qty, 0.01 * trade_value))
        else:
            raise ValueError(f"Unsupported commission type: {self.commission_type}")

    def full_fee_breakdown(self, price, qty, action, liquidity_type):
        """
        Calculate complete IBKR fee breakdown.

        For tiered pricing, includes commission plus all exchange fees.
        For fixed pricing, only includes the all-inclusive commission.

        Args:
            price (float): Price per share
            qty (int): Number of shares
            action (str): Trade action ('buy' or 'sell')
            liquidity_type (str): Liquidity type ('add' or 'remove')

        Returns:
            dict: Fee breakdown containing:
                - Always: 'commission_fees'
                - Tiered only: 'clearing_fees', 'nyse_pass_through_fees',
                  'finra_pass_through_fees', 'exchange_fees'
        """
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
    """
    Market impact and slippage modeling for trade execution.

    Simulates the price impact of trading by adjusting execution prices
    based on trade direction and a fixed slippage percentage.

    Attributes:
        slippage (float): Slippage rate as a decimal (e.g., 0.001 for 0.1%)
    """
    def __init__(self, slippage):
        """
        Initialize slippage model.

        Args:
            slippage (float): Slippage percentage (e.g., 0.1 for 0.1% slippage)
        """
        self.slippage = slippage / 100

    def apply(self, action, price):
        """
        Apply slippage adjustment to execution price.

        Buys experience positive slippage (higher execution price),
        sells experience negative slippage (lower execution price).

        Args:
            action (str): Trade action ('buy' or 'sell')
            price (float): Original order price

        Returns:
            float: Slippage-adjusted execution price
        """
        direction = 1 if action == 'buy' else -1
        return price * (1 + direction * self.slippage)


class TradeCostModel:
    """
    Comprehensive trading cost model combining all fee components.

    Integrates slippage, broker fees, and regulatory fees to provide
    a complete cost analysis for equity trades.

    Attributes:
        slippage_model (SlippageModel): Market impact model
        broker_fee (IBKRFees): Broker fee calculator
        reg_fees (EquityRegFees): Regulatory fee calculator
    """

    def __init__(self, commission_type='fixed', slippage=0.0):
        """
        Initialize the comprehensive trade cost model.

        Args:
            commission_type (str): IBKR commission type ('fixed' or 'tiered')
            slippage (float): Market impact slippage percentage (default 0.0)
        """
        self.slippage_model = SlippageModel(slippage)
        self.broker_fee = IBKRFees(commission_type)
        self.reg_fees = EquityRegFees()

    def apply(self, trade_effect):
        """
        Apply complete cost model to a trade and return updated trade details.

        Processes a trade dictionary to add slippage-adjusted pricing and
        comprehensive fee breakdown including total costs.

        Args:
            trade_effect (dict): Trade details containing:
                - 'action': Trade action ('buy' or 'sell')
                - 'liquidity_type': Liquidity provision ('add' or 'remove')
                - 'price': Original order price
                - 'qty': Number of shares

        Returns:
            dict: Updated trade_effect dictionary with added fields:
                - 'price': Slippage-adjusted execution price
                - Individual fee breakdowns (varies by commission type)
                - 'total_fees': Sum of all applicable fees

        Note:
            The input dictionary is modified in-place and also returned.
        """
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
