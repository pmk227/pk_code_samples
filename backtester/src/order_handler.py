from src.core.backtesting.order_factory import OrderFactory

class OrderManager:
    def __init__(self, pending_orders):
        self.pending_orders = {}

    def cancel_order(self, signal):
        trade_id = signal.get("trade_id")
        if trade_id not in self.pending_orders:
            raise ValueError(f"Trade ID {trade_id} not found for cancellation.")
        del self.pending_orders[trade_id]

    def cancel_all_orders(self):
        self.pending_orders.clear()

    def modify_order(self, signal):
        trade_id = signal.get("trade_id")
        if trade_id not in self.pending_orders:
            raise ValueError(f"Trade ID {trade_id} not found for modification.")

        order = self.pending_orders[trade_id]
        for key in ['qty', 'limit_price', 'stop_price', 'expiration']:
            if key in signal:
                order.signal[key] = signal[key]

    def replace_order(self, signal):
        trade_id = signal['trade_id']
        if trade_id not in self.pending_orders:
            raise ValueError(f"Cannot replace non-existent order {trade_id}")

        current_order = self.pending_orders[trade_id]
        original_action = current_order.signal['action']

        signal['action'] = original_action

        self.pending_orders[trade_id] = OrderFactory.create(signal)

    def has_order(self, trade_id):
        return trade_id in self.pending_orders

    def get_pending_order(self, trade_id):
        return self.pending_orders.get(trade_id)

    def store_pending_order(self, signal):
        trade_id = signal.get('trade_id')
        if trade_id in self.pending_orders:
            raise ValueError(f"Duplicate trade_id '{trade_id}' detected.")

        signal['entry_time'] = signal['submission_time']
        order = OrderFactory.create(signal)
        self.pending_orders[trade_id] = order

    def evaluate_orders(self, row):
        still_pending = {}
        executed_trades = []

        for trade_id, order in self.pending_orders.items():
            trade_effect = order.evaluate(row)
            if trade_effect:
                executed_trades.append(trade_effect)
            else:
                still_pending[trade_id] = order

        self.pending_orders = still_pending
        return executed_trades

    def process_signal(self, signal):
        action = signal['action']

        match action:
            case 'cancel_order':
                self.cancel_order(signal)
            case 'modify_order':
                self.modify_order(signal)
            case 'cancel_all_orders':
                self.cancel_all_orders()
            case 'replace_order':
                self.replace_order(signal)
            case _:
                self.store_pending_order(signal)

