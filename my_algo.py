"""
Multi-Instrument Trading Bot
- Instruments: EURUSD, XAUUSD, GBPUSD, USDJPY, BTCUSDT, NAS100, US30, USOIL, GER30
- Strategy: Candlestick patterns + Support/Resistance + Volume + 1:2 RR + logical stops
- Execution: MT5 (forex/indices/commodities), Binance Futures / Delta Exchange (crypto)
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from dotenv import load_dotenv


# ---------- Broker APIs ----------
import MetaTrader5 as mt5                     # pip install MetaTrader5
import ccxt                                  # pip install ccxt
from telegram_service import send_vip_signal

# ---------- Configuration ----------
load_dotenv()
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API keys from environment (safer than hardcoding)
MT5_LOGIN = int(os.getenv("MT5_LOGIN", ""))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")
DELTA_API_KEY = os.getenv("DELTA_API_KEY", "")
DELTA_SECRET = os.getenv("DELTA_SECRET", "")


def _env_list(name: str, default):
    raw = os.getenv(name, "")
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


# Instruments grouped by broker
MT5_INSTRUMENTS = _env_list("MT5_INSTRUMENTS", [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "GBPAUD", "GBPCAD", "AUDCAD", "AUDNZD", "NZDJPY",
    "XAUUSD", "XAGUSD", "NAS100", "US30", "SPX500", "GER30", "USOIL",
])
CRYPTO_INSTRUMENTS = _env_list("CRYPTO_INSTRUMENTS", ["BTCUSDT", "ETHUSDT"])

# Strategy parameters
TIMEFRAME = mt5.TIMEFRAME_H1          # MT5 timeframe (H1 candle)
LOOKBACK_SR = 20                      # bars for support/resistance detection
VOLUME_MA_PERIOD = 20                 # volume moving average length
VOLUME_THRESHOLD = 1.2               # volume must be > average * threshold
RISK_PER_TRADE = 0.01                # 1% risk per trade
TP1_RATIO = 2.0                      # risk:reward for first take profit (1:2)
TP2_RATIO = 2.65                     # risk:reward for second take profit (1:2.65)
POSITION_SPLIT = 0.5                 # 50% of position at TP1, 50% at TP2

# Candlestick pattern tolerance (in pips/points)
DOJI_BODY_RATIO = 0.1                # body <= 10% of total range
ENGULFING_CONFIRM = True             # require previous trend condition


# ======================== Broker Interface ========================
class Broker:
    """Abstract broker for placing orders."""
    def __init__(self, name):
        self.name = name

    def get_balance(self) -> float:
        raise NotImplementedError

    def get_lot_size(self, symbol: str, risk_amount: float, stop_points: float) -> float:
        raise NotImplementedError

    def place_order(self, symbol: str, side: str, volume: float,
                    sl: float, tp1: float, tp2: float) -> bool:
        raise NotImplementedError


class MT5Broker(Broker):
    def __init__(self):
        super().__init__("MT5")
        if not mt5.initialize():
            raise ConnectionError(f"MT5 init failed: {mt5.last_error()}")
        if MT5_LOGIN and MT5_PASSWORD:
            if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
                raise ConnectionError(f"MT5 login failed: {mt5.last_error()}")
            logger.info("MT5 logged in successfully")

    def get_balance(self):
        acc = mt5.account_info()
        if acc:
            return acc.balance
        return 0

    def get_lot_size(self, symbol, risk_amount, stop_points):
        """Convert stop distance (points) and risk amount to MT5 volume."""
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return 0.0
        tick_value = symbol_info.trade_tick_value        # value of one point movement in account currency
        tick_size = symbol_info.trade_tick_size           # minimum price change
        if tick_value <= 0 or tick_size <= 0:
            return 0.0
        # lot size = risk_amount / (stop_points * tick_value / tick_size)
        # because tick_value is for one lot
        lot = risk_amount / (stop_points * (tick_value / tick_size))
        # Normalize to allowed step
        step = symbol_info.volume_step
        lot = round(lot / step) * step
        return max(symbol_info.volume_min, min(lot, symbol_info.volume_max))

    def place_order(self, symbol, side, volume, sl, tp1, tp2):
        """Place market order with SL and two TPs (using pending orders for TP2)."""
        # Determine order type
        if side == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid

        # Main order with SL and first TP
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp1,
            "deviation": 20,
            "magic": 123456,
            "comment": "AlgoBot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"MT5 order failed: {result.comment}")
            return False

        # Place second limit order for remaining position (TP2)
        # Opposite direction for the remaining volume
        second_vol = volume * (1 - POSITION_SPLIT)
        if second_vol <= 0:
            return True

        opposite_side = mt5.ORDER_TYPE_SELL if side == "BUY" else mt5.ORDER_TYPE_BUY
        limit_price = tp2
        request2 = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": second_vol,
            "type": opposite_side,
            "price": limit_price,
            "sl": 0.0,
            "tp": 0.0,
            "deviation": 20,
            "magic": 123456,
            "comment": "AlgoTP2",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        result2 = mt5.order_send(request2)
        if result2.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warning(f"TP2 limit order failed: {result2.comment}")
        return True


class CryptoBroker(Broker):
    """Crypto broker using CCXT (Binance Futures / Delta Exchange)."""
    def __init__(self, exchange_id, api_key, secret, passphrase=None):
        super().__init__(exchange_id)
        exchange_class = getattr(ccxt, exchange_id)
        config = {
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},  # for Binance Futures
        }
        if passphrase:
            config['password'] = passphrase   # for Delta Exchange
        self.exchange = exchange_class(config)
        # Load markets
        self.exchange.load_markets()
        logger.info(f"Connected to {exchange_id}")

    def get_balance(self):
        # Fetch futures balance (USDT)
        try:
            balance = self.exchange.fetch_balance()
            return balance['total']['USDT'] or 0.0
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return 0.0

    def get_lot_size(self, symbol: str, risk_amount: float, stop_points: float) -> float:
        """For crypto, return the contract quantity (in USDT or coins)."""
        market = self.exchange.market(symbol)
        # For linear futures (BTC/USDT:USDT), contract size is in USDT or coin depending on exchange
        # We'll calculate quantity based on risk percentage.
        # For simplicity, assume we want to risk 'risk_amount' USDT.
        # If stop_loss is stop_points (in price units), then position size = risk_amount / stop_points.
        # But need to consider minimum notional and contract size.
        if stop_points <= 0:
            return 0.0
        # For USDT-margined contracts, quantity is in number of contracts = risk_amount / stop_points
        qty = risk_amount / stop_points
        # Apply market limits
        min_qty = market['limits']['amount']['min'] or 0.0
        max_qty = market['limits']['amount']['max'] or qty
        step = market['precision']['amount']
        qty = max(min_qty, min(qty, max_qty))
        # Round to step
        qty = round(qty / step) * step if step else qty
        return qty

    def place_order(self, symbol: str, side: str, volume: float,
                    sl: float, tp1: float, tp2: float) -> bool:
        """Place market order with stop loss and take profit orders."""
        # For Binance Futures: use 'STOP_MARKET' for SL, 'TAKE_PROFIT_MARKET' for TP.
        # For Delta Exchange: use stop loss order.
        try:
            normalized_side = side.lower()
            # Main market order
            main_order = self.exchange.create_order(symbol, 'market', normalized_side, volume)
            logger.info(f"Main order placed: {main_order}")

            # Place SL order
            sl_side = 'sell' if normalized_side == 'buy' else 'buy'
            sl_order = self.exchange.create_order(
                symbol, 'STOP_MARKET', sl_side, volume,
                params={'stopPrice': sl, 'reduceOnly': True}
            )
            logger.info(f"SL order placed: {sl_order}")

            # Place TP1 order (50% of position)
            tp1_vol = volume * POSITION_SPLIT
            if tp1_vol > 0:
                tp1_order = self.exchange.create_order(
                    symbol, 'TAKE_PROFIT_MARKET', sl_side, tp1_vol,
                    params={'stopPrice': tp1, 'reduceOnly': True}
                )
                logger.info(f"TP1 order placed: {tp1_order}")

            # Place TP2 order (remaining 50%)
            tp2_vol = volume * (1 - POSITION_SPLIT)
            if tp2_vol > 0:
                tp2_order = self.exchange.create_order(
                    symbol, 'TAKE_PROFIT_MARKET', sl_side, tp2_vol,
                    params={'stopPrice': tp2, 'reduceOnly': True}
                )
                logger.info(f"TP2 order placed: {tp2_order}")
            return True
        except Exception as e:
            logger.error(f"Crypto order error: {e}")
            return False


# ======================== Data Helpers ========================
def fetch_mt5_rates(symbol: str, count: int = 100):
    """Fetch historical candles from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, count)
    if rates is None:
        logger.error(f"Failed to get rates for {symbol}: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close', 'tick_volume']]

def fetch_crypto_rates(exchange, symbol: str, count: int = 100):
    """Fetch OHLCV from crypto exchange."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=count)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.error(f"Failed to fetch crypto rates: {e}")
        return None


# ======================== Strategy Logic ========================
def detect_candlestick_patterns(df: pd.DataFrame) -> Optional[str]:
    """
    Detect bullish/bearish reversal patterns.
    Returns 'BUY', 'SELL', or None.
    """
    if len(df) < 2:
        return None
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # Doji: small body relative to range
    body = abs(curr['close'] - curr['open'])
    range_hl = curr['high'] - curr['low']
    if range_hl > 0 and body / range_hl <= DOJI_BODY_RATIO:
        return None  # Doji alone not a signal; you may combine with trend later

    # Bullish Engulfing
    if (prev['close'] < prev['open'] and                     # previous red
        curr['close'] > curr['open'] and                    # current green
        curr['open'] <= prev['close'] and
        curr['close'] >= prev['open']):
        return 'BUY'

    # Bearish Engulfing
    if (prev['close'] > prev['open'] and                     # previous green
        curr['close'] < curr['open'] and                    # current red
        curr['open'] >= prev['close'] and
        curr['close'] <= prev['open']):
        return 'SELL'

    # Hammer (bullish reversal at downtrend)
    body_low = min(curr['open'], curr['close'])
    body_high = max(curr['open'], curr['close'])
    lower_wick = body_low - curr['low']
    upper_wick = curr['high'] - body_high
    if (lower_wick >= 2 * body and upper_wick <= body * 0.2
            and body > 0):
        # Confirm previous trend down (simple: prev close < prev open)
        if prev['close'] < prev['open']:
            return 'BUY'

    # Shooting Star (bearish reversal at uptrend)
    if (upper_wick >= 2 * body and lower_wick <= body * 0.2
            and body > 0):
        if prev['close'] > prev['open']:
            return 'SELL'

    return None

def find_support_resistance(df: pd.DataFrame, window=LOOKBACK_SR):
    """
    Find recent swing highs and lows as S/R levels.
    Returns (support, resistance) prices.
    """
    highs = df['high'].values
    lows = df['low'].values
    n = len(highs)
    if n < window:
        return None, None

    # Simple rolling min/max for recent swing points
    recent_highs = []
    recent_lows = []
    for i in range(n - window, n - 1):  # skip current forming candle
        # local maximum if high > neighbours within window/2
        half = window // 2
        start = max(0, i - half)
        end = min(n, i + half + 1)
        if highs[i] == max(highs[start:end]):
            recent_highs.append(highs[i])
        if lows[i] == min(lows[start:end]):
            recent_lows.append(lows[i])

    support = max(recent_lows) if recent_lows else None
    resistance = min(recent_highs) if recent_highs else None
    return support, resistance

def volume_condition(df: pd.DataFrame) -> bool:
    """Check if current volume exceeds moving average."""
    if len(df) < VOLUME_MA_PERIOD + 1:
        return False
    vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
    avg_vol = df[vol_col].rolling(VOLUME_MA_PERIOD).mean().iloc[-2]  # exclude current bar
    current_vol = df[vol_col].iloc[-1]
    return current_vol > avg_vol * VOLUME_THRESHOLD

def calculate_stop_and_targets(df: pd.DataFrame, direction: str,
                               support: float, resistance: float):
    """
    Determine logical stop loss and take profit levels.
    Returns (entry, stop_loss, tp1, tp2) or None.
    """
    if len(df) < 2:
        return None
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    if direction == 'BUY':
        # Stop below recent swing low or candle low (whichever is lower)
        stop_loss = min(support or curr['low'], curr['low'])
        # For extra safety: use the lowest of previous candle low as well
        stop_loss = min(stop_loss, prev['low'])
        entry_price = curr['close']
        risk = entry_price - stop_loss
        if risk <= 0:
            return None
        tp1 = entry_price + risk * TP1_RATIO
        tp2 = entry_price + risk * TP2_RATIO
    else:  # SELL
        stop_loss = max(resistance or curr['high'], curr['high'])
        stop_loss = max(stop_loss, prev['high'])
        entry_price = curr['close']
        risk = stop_loss - entry_price
        if risk <= 0:
            return None
        tp1 = entry_price - risk * TP1_RATIO
        tp2 = entry_price - risk * TP2_RATIO

    return entry_price, stop_loss, tp1, tp2

# ======================== Signal Aggregator ========================
def check_signal(df: pd.DataFrame) -> Optional[Dict]:
    """
    Apply all rules and return a trade signal dictionary.
    """
    # 1. Candlestick pattern
    pattern = detect_candlestick_patterns(df)
    if not pattern:
        return None

    # 2. Support / Resistance
    support, resistance = find_support_resistance(df)
    # For BUY, ensure price is near support (within ATR, for example)
    # For simplicity, we just check that the pattern occurred near S/R
    if pattern == 'BUY' and support is None:
        return None
    if pattern == 'SELL' and resistance is None:
        return None

    # 3. Volume confirmation
    if not volume_condition(df):
        return None

    # 4. Calculate stops & targets
    levels = calculate_stop_and_targets(df, pattern, support, resistance)
    if levels is None:
        return None

    entry, sl, tp1, tp2 = levels

    # 5. Minimum 1:2 risk-reward is automatically enforced by TP1_RATIO (2.0)

    return {
        'direction': pattern,
        'entry': entry,
        'stop_loss': sl,
        'take_profit1': tp1,
        'take_profit2': tp2,
    }


# ======================== Main Trading Loop ========================
def run_bot():
    """Main loop: check each instrument and execute if all rules match."""
    # Initialize brokers
    brokers: Dict[str, Broker] = {}
    if MT5_LOGIN:
        mt5_broker = MT5Broker()
        brokers['mt5'] = mt5_broker
    else:
        logger.warning("MT5 not configured; skipping MT5 instruments.")

    crypto_brokers = {}
    if BINANCE_API_KEY and BINANCE_SECRET:
        try:
            binance = CryptoBroker('binance', BINANCE_API_KEY, BINANCE_SECRET)
            crypto_brokers['binance'] = binance
        except Exception as e:
            logger.error(f"Binance init failed: {e}")

    if DELTA_API_KEY and DELTA_SECRET:
        try:
            delta = CryptoBroker('delta', DELTA_API_KEY, DELTA_SECRET)
            crypto_brokers['delta'] = delta
        except Exception as e:
            logger.error(f"Delta init failed: {e}")

    # Map instruments to their broker objects
    instrument_map = {}
    if 'mt5' in brokers:
        for sym in MT5_INSTRUMENTS:
            instrument_map[sym] = ('mt5', brokers['mt5'])

    # For crypto, decide which broker to use (default Binance)
    primary_crypto = 'binance' if 'binance' in crypto_brokers else 'delta' if 'delta' in crypto_brokers else None
    if primary_crypto:
        for sym in CRYPTO_INSTRUMENTS:
            instrument_map[sym] = (primary_crypto, crypto_brokers[primary_crypto])

    if not instrument_map:
        logger.error("No brokers/instruments configured. Exit.")
        return

    logger.info("Starting trading loop...")
    while True:
        try:
            for symbol, (broker_type, broker) in instrument_map.items():
                logger.debug(f"Checking {symbol} on {broker_type}")

                # Fetch data
                if broker_type == 'mt5':
                    df = fetch_mt5_rates(symbol, count=LOOKBACK_SR + VOLUME_MA_PERIOD + 10)
                else:
                    df = fetch_crypto_rates(broker.exchange, symbol,
                                            count=LOOKBACK_SR + VOLUME_MA_PERIOD + 10)

                if df is None or df.empty:
                    continue

                # Check strategy rules
                signal = check_signal(df)
                if not signal:
                    continue

                logger.info(f"SIGNAL on {symbol}: {signal}")

                # Compute risk amount (1% of balance)
                balance = broker.get_balance()
                risk_amount = balance * RISK_PER_TRADE

                # Determine stop distance in account's points
                stop_points = abs(signal['entry'] - signal['stop_loss'])
                if stop_points == 0:
                    continue

                # Calculate position size
                lot = broker.get_lot_size(symbol, risk_amount, stop_points)
                if lot <= 0:
                    logger.warning(f"Invalid lot size for {symbol}: {lot}")
                    continue

                # Execute trade
                side = 'BUY' if signal['direction'] == 'BUY' else 'SELL'
                success = broker.place_order(symbol, side, lot,
                                             signal['stop_loss'],
                                             signal['take_profit1'],
                                             signal['take_profit2'])

                if success:
                    logger.info(f"Trade executed: {side} {lot} {symbol} SL={signal['stop_loss']} TP1={signal['take_profit1']} TP2={signal['take_profit2']}")
                    send_vip_signal(
                        symbol=symbol,
                        side=side,
                        entry=signal['entry'],
                        sl=signal['stop_loss'],
                        tp1=signal['take_profit1'],
                        tp2=signal['take_profit2'],
                        source=f"my_algo {broker_type.upper()}",
                        timeframe=("H1", "30M", "45M","15M"),
                        status="EXECUTED",
                    )
                else:
                    logger.error("Trade execution failed.")

            # Wait for next candle (check every minute)
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_bot()
