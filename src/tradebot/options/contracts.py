from __future__ import annotations

import httpx
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from tradebot.core.logger import get_logger

log = get_logger("options.contracts")


@dataclass(frozen=True)
class OptionContract:
    symbol: str           # OCC option symbol used by Alpaca
    underlying_symbol: str
    expiration_date: str  # YYYY-MM-DD
    strike_price: float
    type: str             # call/put
    volume: int = 0       # Trading volume


def get_next_friday(from_date: datetime.date, skip_if_today: bool = True) -> datetime.date:
    """Get the next Friday expiration date.

    Args:
        from_date: Starting date
        skip_if_today: If True and from_date is Friday, return NEXT Friday (avoid 0 DTE)

    Returns:
        The target Friday date
    """
    # Monday=0, Friday=4
    days_until_friday = (4 - from_date.weekday()) % 7

    # If today is Friday and skip_if_today is True, go to next Friday
    if days_until_friday == 0 and skip_if_today:
        days_until_friday = 7

    return from_date + timedelta(days=days_until_friday)


class AlpacaOptionsContractsClient:
    """REST client for Alpaca Options Contracts and Market Data endpoints.

    Endpoints used:
    - GET /v2/options/contracts - List available contracts
    - GET /v1beta1/options/bars - Get volume data for contracts
    """

    # Data API base URL (different from trading API)
    DATA_API_URL = "https://data.alpaca.markets"

    def __init__(self, base_url: str, api_key: str, api_secret: str, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            },
        )

    def close(self) -> None:
        self.client.close()

    def list_contracts(
        self,
        underlying: str,
        *,
        contract_type: Optional[str] = None,  # "call" / "put"
        exp_gte: Optional[str] = None,
        exp_lte: Optional[str] = None,
        limit: int = 200,
    ) -> list[OptionContract]:
        """List available options contracts from Alpaca."""
        params = {
            "underlying_symbols": underlying,
            "limit": limit,
        }
        if contract_type:
            params["type"] = contract_type
        if exp_gte:
            params["expiration_date_gte"] = exp_gte
        if exp_lte:
            params["expiration_date_lte"] = exp_lte

        url = f"{self.base_url}/v2/options/contracts"
        r = self.client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        items = data.get("option_contracts", data.get("contracts", []))

        out: list[OptionContract] = []
        for it in items:
            out.append(
                OptionContract(
                    symbol=it.get("symbol") or it.get("option_symbol"),
                    underlying_symbol=it.get("underlying_symbol") or underlying,
                    expiration_date=it.get("expiration_date"),
                    strike_price=float(it.get("strike_price")),
                    type=str(it.get("type")).lower(),
                    volume=0,  # Volume fetched separately
                )
            )
        return out

    def get_options_volume(self, symbols: list[str]) -> dict[str, int]:
        """Fetch today's trading volume for a list of option symbols.

        Uses Alpaca's Options Bars endpoint to get volume data.

        Args:
            symbols: List of OCC option symbols

        Returns:
            Dict mapping symbol -> volume
        """
        if not symbols:
            return {}

        # Batch symbols (max 100 per request)
        volumes: dict[str, int] = {}
        batch_size = 100

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                batch_volumes = self._fetch_volume_batch(batch)
                volumes.update(batch_volumes)
            except Exception as e:
                log.warning(f"Failed to fetch volume for batch: {e}")
                # Default to 0 volume for failed batch
                for sym in batch:
                    volumes.setdefault(sym, 0)

        return volumes

    def get_latest_option_quote(self, symbol: str) -> Optional[dict]:
        """Fetch the latest quote for an option symbol."""
        url = f"{self.DATA_API_URL}/v1beta1/options/quotes"
        params = {
            "symbols": symbol,
            "limit": 1,
        }

        r = self.client.get(url, params=params)

        if r.status_code in (404, 422):
            return None

        r.raise_for_status()
        data = r.json()
        quotes = data.get("quotes", {})
        sym_quotes = quotes.get(symbol) or quotes.get(symbol.upper())

        if not sym_quotes:
            return None

        q = sym_quotes[-1]
        bid = q.get("bp") or q.get("bid_price") or q.get("bid")
        ask = q.get("ap") or q.get("ask_price") or q.get("ask")

        if bid is None or ask is None:
            return None

        try:
            return {"bid": float(bid), "ask": float(ask)}
        except (TypeError, ValueError):
            return None

    def _get_latest_option_bar_close(self, symbol: str) -> Optional[float]:
        """Fetch the latest bar close for an option symbol."""
        url = f"{self.DATA_API_URL}/v1beta1/options/bars"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        params = {
            "symbols": symbol,
            "timeframe": "1Min",
            "start": today,
            "limit": 1,
        }

        r = self.client.get(url, params=params)

        if r.status_code in (404, 422):
            return None

        r.raise_for_status()
        data = r.json()
        bars = data.get("bars", {})
        sym_bars = bars.get(symbol) or bars.get(symbol.upper())

        if not sym_bars:
            return None

        bar = sym_bars[-1]
        close = bar.get("c") or bar.get("close")
        try:
            return float(close)
        except (TypeError, ValueError):
            return None

    def get_latest_option_mid_price(self, symbol: str) -> Optional[float]:
        """Get a best-effort mid price for an option symbol."""
        quote = self.get_latest_option_quote(symbol)
        if quote:
            bid = quote.get("bid")
            ask = quote.get("ask")
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                return (bid + ask) / 2.0

        return self._get_latest_option_bar_close(symbol)

    def _fetch_volume_batch(self, symbols: list[str]) -> dict[str, int]:
        """Fetch volume for a batch of symbols."""
        # Use options trades endpoint for volume
        url = f"{self.DATA_API_URL}/v1beta1/options/bars"

        # Get today's bars
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        params = {
            "symbols": ",".join(symbols),
            "timeframe": "1D",
            "start": today,
            "limit": 1000,
        }

        r = self.client.get(url, params=params)

        if r.status_code == 404 or r.status_code == 422:
            # No data available, return 0 volumes
            return {sym: 0 for sym in symbols}

        r.raise_for_status()
        data = r.json()

        volumes: dict[str, int] = {}
        bars = data.get("bars", {})

        for sym in symbols:
            sym_bars = bars.get(sym, [])
            if sym_bars:
                # Sum up volume from all bars
                total_vol = sum(bar.get("v", 0) for bar in sym_bars)
                volumes[sym] = total_vol
            else:
                volumes[sym] = 0

        return volumes


def pick_friday_contract_by_volume(
    contracts: list[OptionContract],
    client: AlpacaOptionsContractsClient,
    *,
    underlying_price: float,
    strike_tolerance: float = 0.03,
    preferred_type: str = "call",
) -> Optional[OptionContract]:
    """Pick an options contract based on:

    1. Expiration is the nearest Friday (next Friday if today is Friday - avoid 0 DTE)
    2. Strike price within tolerance (default 3%) of underlying price
    3. Highest volume among filtered contracts

    Args:
        contracts: List of available contracts
        client: Options client for fetching volume data
        underlying_price: Current price of underlying
        strike_tolerance: Max % distance from current price (default 0.03 = 3%)
        preferred_type: "call" or "put"

    Returns:
        Best contract or None if no suitable contract found
    """
    if not contracts:
        return None

    today = datetime.now(timezone.utc).date()
    target_friday = get_next_friday(today, skip_if_today=True)
    target_friday_str = target_friday.strftime("%Y-%m-%d")

    log.info(f"Target expiration: {target_friday_str} (Friday)")

    # Step 1: Filter by expiration (exact Friday match)
    friday_contracts = [
        c for c in contracts
        if c.expiration_date == target_friday_str
    ]

    if not friday_contracts:
        # Try the following Friday if no contracts for target Friday
        next_friday = target_friday + timedelta(days=7)
        next_friday_str = next_friday.strftime("%Y-%m-%d")
        log.warning(f"No contracts for {target_friday_str}, trying {next_friday_str}")
        friday_contracts = [
            c for c in contracts
            if c.expiration_date == next_friday_str
        ]
        target_friday_str = next_friday_str

    if not friday_contracts:
        log.warning("No Friday expiration contracts found")
        return None

    # Step 2: Filter by type (call/put)
    type_filtered = [
        c for c in friday_contracts
        if c.type == preferred_type.lower()
    ]

    if not type_filtered:
        log.warning(f"No {preferred_type} contracts for {target_friday_str}")
        return None

    # Step 3: Filter by strike tolerance (within 3% of underlying price)
    min_strike = underlying_price * (1 - strike_tolerance)
    max_strike = underlying_price * (1 + strike_tolerance)

    strike_filtered = [
        c for c in type_filtered
        if min_strike <= c.strike_price <= max_strike
    ]

    if not strike_filtered:
        log.warning(
            f"No {preferred_type} contracts within {strike_tolerance*100:.0f}% "
            f"of ${underlying_price:.2f} (range: ${min_strike:.2f}-${max_strike:.2f})"
        )
        return None

    log.info(
        f"Found {len(strike_filtered)} {preferred_type}s for {target_friday_str} "
        f"within strike range ${min_strike:.2f}-${max_strike:.2f}"
    )

    # Step 4: Fetch volume data for filtered contracts
    symbols = [c.symbol for c in strike_filtered]
    volumes = client.get_options_volume(symbols)

    # Create contracts with volume data
    contracts_with_volume = []
    for c in strike_filtered:
        vol = volumes.get(c.symbol, 0)
        contracts_with_volume.append(
            OptionContract(
                symbol=c.symbol,
                underlying_symbol=c.underlying_symbol,
                expiration_date=c.expiration_date,
                strike_price=c.strike_price,
                type=c.type,
                volume=vol,
            )
        )

    # Step 5: Sort by volume (highest first), then by nearest strike as tiebreaker
    contracts_with_volume.sort(
        key=lambda c: (-c.volume, abs(c.strike_price - underlying_price))
    )

    best = contracts_with_volume[0]
    log.info(
        f"Selected: {best.symbol} | Strike: ${best.strike_price:.2f} | "
        f"Exp: {best.expiration_date} | Volume: {best.volume:,}"
    )

    return best


# Keep old function for backward compatibility but update to use new logic
def pick_atm_contract(
    contracts: list[OptionContract],
    *,
    underlying_price: float,
    strike_tolerance: float,
    dte_max: int,
    preferred_type: str,
    client: Optional[AlpacaOptionsContractsClient] = None,
) -> Optional[OptionContract]:
    """Pick a contract - uses Friday + volume selection if client provided.

    For backward compatibility, falls back to original logic if no client.
    """
    if client is not None:
        return pick_friday_contract_by_volume(
            contracts,
            client,
            underlying_price=underlying_price,
            strike_tolerance=strike_tolerance,
            preferred_type=preferred_type,
        )

    # Original fallback logic (no volume data)
    if not contracts:
        return None

    now = datetime.now(timezone.utc).date()
    target_friday = get_next_friday(now, skip_if_today=True)

    filtered = []
    for c in contracts:
        try:
            exp = datetime.strptime(c.expiration_date, "%Y-%m-%d").date()
        except Exception:
            continue
        # Only Friday expirations
        if exp != target_friday:
            continue
        if c.type != preferred_type.lower():
            continue
        # Strike within tolerance
        if abs(c.strike_price - underlying_price) / underlying_price > strike_tolerance:
            continue
        filtered.append((abs(c.strike_price - underlying_price), c))

    if not filtered:
        return None

    # Sort by nearest strike (no volume data available)
    filtered.sort(key=lambda t: t[0])
    return filtered[0][1]
