from __future__ import annotations

import httpx
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

@dataclass(frozen=True)
class OptionContract:
    symbol: str          # OCC option symbol used by Alpaca
    underlying_symbol: str
    expiration_date: str # YYYY-MM-DD
    strike_price: float
    type: str            # call/put

class AlpacaOptionsContractsClient:
    """Lightweight REST client for Alpaca Options Contracts endpoint.

    Docs: GET /v2/options/contracts
    Note: We use httpx directly so you can control filtering and pagination.
    """

    def __init__(self, base_url: str, api_key: str, api_secret: str, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
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
        items = data.get("option_contracts", data.get("contracts", []))  # tolerate shape differences
        out: list[OptionContract] = []
        for it in items:
            out.append(
                OptionContract(
                    symbol=it.get("symbol") or it.get("option_symbol"),
                    underlying_symbol=it.get("underlying_symbol") or underlying,
                    expiration_date=it.get("expiration_date"),
                    strike_price=float(it.get("strike_price")),
                    type=str(it.get("type")).lower(),
                )
            )
        return out

def pick_atm_contract(
    contracts: list[OptionContract],
    *,
    underlying_price: float,
    strike_tolerance: float,
    dte_max: int,
    preferred_type: str,
) -> Optional[OptionContract]:
    """Pick a contract that is closest to ATM within strike tolerance and within DTE window."""
    if not contracts:
        return None

    now = datetime.now(timezone.utc).date()
    max_exp = now + timedelta(days=dte_max)

    filtered = []
    for c in contracts:
        try:
            exp = datetime.strptime(c.expiration_date, "%Y-%m-%d").date()
        except Exception:
            continue
        if exp > max_exp:
            continue
        if c.type != preferred_type.lower():
            continue
        # strike within tolerance
        if abs(c.strike_price - underlying_price) / underlying_price > strike_tolerance:
            continue
        filtered.append((exp, abs(c.strike_price - underlying_price), c))

    if not filtered:
        return None

    # Sort by earliest expiry then nearest strike
    filtered.sort(key=lambda t: (t[0], t[1]))
    return filtered[0][2]
