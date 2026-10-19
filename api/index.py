import json
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import yfinance as yf


AVAILABLE_TICKERS = {"AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"}


def _field_frame(data: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        level = next(
            (
                index
                for index in range(data.columns.nlevels)
                if field in data.columns.get_level_values(index)
            ),
            None,
        )
        if level is None:
            return pd.DataFrame(index=data.index, columns=tickers, dtype=float)
        values = data.xs(field, axis=1, level=level)
    elif field in data.columns:
        values = data[field]
    else:
        return pd.DataFrame(index=data.index, columns=tickers, dtype=float)

    if isinstance(values, pd.Series):
        values = values.to_frame(name=tickers[0])
    return values.reindex(columns=tickers)


def _json_safe_frame(frame: pd.DataFrame) -> dict[str, list[float | None]]:
    result: dict[str, list[float | None]] = {}
    for ticker in frame.columns:
        result[str(ticker)] = [
            float(value) if pd.notna(value) else None
            for value in frame[ticker]
        ]
    return result


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        if parsed_url.path != "/api/stocks" and parsed_url.path != "/api/index.py":
            self._respond(404, {"error": "Endpoint not found."})
            return

        query = parse_qs(parsed_url.query)
        tickers = query.get("tickers", ["AAPL,MSFT,GOOGL"])[0].split(",")
        tickers = list(dict.fromkeys(ticker.strip().upper() for ticker in tickers))
        unknown_tickers = sorted(set(tickers) - AVAILABLE_TICKERS)
        if not tickers or unknown_tickers:
            self._respond(
                400,
                {
                    "error": "Choose one or more supported stock tickers.",
                    "unsupported_tickers": unknown_tickers,
                },
            )
            return

        try:
            start_date = date.fromisoformat(query.get("start", [""])[0])
            end_date = date.fromisoformat(query.get("end", [""])[0])
        except ValueError:
            self._respond(400, {"error": "Start and end must be ISO dates (YYYY-MM-DD)."})
            return

        if start_date > end_date:
            self._respond(400, {"error": "Start date must be on or before end date."})
            return
        if end_date > date.today():
            self._respond(400, {"error": "End date cannot be in the future."})
            return

        try:
            downloaded = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date + timedelta(days=1),
                progress=False,
                auto_adjust=False,
                group_by="column",
                threads=True,
            )
        except Exception as error:
            self._respond(502, {"error": f"Unable to download stock data: {error}"})
            return

        if downloaded.empty:
            self._respond(404, {"error": "No stock data was found for these filters."})
            return

        adjusted_close = _field_frame(downloaded, "Adj Close", tickers).dropna(
            how="all"
        )
        volumes = _field_frame(downloaded, "Volume", tickers).reindex(
            adjusted_close.index
        )
        if adjusted_close.empty:
            self._respond(404, {"error": "No adjusted closing-price data was returned."})
            return

        response: dict[str, Any] = {
            "dates": [timestamp.strftime("%Y-%m-%d") for timestamp in adjusted_close.index],
            "tickers": tickers,
            "adjusted_close": _json_safe_frame(adjusted_close),
            "volume": _json_safe_frame(volumes),
        }
        self._respond(200, response)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Allow", "GET, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "s-maxage=300, stale-while-revalidate=600")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
