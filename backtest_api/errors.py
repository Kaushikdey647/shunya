"""Domain errors for backtest API (avoid FastAPI types in shared layers)."""


class FinTsConfigurationError(Exception):
    """Invalid or unavailable FinTs / market data configuration.

    Raised from :mod:`backtest_api.fin_ts_factory`; HTTP handlers map ``status_code``
    to :class:`fastapi.HTTPException`.
    """

    def __init__(self, message: str, *, status_code: int = 503) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
