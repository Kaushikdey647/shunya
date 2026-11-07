"""Daily live trading: market-on-open runner, kill switch, and Postgres ledger.

Modules: :mod:`shunya.live.runner` (cron ``tick``), :mod:`shunya.live.control` (kill switch),
:mod:`shunya.live.broker` (guarded Alpaca access), :mod:`shunya.live.signals` (FinStrat replay),
:mod:`shunya.live.store` (``live_*`` tables), :mod:`shunya.live.cli` (``shunya-live``).
"""
