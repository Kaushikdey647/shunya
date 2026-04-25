from __future__ import annotations

import jax.numpy as jnp


def alpha(ctx) -> jnp.ndarray:
    """
    Cross-sectional quality tilt from statement-backed fundamentals (e.g. yfinance):
    high ROE / operating margin / ROA / FCF, low debt-to-equity.
    """
    roe = ctx.feature("Return_On_Equity")
    opm = ctx.feature("Operating_Margin")
    roa = ctx.feature("Return_On_Assets")
    fcf = ctx.feature("Free_Cash_Flow")
    de = ctx.feature("Debt_To_Equity")
    z_roe = ctx.cs.zscore(roe)
    z_opm = ctx.cs.zscore(opm)
    z_roa = ctx.cs.zscore(roa)
    z_fcf = ctx.cs.zscore(fcf)
    z_de = ctx.cs.zscore(de)
    raw = z_roe + z_opm + z_roa + z_fcf - z_de
    return jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(jnp.float32)
