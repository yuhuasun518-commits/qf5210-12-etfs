# Dashboard validation

Validated on 28 October 2026 using the uploaded snapshot.

- `python -m pytest -q`: **30 tests passed**, including project downloads and shared-link access.
- Streamlit AppTest exercised each of the **18 period/portfolio combinations**, plus an empty comparison selection in each period.
- All **144 numeric fields** (8 fields × 6 portfolios × 3 periods) reconciled to the original performance table within CSV-rounding tolerance.
- Tests cover first-day drawdown, period rebasing, order counts, missing BIL dates, zero volatility/downside, and monthly compounding with missing months.
- Both input hashes match the supplied backtest manifest. Daily NAV and weights pass consistency checks.
- All **53 original files** match the uploaded ZIP byte for byte.
- The full-project ZIP contains every one of the 67 allowlisted project files, checked byte for byte. Runtime secrets and environment files are excluded; unsafe paths and symlinks are rejected.
- Hosted access stops before rendering metrics, selectors, or downloads when the shared-link token is absent, invalid, or not configured. A valid token opens both the dashboard and project downloads. Local mode requires an explicit server-side environment flag.

Validated dependencies are pinned in the root requirements files. The original strategy was not rerun; it is not needed to display the archived results. The dashboard reports the supplied causality results as archived checks rather than representing them as newly executed tests.

Testing contains only 17 sessions. Sortino is a new dashboard calculation with an explicit BIL-based downside definition; it is not part of the original performance table.

Hosting still requires Streamlit account sign-in, app deployment, a random token in Streamlit Secrets, and verification of the deployed link. Local tests do not establish that a live deployment exists.
