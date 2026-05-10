-- Extend instrument_yfinance_documents.resource_type for additional yfinance-backed caches.
-- Include option_iv_surface here (not only in 010): the API may cache IV rows before 010 runs,
-- so ADD CONSTRAINT must allow existing rows when upgrading from 008.

ALTER TABLE instrument_yfinance_documents
    DROP CONSTRAINT IF EXISTS instrument_yfinance_documents_resource_chk;

ALTER TABLE instrument_yfinance_documents
    ADD CONSTRAINT instrument_yfinance_documents_resource_chk CHECK (
        resource_type IN (
            'overview',
            'financials_income',
            'financials_balance',
            'financials_cashflow',
            'holders',
            'option_expirations',
            'option_chain',
            'option_iv_surface',
            'valuation_measures',
            'analyst_price_targets',
            'earnings_estimate',
            'revenue_estimate',
            'earnings_history',
            'eps_trend',
            'eps_revisions',
            'growth_estimates',
            'recommendations',
            'recommendations_summary',
            'upgrades_downgrades',
            'sustainability',
            'insider_purchases',
            'insider_transactions',
            'insider_roster_holders',
            'major_holders',
            'calendar',
            'sec_filings'
        )
    );
