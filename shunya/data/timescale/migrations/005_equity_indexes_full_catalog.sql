-- Full PyTickerSymbols index catalog (codes align with shunya.data.timescale.index_membership_sync).
-- Apply with: python -m shunya.data.timescale.cli migrate

INSERT INTO equity_indexes (code, display_name)
VALUES
    ('AEX', 'AEX'),
    ('BEL20', 'BEL 20'),
    ('CAC40', 'CAC_40'),
    ('CACMID60', 'CAC Mid 60'),
    ('DAX', 'DAX'),
    ('DOWJONES', 'DOW JONES'),
    ('EUROSTOXX50', 'EURO STOXX 50'),
    ('FTSE100', 'FTSE 100'),
    ('IBEX35', 'IBEX 35'),
    ('MDAX', 'MDAX'),
    ('NASDAQ100', 'NASDAQ 100'),
    ('NIKKEI225', 'NIKKEI 225'),
    ('OMXHEL25', 'OMX Helsinki 25'),
    ('OMXSTO30', 'OMX Stockholm 30'),
    ('SDAX', 'SDAX'),
    ('SP100', 'S&P 100'),
    ('SP500', 'S&P 500'),
    ('SP600', 'S&P 600'),
    ('SWISS20', 'Switzerland 20'),
    ('TECDAX', 'TecDAX')
ON CONFLICT (code) DO NOTHING;
