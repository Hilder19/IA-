-- Inserción en exchange_rates_history para el par EUR/USD
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'EUR/USD'),  -- ID del par EUR/USD
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par GBP/USD
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'GBP/USD'),  -- ID del par GBP/USD
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par USD/JPY
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'USD/JPY'),  -- ID del par USD/JPY
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par AUD/USD
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'AUD/USD'),  -- ID del par AUD/USD
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par USD/CAD
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'USD/CAD'),  -- ID del par USD/CAD
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par NZD/USD
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'NZD/USD'),  -- ID del par NZD/USD
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par CHF/JPY
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'CHF/JPY'),  -- ID del par CHF/JPY
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par USD/CHF
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'USD/CHF'),  -- ID del par USD/CHF
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par AUD/NZD
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'AUD/NZD'),  -- ID del par AUD/NZD
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par GBP/JPY
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'GBP/JPY'),  -- ID del par GBP/JPY
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par EUR/GBP
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'EUR/GBP'),  -- ID del par EUR/GBP
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par USD/SGD
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'USD/SGD'),  -- ID del par USD/SGD
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par USD/HKD
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'USD/HKD'),  -- ID del par USD/HKD
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par AUD/JPY
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'AUD/JPY'),  -- ID del par AUD/JPY
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par EUR/AUD
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'EUR/AUD'),  -- ID del par EUR/AUD
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par EUR/CHF
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'EUR/CHF'),  -- ID del par EUR/CHF
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par EUR/CAD
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'EUR/CAD'),  -- ID del par EUR/CAD
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par GBP/CAD
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'GBP/CAD'),  -- ID del par GBP/CAD
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par AUD/CAD
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'AUD/CAD'),  -- ID del par AUD/CAD
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par NZD/JPY
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'NZD/JPY'),  -- ID del par NZD/JPY
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);

-- Inserción en exchange_rates_history para el par GBP/NZD
INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
VALUES (
    (SELECT id FROM currency_pairs WHERE pair_name = 'GBP/NZD'),  -- ID del par GBP/NZD
    1.234567,    -- Tipo de cambio actual
    1.230000,    -- Precio de apertura
    1.240000,    -- Precio más alto
    1.220000,    -- Precio más bajo
    100000.00,   -- Volumen de operaciones
    NOW()        -- Fecha de la última actualización
);
