"""Single source of truth for feature names and window sizes.

Both the offline (batch, training) and online (streaming, serving) feature
code import these constants, which is the first line of defence against
training/serving skew. The second line is `tests/test_features.py`, which
asserts the two implementations produce identical values.
"""

WINDOWS_S = {"1h": 3_600, "24h": 86_400}

FEATURE_COLUMNS = [
    "amount",
    "amount_log",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_1h",
    "amount_sum_24h",
    "user_txn_count",
    "user_avg_amount",
    "amount_to_avg_ratio",
    "seconds_since_last_txn",
    "is_new_device",
    "is_new_location",
    "is_new_card",
    "device_changed",
    "location_changed",
    "device_user_count",
    "card_user_count",
    "hour_of_day",
    "channel_code",
]

CHANNEL_CODES = {"CARD_PRESENT": 0, "ONLINE": 1, "TRANSFER": 2, "CASH_OUT": 3, "PAYMENT": 4,
                 "CASH_IN": 5, "DEBIT": 6}

# Sentinel used when a user has no previous transaction.
NO_HISTORY_GAP_S = 30 * 86_400.0
