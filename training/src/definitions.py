# Constants and Feature Definitions

LEVEL_FEATURE_COUNT = 3

LEVEL_OFFSETS_COLUMNS = [
    f"bid_offset_level_{k}_ticks" for k in range(1, LEVEL_FEATURE_COUNT + 1)
] + [f"ask_offset_level_{k}_ticks" for k in range(1, LEVEL_FEATURE_COUNT + 1)]

LEVEL_SIZE_COLUMNS = [
    f"bid_size_level_{k}_rel" for k in range(1, LEVEL_FEATURE_COUNT + 1)
] + [f"ask_size_level_{k}_rel" for k in range(1, LEVEL_FEATURE_COUNT + 1)]

CORE_FEATURE_COLUMNS = [
    "mid_return_bar",
    "spread_ticks",
    "spread_change_ticks",
    "mid_range_rel",
    "imbalance_best",
    "cum_bid_size_l_rel",
    "cum_ask_size_l_rel",
    "imbalance_l",
    "trade_volume_sum_rel",
    "trade_count_log",
    "rv_log",
]

GROUP_D_COLUMNS = [
    "limit_add_bid_volume_rel",
    "limit_add_ask_volume_rel",
    "limit_cancel_bid_volume_rel",
    "limit_cancel_ask_volume_rel",
    "limit_of_imbalance",
]

OFI_COLUMNS = [
    "ofi_bid",
    "ofi_ask",
    "ofi_net_log",
]

AGGRESSOR_COLUMNS = [
    "buy_trade_volume_rel",
    "sell_trade_volume_rel",
    "trade_imbalance_ratio",
    "avg_buy_dist_to_ask",
    "avg_sell_dist_to_bid",
    "has_buy_trade",
    "has_sell_trade",
]

PRESENCE_COLUMNS = [
    "bid_level_2_present",
    "bid_level_3_present",
    "ask_level_2_present",
    "ask_level_3_present",
]

PHASE4_FEATURE_COLUMNS = (
    CORE_FEATURE_COLUMNS
    + GROUP_D_COLUMNS
    + LEVEL_OFFSETS_COLUMNS
    + LEVEL_SIZE_COLUMNS
    + PRESENCE_COLUMNS
)

PHASE5_FEATURE_COLUMNS = PHASE4_FEATURE_COLUMNS + OFI_COLUMNS + AGGRESSOR_COLUMNS

FEATURE_SET_COLUMNS = {
    "phase4": PHASE4_FEATURE_COLUMNS,
    "phase5": PHASE5_FEATURE_COLUMNS,
}

PRICE_COLUMNS = ["mid_close_price", "mid_high_price", "mid_low_price"]
TARGET_CLASS_VALUES = [-1, 1]
NUM_TARGET_CLASSES = 2
UP_CLASS_INDEX = 1
MIN_STD = 1e-6