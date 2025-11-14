# Preprocessing

## Getting started with the Rust scaffold

-   Run `cargo run -- --dry-run` inside `preprocessing/` to inspect the configured resolutions without touching disk.
-   Override IO paths ad-hoc via `cargo run -- --input data/raw/training/TODO --output data/preprocessed/training/TODO`.
-   See `docs/ARCHITECTURE.md` for a walkthrough of the modules (`cli`, `config`, `domain`, `normalization`, `pipeline`, `io`, `utils`) and how they map to the schema below.

## 0\. Conventions

-   Resolutions:
    
    -   `fast` – 1 s or event-based bars
        
    -   `mid` – 10 s bars
        
    -   `slow` – 60 s bars
        
-   General:
    
    -   `p_bid_1`, `v_bid_1` = best bid price/size, `p_ask_1`, `v_ask_1` = best ask price/size.
        
    -   `mid = 0.5 * (p_bid_1 + p_ask_1)`
        
    -   `log_mid = ln(mid)`
        
    -   `ε` small constant, e.g. `1e-12`, to avoid division by zero.
        
    -   `L` = number of levels you keep per side (e.g. 5 or 10).
        
-   Naming pattern:
    
    -   All features are `snake_case`.
        
    -   Optional: prefix with group names in code (e.g. `top_`, `depth_`, `of_`, etc.) if you like.
        

---

## 1\. Top-of-book state (per bar)

**Applies to:** fast, mid, slow (identical structure)

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `best_bid_price` | float64 | F/M/S | `p_bid_1` at bar close |
| `best_ask_price` | float64 | F/M/S | `p_ask_1` at bar close |
| `best_bid_size` | float64 | F/M/S | `v_bid_1` at bar close |
| `best_ask_size` | float64 | F/M/S | `v_ask_1` at bar close |
| `mid_price` | float64 | F/M/S | `0.5 * (best_bid_price + best_ask_price)` |
| `log_mid_price` | float64 | F/M/S | `ln(mid_price)` |
| `spread_abs` | float64 | F/M/S | `best_ask_price - best_bid_price` |
| `spread_rel` | float64 | F/M/S | `spread_abs / mid_price` |
| `mid_open` | float64 | F/M/S | midprice at first event in bar |
| `mid_close` | float64 | F/M/S | midprice at last event in bar |
| `mid_high` | float64 | F/M/S | max midprice within bar |
| `mid_low` | float64 | F/M/S | min midprice within bar |
| `mid_return_bar` | float64 | F/M/S | `ln(mid_close) - ln(mid_open)` |
| `spread_open` | float64 | F/M/S | spread at first event in bar |
| `spread_close` | float64 | F/M/S | spread at last event in bar |
| `spread_change` | float64 | F/M/S | `spread_close - spread_open` |
| `best_bid_change` | float64 | F/M/S | `best_bid_price - best_bid_price_open` (store open value) |
| `best_ask_change` | float64 | F/M/S | `best_ask_price - best_ask_price_open` |

You will need to keep “open” values as you accumulate the bar.

---

## 2\. Depth profile (multi-level book snapshot at bar close)

**Applies to:**

-   fast: full L levels
    
-   mid: L levels or reduced (e.g. 3); you can still use same schema
    
-   slow: optional, you may keep only summaries
    

### 2.1. Raw level prices/sizes

For each `k = 1..L`:

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `bid_price_level_k` | float64 | F/M/S | Bid price at level k at bar close |
| `bid_size_level_k` | float64 | F/M/S | Bid size at level k at bar close |
| `ask_price_level_k` | float64 | F/M/S | Ask price at level k at bar close |
| `ask_size_level_k` | float64 | F/M/S | Ask size at level k at bar close |

(Implement in Rust as arrays `[L]` or flattened columns like `bid_price_level_1`, etc.)

### 2.2. Price offsets to mid (often better than raw prices)

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `bid_offset_level_k` | float64 | F/M/S | `mid_price - bid_price_level_k` |
| `ask_offset_level_k` | float64 | F/M/S | `ask_price_level_k - mid_price` |

### 2.3. Cumulative sizes

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `cum_bid_size_L` | float64 | F/M/S | `sum_{k=1..L} bid_size_level_k` |
| `cum_ask_size_L` | float64 | F/M/S | `sum_{k=1..L} ask_size_level_k` |
| `cum_bid_size_3` | float64 | F/M/S | `sum_{k=1..3} bid_size_level_k` (if L ≥ 3) |
| `cum_ask_size_3` | float64 | F/M/S | `sum_{k=1..3} ask_size_level_k` |

### 2.4. Imbalances

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `imbalance_best` | float64 | F/M/S | `(best_bid_size - best_ask_size) / (best_bid_size + best_ask_size + ε)` |
| `imbalance_3` | float64 | F/M/S | `(cum_bid_size_3 - cum_ask_size_3) / (cum_bid_size_3 + cum_ask_size_3 + ε)` |
| `imbalance_L` | float64 | F/M/S | `(cum_bid_size_L - cum_ask_size_L) / (cum_bid_size_L + cum_ask_size_L + ε)` |

### 2.5. Book shape / slope

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `book_slope_bid` | float64 | F/M/S | `(bid_price_level_L - bid_price_level_1) / (cum_bid_size_L + ε)` |
| `book_slope_ask` | float64 | F/M/S | `(ask_price_level_L - ask_price_level_1) / (cum_ask_size_L + ε)` |

Optionally, per-level gaps:

For `k = 1..L-1`:

-   `bid_gap_level_k = bid_price_level_k - bid_price_level_{k+1}`
    
-   `ask_gap_level_k = ask_price_level_{k+1} - ask_price_level_k`
    

---

## 3\. Order-flow (limit order adds/cancels) inside bar

You have Operations (Add/Update/Remove) per L2 event.

### 3.1. Volume flows at each side (aggregated over all levels)

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `limit_add_bid_volume` | float64 | F/M/S | Sum of volume added to bid side in bar |
| `limit_add_ask_volume` | float64 | F/M/S | Sum of volume added to ask side in bar |
| `limit_cancel_bid_volume` | float64 | F/M/S | Sum of volume removed from bid side in bar (`Remove` operations) |
| `limit_cancel_ask_volume` | float64 | F/M/S | Sum of volume removed from ask side in bar |
| `limit_total_bid_volume` | float64 | F/M/S | `limit_add_bid_volume + limit_cancel_bid_volume` |
| `limit_total_ask_volume` | float64 | F/M/S | `limit_add_ask_volume + limit_cancel_ask_volume` |
| `net_limit_bid_volume` | float64 | F/M/S | `limit_add_bid_volume - limit_cancel_bid_volume` |
| `net_limit_ask_volume` | float64 | F/M/S | `limit_add_ask_volume - limit_cancel_ask_volume` |

### 3.2. Ratios / imbalance

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `limit_cancel_to_add_bid` | float64 | F/M/S | `limit_cancel_bid_volume / (limit_add_bid_volume + ε)` |
| `limit_cancel_to_add_ask` | float64 | F/M/S | `limit_cancel_ask_volume / (limit_add_ask_volume + ε)` |
| `limit_of_imbalance` | float64 | F/M/S | `(net_limit_bid_volume - net_limit_ask_volume) / (net_limit_bid_volume + net_limit_ask_volume + ε)` |

### 3.3. OFI (Order Flow Imbalance) at best levels

One simple bar-wise approximation (Cont-like):

Maintain for each event where best bid/ask price or size changes:

-   For bid side:
    
    -   If `p_bid_1` increases or size increases: add positive contribution
        
    -   If `p_bid_1` decreases or size decreases: add negative contribution
        

Analogous for ask side; then sum for bar.

Store:

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `ofi_bid` | float64 | F/M/S | Sum of OFI contributions on bid side in bar |
| `ofi_ask` | float64 | F/M/S | Sum of OFI contributions on ask side in bar |
| `ofi_net` | float64 | F/M/S | `ofi_bid - ofi_ask` |

Exact OFI formula can be your chosen variant; keep it consistent.

---

## 4\. Trade features (MarketDataType = Last)

Within each bar, accumulate over all trade events.

### 4.1. Basic trade stats

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `trade_count` | int32 | F/M/S | Number of trades in bar |
| `trade_volume_sum` | float64 | F/M/S | Total traded volume in bar |
| `trade_volume_mean` | float64 | F/M/S | `trade_volume_sum / max(trade_count, 1)` |
| `trade_volume_max` | float64 | F/M/S | Max trade size in bar |
| `trade_price_open` | float64 | F/M/S | Price of first trade in bar (if any) |
| `trade_price_close` | float64 | F/M/S | Price of last trade in bar (if any) |
| `trade_price_high` | float64 | F/M/S | Max trade price in bar |
| `trade_price_low` | float64 | F/M/S | Min trade price in bar |
| `vwap` | float64 | F/M/S | `sum(trade_price_i * trade_volume_i) / (trade_volume_sum + ε)` |

### 4.2. Aggressor-based features

Infer trade direction using contemporaneous quotes:

-   If trade price ≥ best\_ask at that time → aggressive buy
    
-   If trade price ≤ best\_bid at that time → aggressive sell
    

Per bar:

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `buy_trade_count` | int32 | F/M/S | Number of trades classified as aggressive buys |
| `sell_trade_count` | int32 | F/M/S | Number of aggressive sell trades |
| `buy_trade_volume` | float64 | F/M/S | Sum of volumes of aggressive buys |
| `sell_trade_volume` | float64 | F/M/S | Sum of volumes of aggressive sells |
| `trade_imbalance_volume` | float64 | F/M/S | `(buy_trade_volume - sell_trade_volume)` |
| `trade_imbalance_ratio` | float64 | F/M/S | `(buy_trade_volume - sell_trade_volume) / (buy_trade_volume + sell_trade_volume + ε)` |

Distance to quotes:

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `avg_buy_dist_to_ask` | float64 | F/M/S | Mean over buys of `(ask_price_at_trade - trade_price)` (use 0 if no buys) |
| `avg_sell_dist_to_bid` | float64 | F/M/S | Mean over sells of `(trade_price - bid_price_at_trade)` (0 if no sells) |

### 4.3. Trade intensity

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `trade_intensity` | float64 | F/M/S | `trade_count / bar_duration_sec` |
| `inter_trade_time_mean` | float64 | F/M/S | Mean time between trades within bar (0 if <2 trades) |

---

## 5\. Volatility & returns (short horizon)

### 5.1. Within bar (already partly in 1.)

Add:

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `mid_range` | float64 | F/M/S | `mid_high - mid_low` |
| `realized_var_bar` | float64 | F/M/S | Sum of squared log returns of midprice at tick level in bar |

You can compute `realized_var_bar` on the fast series and aggregate up for mid/slow.

### 5.2. Rolling across bars

Define small sets of lookback sizes, e.g. N ∈ {5, 10, 20}. For each N (per resolution):

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `rv_N` | float64 | F/M/S | `sum_{i=0..N-1} mid_return_bar_{t-i}^2` |
| `vol_est_N` | float64 | F/M/S | `sqrt(rv_N)` |
| `mid_return_bar_lag1` | float64 | F/M/S | `mid_return_bar` of previous bar |

You can restrict to 1–2 N values to keep feature count manageable.

---

## 6\. Liquidity / impact proxies

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `kyle_lambda_like` | float64 | F/M/S | `abs(mid_return_bar) / (abs(trade_imbalance_volume) + ε)` |
| `amihud_like` | float64 | F/M/S | `abs(mid_return_bar) / (trade_volume_sum + ε)` |
| `large_trade_volume` | float64 | F/M/S | Sum of volumes of trades larger than a threshold (e.g. percentile or fixed size) |
| `large_trade_count` | int32 | F/M/S | Number of such large trades in bar |

You can compute threshold offline as long-run quantile and then hardcode.

---

## 7\. Time-of-day and regime features

### 7.1. Time-of-day

For each bar:

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `seconds_since_open` | int32 | F/M/S | Seconds from session open (e.g. official market open) |
| `seconds_until_close` | int32 | F/M/S | Seconds until session close |
| `tod_sin` | float64 | F/M/S | `sin(2π * seconds_since_open / session_length_seconds)` |
| `tod_cos` | float64 | F/M/S | `cos(2π * seconds_since_open / session_length_seconds)` |
| `day_of_week` | int32 | F/M/S | 0–4 (Mon–Fri) |

(You can one-hot encode `day_of_week` later in Python if needed.)

### 7.2. Regime z-scores

Choose rolling windows, e.g. 60 bars (per resolution). For each bar:

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `z_volume` | float64 | F/M/S | `(trade_volume_sum - mean_volume_roll) / (std_volume_roll + ε)` |
| `z_spread` | float64 | F/M/S | `(spread_abs - mean_spread_roll) / (std_spread_roll + ε)` |
| `z_volatility` | float64 | F/M/S | `(vol_est_N - mean_vol_roll) / (std_vol_roll + ε)` |

You maintain rolling means/std in Rust.

---

## 8\. Cross-resolution aggregates

Only needed on `mid` and `slow`. Use data from lower resolution bars that fall into the larger bar window.

### For mid bars (aggregating fast)

Per mid bar, over all included fast bars:

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `avg_fast_spread_abs` | float64 | M | Mean of `spread_abs` from fast bars in this mid bar |
| `max_fast_spread_abs` | float64 | M | Max of fast `spread_abs` in mid bar |
| `avg_fast_imbalance_best` | float64 | M | Mean of fast `imbalance_best` |
| `sum_fast_trade_volume` | float64 | M | Sum of fast `trade_volume_sum` within mid bar |
| `avg_fast_trade_intensity` | float64 | M | Mean of fast `trade_intensity` |
| `sum_fast_ofi_net` | float64 | M | Sum of fast `ofi_net` |
| `fast_realized_var_in_mid` | float64 | M | Sum of fast `realized_var_bar` within this mid bar |

### For slow bars (aggregating mid or fast)

For slow from mid:

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `avg_mid_spread_abs` | float64 | S | Mean of mid `spread_abs` in slow bar |
| `sum_mid_trade_volume` | float64 | S | Sum of mid `trade_volume_sum` |
| `sum_mid_ofi_net` | float64 | S | Sum of mid `ofi_net` |
| `mid_realized_var_in_slow` | float64 | S | Sum of mid `realized_var_bar` in slow bar |

You can add more aggregates similarly if needed.

---

## 9\. Meta / missingness flags

Useful to handle illiquid periods and missing data.

| Name | Type | Resolution | Definition |
| --- | --- | --- | --- |
| `no_trade_flag` | int8 | F/M/S | 1 if `trade_count == 0`, else 0 |
| `no_quote_flag` | int8 | F/M/S | 1 if book snapshot missing, else 0 |
| `low_liquidity_flag` | int8 | F/M/S | 1 if `cum_bid_size_L + cum_ask_size_L` below threshold, else 0 |

---

## Implementation notes

1.  Decide `L` (e.g. 5). Generate columns for per-level features:
    
    -   `bid_price_level_1` … `bid_price_level_L`, etc.
        
2.  For each event, update current order book state and accumulators for the active bar.
    
3.  When bar ends:
    
    -   Compute all per-bar aggregates and derived features.
        
    -   Store them, reset bar accumulators, carry forward last book snapshot for new bar as needed.
        
4.  For mid/slow bars, either:
    
    -   Compute directly from raw events, or
        
    -   Aggregate precomputed fast features as above (often simpler).
        