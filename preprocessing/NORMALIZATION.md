I’ll describe a concrete normalization plan based on the schema above, and explicitly note where to do it (Rust vs Python) and how to keep it strictly causal.

Think in two layers:

1.  In Rust: make features scale-free and dimensionless using only *current/past* info (no future).
    
2.  In Python: standardize each feature using *training-set-only* statistics (global mean/std).
    

---

## 1\. General rules

1.  Do not feed raw absolute prices or raw large magnitudes directly. Always:
    
    -   Use returns or differences.
        
    -   Divide by mid price or tick size.
        
2.  For highly skewed positive variables (volumes, depths, RV):
    
    -   Use `log1p` or `sign(x) * log1p(|x|)` before standardization.
        
3.  For ratios already in a bounded range (e.g. \[-1, 1\] or \[0, 1\]), normalization is optional.
    
4.  For time-of-day sin/cos, they are already in \[-1, 1\]; just standardize lightly or leave as-is.
    
5.  For binary flags, do not normalize.
    

All “rolling” quantities and normalizers in Rust must be computed causally: at time `t`, use only data from bars `< t` to form the “typical” value.

---

## 2\. Price-related features

### 2.1. Mid, bid, ask

You should not feed:

-   `best_bid_price`, `best_ask_price`, `mid_price`, `log_mid_price` as raw magnitudes.
    

Instead:

1.  In Rust (structural normalization):
    
    -   Normalize spreads and level prices by mid or tick size:
        
        -   `spread_ticks = spread_abs / tick_size`
            
        -   `mid_range_rel = mid_range / mid_price`
            
        -   For each level k:
            
            -   `bid_offset_level_k_ticks = (mid_price - bid_price_level_k) / tick_size`
                
            -   `ask_offset_level_k_ticks = (ask_price_level_k - mid_price) / tick_size`
                
    -   Keep `mid_return_bar = ln(mid_close) - ln(mid_open)` as is (already scale-free).
        
    -   For deltas:
        
        -   `best_bid_change_ticks = (best_bid_price - best_bid_price_open) / tick_size`
            
        -   `best_ask_change_ticks = (best_ask_price - best_ask_price_open) / tick_size`
            
        -   `spread_change_ticks = (spread_close - spread_open) / tick_size`
            
2.  In Python (global standardization):
    
    For each of these *already relative/tick-based* features:
    
    -   Compute over training set only:
        
        -   `μ_feature`, `σ_feature`
            
    -   Standardize:
        
        -   `feature_norm = (feature - μ_feature) / σ_feature`
            
    
    This includes:
    
    -   `spread_ticks`, `spread_rel`, `mid_return_bar`, `mid_range_rel`, `best_bid_change_ticks`, `best_ask_change_ticks`, `spread_change_ticks`, all offsets/gaps in ticks.
        

---

## 3\. Depth / size features

### 3.1. Raw sizes and cumulatives

You should not use raw sizes directly without scaling; depths and sizes are heavy-tailed.

In Rust:

1.  For each size/depth-like variable:
    
    -   `best_bid_size`, `best_ask_size`
        
    -   `bid_size_level_k`, `ask_size_level_k`
        
    -   `cum_bid_size_3`, `cum_ask_size_3`, `cum_bid_size_L`, `cum_ask_size_L`
        
    -   `limit_add_*_volume`, `limit_cancel_*_volume`, etc.
        
    -   `trade_volume_sum`, `buy_trade_volume`, `sell_trade_volume`, etc.
        
    
    First compute a *causal rolling typical size* per instrument and resolution. Example:
    
    -   Maintain `rolling_mean_depth_L(t)` = EWMA or simple moving average of `cum_bid_size_L + cum_ask_size_L` up to bar `t-1`.
        
    -   Maintain `rolling_mean_trade_volume(t)` = EWMA/MA of `trade_volume_sum` up to bar `t-1`.
        
2.  Then define normalized versions in Rust:
    
    -   `cum_bid_size_L_rel = cum_bid_size_L / (rolling_mean_depth_L(t) + ε)`
        
    -   `cum_ask_size_L_rel = cum_ask_size_L / (rolling_mean_depth_L(t) + ε)`
        
    -   `best_bid_size_rel = best_bid_size / (rolling_mean_depth_L(t) + ε)`
        
    -   Level sizes:
        
        -   `bid_size_level_k_rel = bid_size_level_k / (rolling_mean_depth_L(t) + ε)`
            
        -   `ask_size_level_k_rel = ask_size_level_k / (rolling_mean_depth_L(t) + ε)`
            
    
    For volumes:
    
    -   `trade_volume_sum_rel = trade_volume_sum / (rolling_mean_trade_volume(t) + ε)`
        
    -   `buy_trade_volume_rel = buy_trade_volume / (rolling_mean_trade_volume(t) + ε)`
        
    -   `sell_trade_volume_rel = sell_trade_volume / (rolling_mean_trade_volume(t) + ε)`
        
    -   Likewise for `limit_add_*_volume`, `limit_cancel_*_volume`.
        
3.  Optionally also apply sign-log transform (especially for OFI-like and net flows):
    
    For any signed heavy-tailed variable `x` (e.g. `net_limit_bid_volume`, `ofi_net`):
    
    -   In Rust:
        
        -   `x_log = sign(x) * log1p(|x|)`
            

In Python:

-   Standardize only the *relative* or log-transformed versions, not the raw ones:
    
    -   `size_rel_norm = (size_rel - μ_size_rel_train) / σ_size_rel_train`
        
    -   `x_log_norm = (x_log - μ_x_log_train) / σ_x_log_train`
        

---

## 4\. Imbalances and ratios

These are already bounded in \[-1, 1\] or near that:

-   `imbalance_best`, `imbalance_3`, `imbalance_L`
    
-   `limit_cancel_to_add_bid`, `limit_cancel_to_add_ask`
    
-   `trade_imbalance_ratio`
    
-   Any additional normalized OFI, etc.
    

In Rust:

-   Just compute them as defined; no extra scaling needed.
    

In Python:

-   Either:
    
    -   Leave them as-is, or
        
    -   Light z-score:
        
        -   `imbalance_norm = (imbalance - μ_imbalance_train) / σ_imbalance_train`
            

This does not cause leakage (you compute μ,σ from training only).

---

## 5\. Order-flow / OFI features

These are usually signed and heavy-tailed:

-   `net_limit_bid_volume`, `net_limit_ask_volume`, `limit_of_imbalance`
    
-   `ofi_bid`, `ofi_ask`, `ofi_net`
    

In Rust:

1.  First normalize by typical depth/volume, to make them dimensionless:
    
    -   `ofi_net_rel = ofi_net / (rolling_mean_depth_L(t) + ε)` or
        
    -   `ofi_net_rel_vol = ofi_net / (rolling_mean_trade_volume(t) + ε)`
        
2.  Then apply sign-log transform:
    
    -   `ofi_net_log = sign(ofi_net_rel) * log1p(|ofi_net_rel|)`
        

Do the same pattern for `ofi_bid`, `ofi_ask`, `net_limit_*_volume`.

In Python:

-   Standardize the log-transformed versions only:
    
    -   `ofi_net_log_norm = (ofi_net_log - μ_ofi_net_log_train) / σ_ofi_net_log_train`
        

---

## 6\. Trade features

Already partly covered with volume normalization. Add:

-   `trade_count`
    
-   `buy_trade_count`, `sell_trade_count`
    
-   `trade_volume_mean`, `trade_volume_max`
    
-   `trade_intensity`
    
-   `inter_trade_time_mean`
    

In Rust:

1.  For count/intensity variables (non-negative, skewed):
    
    -   Option A: relative to typical count:
        
        -   Maintain `rolling_mean_trade_count(t)` up to `t-1`.
            
        -   `trade_count_rel = trade_count / (rolling_mean_trade_count(t) + ε)`
            
        -   `trade_intensity_rel = trade_intensity / (rolling_mean_trade_intensity(t) + ε)`
            
    -   Option B: log transform:
        
        -   `trade_count_log = log1p(trade_count)`
            
        -   `trade_intensity_log = log1p(trade_intensity)`
            
2.  For `trade_volume_mean`, `trade_volume_max`:
    
    -   Normalize by rolling mean trade volume, then `log1p` if needed:
        
        -   `trade_volume_mean_rel = trade_volume_mean / (rolling_mean_trade_volume(t) + ε)`
            
        -   `trade_volume_max_rel = trade_volume_max / (rolling_mean_trade_volume(t) + ε)`
            

In Python:

-   Standardize only the relative/log versions.
    

---

## 7\. Volatility & realized variance

-   `realized_var_bar`
    
-   `rv_N`
    
-   `vol_est_N`
    

These scale roughly like price² and can be large.

In Rust:

1.  Make them dimensionless:
    
    -   `realized_var_bar_rel = realized_var_bar / (mid_price^2 + ε)`
        
    -   `rv_N_rel = rv_N / (mid_price^2 + ε)` (use mid at bar t)
        
    -   Or use average mid of rolling window if you prefer.
        
2.  Apply log transform:
    
    -   `rv_N_log = log1p(rv_N_rel)`
        
    -   `vol_est_N_log = log1p(vol_est_N)` (vol is already sqrt, but log helps heavy tails)
        

In Python:

-   Standardize the log/relative versions:
    
    -   `rv_N_log_norm = (rv_N_log - μ_rv_N_log_train) / σ_rv_N_log_train`
        

---

## 8\. Liquidity / impact proxies

-   `kyle_lambda_like`
    
-   `amihud_like`
    
-   `large_trade_volume`, `large_trade_count`
    

These are also heavy-tailed; treat similarly.

In Rust:

1.  For `kyle_lambda_like` and `amihud_like`:
    
    -   Optional: log transform directly (they’re positive):
        
        -   `kyle_lambda_log = log1p(kyle_lambda_like)`
            
        -   `amihud_log = log1p(amihud_like)`
            
2.  For `large_trade_volume`:
    
    -   Normalize by `rolling_mean_trade_volume(t)` then log1p:
        
        -   `large_trade_volume_rel = large_trade_volume / (rolling_mean_trade_volume(t) + ε)`
            
        -   `large_trade_volume_rel_log = log1p(large_trade_volume_rel)`
            
3.  For `large_trade_count`:
    
    -   `large_trade_count_log = log1p(large_trade_count)`
        

In Python:

-   Standardize the log/relative variants only.
    

---

## 9\. Time-of-day and regime features

### Time-of-day:

-   `tod_sin`, `tod_cos` are already in \[-1, 1\].
    

In Python:

-   Either:
    
    -   Leave them as-is (no scaling), or
        
    -   Light standardization from training stats.
        

### Regime z-scores:

You already define:

-   `z_volume`, `z_spread`, `z_volatility` in Rust as rolling z-scores using causal rolling mean/std.
    

These are already normalized, usually mean ~0, std ~1 over time.

In Python:

-   You can use them directly (no further normalization required).
    
-   If you want strict consistency, you may standardize them again slightly, but it’s not necessary.
    

---

## 10\. Flags and categorical features

-   `no_trade_flag`, `no_quote_flag`, `low_liquidity_flag`, `day_of_week`
    

Do not normalize the binary flags.

For `day_of_week`:

-   In Python, either:
    
    -   Convert to one-hot and feed as is, or
        
    -   Leave as integer and let embedding layer handle it (if you use embeddings).
        

No scaling necessary.

---

## 11\. How to avoid future data in normalization

1.  Split by time into train/validation/test (e.g. chronological split).
    
2.  In Rust:
    
    -   All rolling means/stds used for:
        
        -   `rolling_mean_depth_L(t)`
            
        -   `rolling_mean_trade_volume(t)`
            
        -   `rolling_mean_trade_count(t)`
            
        -   etc.
            
    -   Must be updated sequentially:
        
        -   When you process bar `t`, compute features using rolling stats only from bars `< t`.
            
    -   This is naturally causal if you process in time order.
        
3.  In Python:
    
    -   After you export all features (already dimensionless/log-transformed) for all sets:
        
        -   Compute per-feature `μ_train`, `σ_train` using only the training portion.
            
        -   Apply `x_norm = (x - μ_train) / σ_train` to:
            
            -   training
                
            -   validation
                
            -   test
                
    
    This does not introduce lookahead; normalization parameters are fixed based on past (train) period only.
    

---

## 12\. Minimal “must-normalize” checklist

If you want a short checklist of things that absolutely should not be fed raw:

Normalize (per rules above):

1.  All prices, spreads, offsets → to ticks or relative to mid.
    
2.  All sizes/volumes/depths/counts → divide by causal rolling mean and/or log1p.
    
3.  OFI, net limit flows → divide by typical depth/volume, then sign-log.
    
4.  Realized variance/vol → divide by mid² and/or log1p.
    
5.  Impact proxies (Kyle, Amihud, large trade stats) → log1p.
    

Leave mostly as is (maybe light z-score):

1.  Ratios/imbalances in \[-1,1\].
    
2.  Time-of-day sin/cos.
    
3.  Causal z-scores (`z_volume`, etc.).
    
4.  Flags.
    