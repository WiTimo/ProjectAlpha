The idea:

-   Start with very small, robust groups.
    
-   Only add new groups when the previous ones are fully tested.
    
-   The more that can go wrong in a group, the smaller that group should be.
    

---

## Phase 0 – Labeling and data split

Before touching features:

1.  Implement label logic in Rust:
    
    -   Example: “hit +X ticks before −Y within K events from time t”.
        
2.  Check labels:
    
    -   Compute label distribution over time (ratio of 1/0).
        
    -   Make sure there is no obvious drift when you change parameters.
        
3.  Split data strictly by time:
    
    -   Train / validation / test periods.
        

If labels or splits are wrong, nothing else matters.

---

## Phase 1 – Core minimal feature set (Group A: very low risk)

Implement in Rust, export to Parquet, then test in Python.

### A.1 Implementation (Rust)

Features (per resolution):

-   Price/return:
    
    -   `mid_return_bar`
        
    -   `spread_ticks` (spread in ticks)
        
-   Top-of-book state:
    
    -   `imbalance_best`
        
-   Basic trade:
    
    -   `trade_volume_sum_rel` (volume / rolling mean volume)
        
    -   `trade_count_log = log1p(trade_count)`
        
-   Simple volatility:
    
    -   `rv_N_log` for a single N (e.g. N=10) with proper causal rolling.
        

Normalization in Rust:

-   Use mid, tick size, rolling means as discussed.
    
-   No advanced OFI, no depth levels yet.
    

### A.2 Testing (Python)

1.  Load data.
    
2.  Check:
    
    -   No NaNs or infinities.
        
    -   Reasonable ranges (e.g. `spread_ticks` mostly small integers; `mid_return_bar` centered near 0).
        
3.  Compute train-only mean/std and standardize.
    
4.  Train a small TCN using only this group.
    
5.  Evaluate:
    
    -   Label metrics (AUC, accuracy) and trading metrics on OOS segments.
        
    -   Check for obvious bugs (model predicting constant, exploding loss, etc.).
        

If this doesn’t work reasonably, fix it here before adding anything else.

---

## Phase 2 – Top-of-book + basic depth (Group B: low–medium risk)

Now extend the feature set slightly.

### B.1 Add features (Rust)

Add:

-   Top-of-book:
    
    -   `spread_change_ticks`
        
    -   `mid_range_rel`
        
-   Depth summary:
    
    -   `cum_bid_size_L_rel`, `cum_ask_size_L_rel`
        
    -   `imbalance_L`
        

Normalization:

-   Depths relative to rolling mean depth L, as already planned.
    

### B.2 Testing

Pipeline:

1.  Export new Parquet with Group A + Group B.
    
2.  In Python:
    
    -   Recompute train mean/std (for all features).
        
    -   Train the same TCN architecture.
        
    -   Compare performance to Group A only:
        
        -   Same train/validation/test periods.
            
        -   Same training schedule (epochs, learning rate, etc.).
            
3.  Decision:
    
    -   If Group B improves OOS results consistently → keep it.
        
    -   If change is noise / mixed / negative → check implementation or drop some B features.
        

If something looks off, you can temporarily add only half of Group B (e.g. depth without `imbalance_L`) to localize issues.

---

## Phase 3 – Deeper book structure (Group C: medium risk)

Here you introduce per-level info where bugs are easier.

### C.1 Add features (Rust)

Per level k (maybe L small, e.g. 3 at first):

-   `bid_offset_level_k_ticks`, `ask_offset_level_k_ticks`
    
-   `bid_size_level_k_rel`, `ask_size_level_k_rel`
    

Start with k=1..3 (not full L) to reduce complexity.

### C.2 Testing

1.  Export: Group A + B + C(1..3).
    
2.  Python:
    
    -   Standardize.
        
    -   Train same TCN, compare with A+B.
        
3.  If stable and helpful:
    
    -   Optionally extend to full L later; repeat tests.
        

If performance degrades or training becomes unstable:

-   Check distributions (per level).
    
-   Plot a few time series of offsets/sizes to ensure they make sense.
    
-   Consider leaving only 3 levels if full L adds little.
    

---

## Phase 4 – Order-flow: simple (Group D: medium–high risk)

Order-flow is powerful but easy to implement incorrectly.

### D.1 Add features (Rust)

Start with simple, aggregated flows:

-   `limit_add_bid_volume_rel`, `limit_add_ask_volume_rel`
    
-   `limit_cancel_bid_volume_rel`, `limit_cancel_ask_volume_rel`
    
-   `limit_of_imbalance` (properly normalized and sign-log if heavy-tailed)
    

Avoid advanced OFI variants for now.

### D.2 Testing

1.  Export: A + B + C + D.
    
2.  Python:
    
    -   Standardize.
        
    -   Train, compare to A + B + C.
        
3.  If strange behavior:
    
    -   Check that “add” and “cancel” volumes make sense (no crazy spikes or negative values).
        
    -   Check that your causal rolling means for these are correct.
        

If needed, break D into two sub-groups:

-   D1: add/cancel volumes only.
    
-   D2: derived ratios/imbalances.
    

---

## Phase 5 – Order-flow: OFI and aggressor trades (Group E: high risk)

These are powerful but more fragile, so add them slowly.

### E.1 Add OFI

Start with:

-   `ofi_bid`, `ofi_ask`, `ofi_net_log` (relative + sign-log)
    

Only OFI first, no extra new stuff.

Test like before (A+B+C+D+E(OFI only)).

### E.2 Then add aggressor trade details

If OFI is stable, add:

-   `buy_trade_volume_rel`, `sell_trade_volume_rel`
    
-   `trade_imbalance_ratio`
    
-   `avg_buy_dist_to_ask`, `avg_sell_dist_to_bid`
    

Again: export → standardize → train → compare.

If something breaks, remove half and re-test to localize.

---

## Phase 6 – Volatility & impact advanced (Group F: medium risk)

Now the more exotic quantities.

Add in small chunks:

1.  F1: Additional RV/vol windows
    
    -   Add a second `rv_N_log` for a different N.
        
    -   Test.
        
2.  F2: Impact proxies
    
    -   `kyle_lambda_log`, `amihud_log`.
        
    -   Test again.
        

Check that these features do not blow up in illiquid periods (use flags and ε).

---

## Phase 7 – Regime / time-of-day (Group G: low risk)

These are usually safe.

Add:

-   `tod_sin`, `tod_cos`
    
-   `z_volume`, `z_spread`, `z_volatility`
    

Export, standardize, train, compare.  
These often help with stability across sessions and regimes.

---

## Phase 8 – Cross-resolution aggregates (Group H: high risk)

These are more complex, especially with 3 resolutions.

Add in very small steps:

1.  H1: Only a few fast→mid aggregates, e.g.:
    
    -   `avg_fast_spread_abs`, `sum_fast_trade_volume`, `sum_fast_ofi_net`.
        
2.  Test.
    
3.  H2: Then a few mid→slow aggregates.
    
4.  Test again.
    

Here you must check very carefully:

-   Alignment between fast, mid, slow bars.
    
-   That each group uses only data inside the current bar (no future).
    

---

## General testing loop

For each feature group X:

1.  Implement in Rust, regenerate Parquet with previous groups + X.
    
2.  In Python:
    
    -   Recompute train-only mean/std.
        
    -   Train multiple runs (different seeds or walk-forward splits) with and without X.
        
3.  Compare:
    
    -   OOS performance across runs.
        
    -   If X consistently helps → keep it.
        
    -   If X is neutral/noisy, you may keep it but watch model complexity.
        
    -   If X hurts, inspect for bugs or drop it.
        

Optionally, maintain a small table:

| Group | Features added | Status | Effect on OOS | Notes |
| --- | --- | --- | --- | --- |
| A | core minimal | base | – |  |
| B | top + depth | kept | +small |  |
| C | levels 1–3 | kept | + |  |
| D | add/cancel | kept | + |  |
| E | OFI, aggressor | ? | test |  |
| ... |  |  |  |  |

This gives you a clear history of what you added and how it behaved.

---

Summary of the pipeline

-   Implement features in **groups ordered by complexity and risk**.
    
-   For risky groups (OFI, cross-resolution, advanced impact), add **fewer features at a time**.
    
-   After each group:
    
    -   Export → standardize → train → compare OOS.
        
    -   Keep or drop based on repeated, consistent results, not one lucky run.
        
