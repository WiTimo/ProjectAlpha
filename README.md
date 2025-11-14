# Alpha Model

This repository contains the implementation of the Alpha Model, a machine learning model for Tick Data Trading on the NQ Market using data from Ninjatrader L2.

## Structure

- preprocessing: Scripts for data cleaning and feature engineering.
- training: Code for training the Alpha Model.
- evaluation: Tools for assessing model performance.
- deployment: Instructions for deploying the model in a live trading environment.

# Overview

Goal  
Predict short-horizon price direction in a market where “time to move” varies (sometimes seconds, sometimes many minutes), using a single multi-scale neural network with minimal Python-side work.

Core modeling idea

1.  Define labels in **event/price space**, not fixed clock time:  
    – Example: label = 1 if price hits `+X ticks` before `−Y ticks` within the next `K` events/bars; label = 0 otherwise. (Default 10 pips / 40 ticks in each direction)
    – This makes targets more stable across fast vs slow markets.
    
2.  Use a **single multi-scale model** (multi-branch TCN):  
    – Branch for high resolution (microstructure, fast patterns).  
    – Branch for medium resolution (intermediate patterns).  
    – Branch for low resolution (trend/regime context).  
    – Outputs from all branches are concatenated and passed to a shared dense + sigmoid head that predicts a single probability in (0,1).
    

Preprocessing approach (three resolutions precomputed in Rust)  
Do the heavy lifting in Rust, then keep Python thin:

1.  In Rust (one main pipeline + parameterized resampling):  
    a) Clean and standardize raw tick data into a canonical format (trades/quotes, midprice, spread, etc.).  
    b) Build multiple bar streams at different resolutions, e.g.:  
    – `fast` : 1-second (or small event-based) bars  
    – `mid` : 10-second bars  
    – `slow` : 60-second bars  
    c) For each resolution, compute features:  
    – returns, rolling volatility, volume, spread, imbalance, etc.  
    d) Define labels once using your chosen event-based rule (“hit +X before −Y within K events from time t”).  
    – Ensure each bar stream gets aligned labels that refer to the same decision time t.  
    e) Save as three Parquet files (Pattern A):  
    – `data_fast.parquet`  
    – `data_mid.parquet`  
    – `data_slow.parquet`  
    Each with: `timestamp/index`, features for that resolution, and `label`.
    
2.  In Python (minimal work):  
    a) Load the three Parquet files.  
    b) Align rows across resolutions by timestamp/index so each training sample has:  
    – a window from the fast series,  
    – a window from the mid series,  
    – a window from the slow series,  
    – and a single label.  
    c) Convert these into tensors:  
    – `X_fast : (batch, L_fast, F_fast)`  
    – `X_mid  : (batch, L_mid, F_mid)`  
    – `X_slow : (batch, L_slow, F_slow)`  
    – `y      : (batch,)`  
    d) Feed them into the multi-branch CNN/TCN model and train.
    

Result  
You end up with:

-   A single multi-scale neural network that outputs one probability (up vs down) and is robust to fast and slow markets.
    
-   Most complexity (cleaning, feature construction, multi-resolution resampling, labeling) lives in Rust.
    
-   Python is mainly a thin layer: load Parquet → build windows → train/evaluate the model.