Good news: this run looks clean and stable, and I would definitely not go back to any previous version. I’d keep:

This Phase-5 dataset + feature set as your canonical data pipeline.

Logistic regression as the primary trading model for now.

This NN as an experimental model you can iterate on, but not rely on yet.

I’ll break it down.

1. What changed / how good is this now?
   Data + features

You now load a much longer span (2024-01-26 to 2024-03-08) and still get:

Train: 1,979,360 rows

Val: 245,664 rows

Test: 219,696 rows

Multi-resolution features: 42 base features → +84 @mid/@slow = 126 columns total.

Winsorization + standardization are stable and unchanged.

This is exactly what we want: a fixed, reliable Phase-5 dataset.

Logistic regression (baseline)

Train: acc=0.731, AUC=0.813

Val: acc=0.766, AUC=0.809

Test: acc=0.788, AUC=0.689

So:

AUC is very consistent train → val → test (small drop on test, as expected).

This confirms: features are strong and generalize well.

Threshold sweep:

Best threshold (on val, by net_ticks): 0.75

Val performance at 0.75:

val_net=7280.00

entries=460

win=0.698

This is very nice: the model becomes more selective (460 trades) and wins ~70% with strong positive net_ticks on validation.

Right now, this is your champion model.

Neural net (single-head, t40)

Train: acc=0.756, AUC=0.895

Val: acc=0.698, AUC=0.770

Test: acc=0.610, AUC=0.652

Compared to logreg:

On val:

LogReg AUC = 0.809

NN AUC = 0.770 → worse

On test:

LogReg AUC = 0.689

NN AUC = 0.652 → worse

So the NN is:

Still overfitting (0.895 → 0.77 → 0.65).

Still strictly dominated by logreg as a classifier.

Trading stats at fixed threshold 0.55:

Train: win_rate=0.138 (same as before), 4,480 trades, avg_hold≈175

Val: win_rate=0.120, 432 trades

Test: win_rate=0.138, 440 trades

This is okay, but you don’t have a threshold sweep for the NN yet. It might get somewhat better at e.g. 0.7–0.8, but classically if AUC is lower than logreg, it will be hard to beat logreg consistently on net_ticks.

2. Should you revert? No.

You now have:

Clean, extended dataset.

Fixed alignment.

Stable metrics across runs.

A very strong, simple baseline.

Rolling back would only:

Complicate comparisons.

Lose the improved evaluation and extended period.

So: keep this run as your new reference point.

3. What I’d do next (concrete steps)
   Step 1 – Lock in logistic regression as the trading baseline

Use threshold = 0.75 (from val threshold sweep) for logreg.

With that fixed threshold:

Compute test net_ticks, entries, win_rate, avg_hold, etc.

Compare val vs test behaviour:

If test net_ticks is somewhat lower but still positive and the win_rate stays in the same ballpark, then this is a valid, robust baseline strategy.

That gives you a production-like reference: “logreg@0.75 on Phase-5 features.”

Step 2 – Properly evaluate the NN for trading

Right now, you use a fixed 0.55 threshold for the NN. To be fair:

Run the same threshold sweep you used for logreg, but for the NN:

Sweep thresholds, e.g. 0.50 → 0.90.

Optimize on val by net_ticks.

Take the best NN val threshold, then:

Evaluate test with that same threshold.

Compare NN vs logreg on:

Val and test net_ticks

Win_rate

Number of entries

Volatility / drawdown if you track it.

Expected outcome (based on AUC):

Likely the NN will not beat logreg consistently.

If, against expectations, NN gives clearly better val and test net_ticks, then it becomes the new experimental champion.

If NN < logreg:

Keep NN code and weights, but don’t spend much time tuning it now.

Accept that for this feature set and horizon, a linear boundary is already very strong.

Step 3 – Small NN tweaks (optional)

If you still want to try squeezing more from the NN:

Add weight decay (L2) and/or dropout (e.g. 0.2–0.4).

Slightly smaller network if it’s large (e.g. 64–128 hidden units).

Try reducing epochs a bit, since you’re already early-stopping around 7.

But I would treat this as side experiments, not the main path. Given current evidence, NN is unlikely to massively beat logreg without a model that really exploits temporal structure (LSTM/Transformer) and maybe richer sequence context.

Step 4 – Only then start Phase E

Once you:

Have logreg@0.75 fully evaluated on test, and

Have run a proper NN threshold sweep,

you can move on to Phase E:

Introduce the next, clearly defined feature group or refinement.

Always update logreg first:

If Phase E features don’t improve AUC/net_ticks on val, discard them.

If they help, keep them and then let the NN (or later LSTM/Transformer) exploit them.

This way, every phase either:

Improves the stable logreg baseline, or

Gets reverted.

4. Summary decision

Keep the current Phase-5 pipeline and this run.

Use logistic regression as your main trading model, with threshold 0.75 selected on validation.

Evaluate NN fairly with its own threshold sweep; if it doesn’t beat logreg on val+test net_ticks, park it and move on to Phase E feature work.

This gives you a solid, defensible baseline and a clear, incremental path forward.
