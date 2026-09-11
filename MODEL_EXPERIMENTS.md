# Ranking and final-ticket audit

Full training compares ranking weights 0 and 0.1 with identical seeds, epochs,
features and chronological splits. The requested ranking model remains active;
test scores never choose parameters. Each variant audits the last 50 validation
and last 50 test draws. Validation is exploratory because calibration uses that
partition. Test forecasts use no gradients or calibration from the test period.
Earlier observed test draws may supply features for later test draws.

Four ticket selectors are compared: production, without rotation, without pair
score, and probabilities only (without pair or spread score). Each block starts
without prior tickets and subsequently rotates against its own preceding tickets.
The main metric is the fraction of draws where either final ticket hits >=3.
Mean hits and Brier are also recorded. The random comparator simulates 100 pairs
per draw with the same overlap between tickets; it does not model temporal rotation.
Fifty draws are too few to establish reliable improvement for a rare event.

Daily updates reserve the preceding 32 examples from the current replay batch
once 64 earlier examples exist. An update is reverted if combined feedback loss
on these examples worsens by more than 1e-6. These examples may have been used in
earlier training: this is a regression guard, not an independent validation set.
Processed draws advance even when rejected, so the same update is not retried.
Offline comparison occurs before production replay of validation/test history.

The exact ticket search is vectorized; it still searches every combination and
preserves overlap constraints. Comparisons replace the extra walk-forward fits
in the scheduled full-training entrypoint. Actual runtime depends on hardware;
no extra paid runner or schedule is configured.

Output: .automation/model_comparison.json. Existing pending predictions are not
overwritten by scheduled training. Manually requested revisions are stored
separately so original forecasts and their evaluations remain auditable.
