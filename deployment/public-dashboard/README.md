# Public dashboard

A bounded live demonstration with three synthetic datasets (credit: 800 rows, churn: 900, claims: 720), automatic HUGIML estimator selection, and eight governance views.

A fixed stratified split reserves 25% of the examples for pruning comparisons. Training and estimator selection use only the other 75%. Training uses 3-fold CV, L=1 or 2 (default 2), topK=20, 50, or 100 (default 20), main_effect LR source policy, one compute thread, and at most 1,000 rows / 32 columns. RPTE uses two trees with depth parameter 2 (tree-depth cap 4). Audit execution retains the evidence needed by governance views. Model artifacts exist only in the disposable compute process; the web process retains presentation evidence.

Automatic selection evaluates the bounded LR and RPTE candidates and chooses the largest mean three-fold CV ROC-AUC; ties select LR. These are selection scores, not an independent test estimate. Advanced allows a manual estimator choice.

Charts support browser interaction. Numeric table columns and chart axes use concise display formatting while retaining their values. Case Review supports row-index and previous/next navigation across the training cases. Case evidence is prepared during training; navigation does not refit the model. Other embedded governance controls are read-only. Uploads and arbitrary training settings are not exposed.

## Pruning

After training, open Pruning and select up to three raw inputs or five mined patterns. Raw-input exclusion rebuilds the full pipeline; pattern removal uses PatternEditor and refits the downstream estimator. The estimator family and training settings stay fixed. Each comparison starts from the original model, which remains available in all other governance views. Reset comparison clears only the comparison.

Before/after ROC-AUC, Brier score, accuracy, input count, pattern count, and changed predictions use the same reserved holdout. Repeated comparisons are exploratory: the holdout is no longer an independent final test once it guides edits. Pattern removal does not remove the corresponding raw inputs, and raw-input exclusion does not guarantee the absence of related proxy information.

Pruning reconstructs the original model in a disposable process and checks its fingerprint before editing. No model pickle or user-supplied model is loaded. The session owns its selection inventory and model revision; stale or foreign selections are rejected.

## Limits

One Gunicorn process with four request threads coordinates one active compute process and two waiting jobs. Eight sessions retain at most 4 MiB of presentation JSON each, with a 15-minute inactivity expiry. Training has a 120-second deadline, a 30-second per-session cooldown, and a 16 KiB HTTP request limit. Session cookies are signed, HttpOnly, Secure, and SameSite=Strict. POST requests require a matching Origin. Job failures return generic status messages.

The service uses a single web worker. Multiple web workers or replicas require a shared session store and job coordinator. Queue bounds limit resource use; they do not provide bot protection or guarantee availability under hostile traffic. Sessions and jobs are lost on restart, deployment, or free-service suspension.
