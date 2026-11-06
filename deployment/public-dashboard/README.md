# Public dashboard

A live demonstration using three synthetic sample datasets: credit risk (800 rows), churn (900), and claims (720). It provides model training, eight governance views, and pruning comparisons. Uploads are not supported.

Choose a pattern budget of 20, 50, or 100 and pattern order 1 or 2. Automatic estimator selection uses three-fold cross-validation on the training split; a manual choice is available under Advanced. Selection scores are not independent test estimates. A fixed stratified split reserves 25% of each dataset for comparisons.

Case Review includes up to 20 training cases sampled in proportion to target classes, with index and previous/next navigation. Other governance views support chart interaction but do not edit the model.

## Pruning

Exclude up to three inputs or remove up to five patterns. Input exclusion retrains the pipeline; pattern removal refits the downstream estimator. Both retain the selected estimator family and settings, and compare performance on the same reserved holdout.

Each comparison starts from the original model. Reset clears the comparison only. Repeated comparisons are exploratory, and removing an input does not guarantee that related information is absent from other inputs.

## Limits

The service runs one training job at a time, with two waiting slots and a two-minute training limit. Sessions expire after 15 minutes of inactivity and are lost on restart. Run one web worker and one service instance; additional replicas require shared session and queue management.
