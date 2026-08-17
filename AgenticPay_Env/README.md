# AgenticPay environment adapter

This package wraps AgenticPay while keeping the seller fixed and replacing only the
focal buyer. It supports the original single-28 tasks, broader task families, and
multi-buyer/multi-seller settings. The native optimization target is AgenticPay
BuyerScore; deal rate, timeout rate, belief calibration, and action flips are diagnostic
metrics rather than substitutes for BuyerScore.

The upstream repository is not vendored. Run `bash scripts/bootstrap_upstreams.sh`, then
`python -m AgenticPay_Env.eval --help`. The universal buyer adapter is
`buyer/universal_framework.py`; variant definitions and ablations are in
`buyer/variants.py`.
