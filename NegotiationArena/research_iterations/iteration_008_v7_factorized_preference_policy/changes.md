# Iteration 008 V7 change record

## Code added before online results

- Added `FactorizedPreferencePolicyBeliefPlannerAgent` as `framework_v7`.
- Joint latent state:
  - existing interpretable preference particles;
  - 27 response-policy particles from three rationality scales, three
    acceptance biases and three counter propensities.
- Responses to locked offers update the joint posterior.
- Opponent-initiated offers weakly update preference only.
- Bounded semantic claims project onto preference while preserving the
  conditional response-policy distribution.
- Low predictive probability triggers a 30% policy-only reset before applying
  response likelihood; preference marginal is preserved at reset time.
- Proposal ranking uses posterior mean action value minus an uncertainty-scaled
  upper-tail particle-regret penalty.
- Registered continuous, frozen, wrong/shuffled/oracle preference and
  uniform/shuffled response-policy variants.
- Added policy intervention action-flip diagnostics.

## Stage-0 correction

The first synthetic point-mass test exposed that raw CVaR regret has different
natural scales in price and resource games and could make every proposal
negative under a diffuse prior. Before any online V7 result, the preregistered
regret penalty was implemented in dimensionless form:

```text
normalized_regret = CVaR_regret / (CVaR_regret + abs(mean_action_value))
penalty = uncertainty_weight × normalized_regret × max(0, mean_action_value)
```

This keeps the zero outside option meaningful and bounds robustness spending by
the action's own positive posterior value. It is shared by all settings.

## Verification

`python -m unittest tests.test_repeated_integration_unittest -v`

- 47 tests passed;
- joint preference/policy update directions covered;
- ACCEPT/COUNTER/REJECT and surprise-reset branches covered;
- point-mass joint posterior selects a zero-regret exploit;
- policy interventions preserve preference marginal;
- no V7 online result existed at verification time.
