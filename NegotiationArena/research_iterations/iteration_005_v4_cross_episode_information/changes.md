# Code changes

- Added `CrossEpisodeInformationPlannerAgent` (`framework_v4`).
- Reused V3.3 posterior, safeguards, action-space-adaptive candidate frontier,
  reciprocal commitment, and locked language realization unchanged.
- When the current action is `ACCEPT`, V4 evaluates the response-supported probe.
- Future information value is
  `DVOI * sqrt(remaining_episodes) * normalized_posterior_entropy`.
- The planner probes only if current contingent value plus future information value
  exceeds certain acceptance plus a progress-dependent margin.
- The bonus is exactly zero in the final episode and vanishes as belief concentrates.
- Added continuous/frozen CLI modes and tests for early activation and final-episode
  deactivation.
