# Code changes

- Added `EndogeneityCalibratedInformationPlannerAgent` (`framework_v5`).
- Preserved the V4 planner, candidates, information bonus, commitment, and action lock.
- Applied log-linear likelihood tempering after each structured update.
- Terminal accept/reject responses to a locked focal offer receive strength 0.90.
- Non-terminal counters to a focal offer receive strength 0.55.
- Opponent-initiated offers receive strength 0.25.
- Exact repeated evidence is discounted by `1/sqrt(repeat_count)`.
- Added continuous/frozen CLI modes, independent V5 traces, and unit tests showing
  higher entropy for weak evidence and diminishing repeated-evidence strength.
