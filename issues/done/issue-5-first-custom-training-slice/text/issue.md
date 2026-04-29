## What to build

Deliver the first complete custom training flow for one reference classifier, including curated hyperparameters, unique naming, async progress, persisted metadata, and integration into the shared model picker and shared leaderboard. In the current UI direction, training uses the left column, while the leaderboard remains visible below the active training area.

## Acceptance criteria

- [ ] A user can train one reference classifier from uploaded data with curated hyperparameters, a unique model name, a saved config snapshot, and a saved seed.
- [ ] Training runs asynchronously with visible progress, obeys the one-job-at-a-time rule, keeps the leaderboard visible below the training area, and creates a new custom leaderboard entry when complete.
- [ ] Completed custom models appear in the same picker and leaderboard used for shipped models, using labeling or filtering rather than a separate screen.
- [ ] Custom models created by this flow can be deleted, and automated verification covers invalid input handling and interruption behavior.

## Blocked by

- Blocked by #2
- Blocked by #3
- Blocked by #4
