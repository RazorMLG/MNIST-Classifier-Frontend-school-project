## What to build

Expand the custom training flow to the remaining classical models so users can create custom k-NN, SVM, and Random Forest entries with the same lifecycle and shared picker and leaderboard integration as the first custom training slice.

## Acceptance criteria

- [ ] Custom training supports k-NN, SVM, and Random Forest with curated hyperparameters and the same async lifecycle as the first custom training slice.
- [ ] Each completed run creates a separate custom entry in the shared model picker and shared leaderboard, with persisted metadata and artifact storage.
- [ ] Regression coverage proves classical custom training reuses the shared training, registry, and evaluation contracts.

## Blocked by

- Blocked by #5
- Blocked by #6
