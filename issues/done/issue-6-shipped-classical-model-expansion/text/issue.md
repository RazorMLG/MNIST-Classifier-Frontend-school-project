## What to build

Expand the shipped model catalog with the remaining classical models so they work through the same registry, prediction, leaderboard, and details flows as the reference slice.

## Acceptance criteria

- [ ] k-NN, SVM, and Random Forest are available as built-in models through the same model-selection flow.
- [ ] Their metrics and details appear in the existing leaderboard and details UI without special-case screens.
- [ ] Regression coverage verifies classical-model adapters honor the shared contracts.

## Blocked by

- Blocked by #3