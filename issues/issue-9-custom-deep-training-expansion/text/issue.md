## What to build

Expand the custom training flow to the deep models so users can create custom MLP and CNN entries with the same async lifecycle, shared picker and leaderboard integration, and approved deep-model detail outputs.

## Acceptance criteria

- [ ] Custom training supports MLP and CNN with curated hyperparameters and the same async lifecycle as the first custom training slice.
- [ ] Completed deep-model runs create custom entries in the shared model picker and shared leaderboard whose details include epoch curves and architecture summary information.
- [ ] Deep-model extras use the same tabbed model-details pattern established for shipped deep models.
- [ ] Regression coverage proves deep custom training honors the shared contracts and version-one limits.

## Blocked by

- Blocked by #5
- Blocked by #7
