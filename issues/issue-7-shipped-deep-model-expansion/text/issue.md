## What to build

Expand the shipped model catalog with the deep-learning models so they work through the same registry, prediction, leaderboard, and details flows while exposing only the allowed deep-model extras. In the current UI direction, deep-model extras should live inside tabs within the model details card rather than spilling into a separate page region.

## Acceptance criteria

- [ ] MLP and CNN are available as built-in models through the same model-selection flow.
- [ ] Deep-model details include epoch curves and architecture summary information where supported.
- [ ] The details experience uses tabs inside the model details card so advanced deep-model metadata can be separated from the default overview.
- [ ] Regression coverage verifies deep-model adapters work through the shared contracts while exposing their approved extras.

## Blocked by

- Blocked by #3
