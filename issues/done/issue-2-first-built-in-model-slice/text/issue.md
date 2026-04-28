## What to build

Integrate one shipped reference model end to end so a user can draw a digit, send raw canvas input to the backend, run canonical preprocessing, and receive a prediction with confidence data.

## Acceptance criteria

- [ ] One built-in reference model ships with the project and can be selected in the app.
- [ ] A drawn digit can be submitted through the UI and returns the predicted class plus confidence values for digits 0 through 9.
- [ ] Automated verification covers the preprocess-and-predict contract for the reference model.

## Blocked by

- Blocked by #1