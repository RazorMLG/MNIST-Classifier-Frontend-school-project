## What to build

Add MNIST-compatible CSV intake so users can upload data, receive clear validation feedback, preview the default train-validation-test split, and override split ratios before training. In the current UI direction, this intake lives in training mode and replaces the prediction canvas area instead of appearing as a separate permanent panel.

## Acceptance criteria

- [ ] Users can upload a MNIST-compatible CSV file and receive clear errors for incompatible schema or label problems.
- [ ] The app shows default split ratios and allows the user to adjust them before starting training.
- [ ] Entering training mode replaces the left-column prediction canvas area with the CSV intake and split-preview flow.
- [ ] Automated verification covers CSV validation and split-preview behavior end to end.

## Blocked by

- Blocked by #1
