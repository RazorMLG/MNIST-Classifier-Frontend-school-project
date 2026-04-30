# Docs

This directory holds architecture notes that support the MNIST project.

- Use `../CONTEXT.md` for shared project language and relationships.
- Use `./adr/` for decisions that are hard to reverse, surprising without context, and the result of a real trade-off.

## Version-One Guidance

- Use the root `start.ps1` script as the only visible startup command on Windows. It bootstraps the Python environment and frontend dependencies before launching the app, so version one does not add a separate user-facing setup step.
- Custom training requires a manual model name. The interface does not auto-suggest or auto-generate defaults.
- Failed training runs only show inline feedback for the current run and do not create failed entries in the shared model catalog.
