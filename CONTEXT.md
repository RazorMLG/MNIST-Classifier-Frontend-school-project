# MNIST Classifier

This context covers the packaged and custom MNIST digit-classification workflows exposed by the project. It names the data, training, prediction, and catalog concepts shared by the backend and frontend.

## Language

### Model Inventory

**Shipped Model**:
A model artifact packaged with the project and available as soon as the app starts.
_Avoid_: built-in model, default model
For the shipped classical models, the artifact and public metadata are regenerated offline from the repository root `train.csv` and then committed with the project.

**Custom Model**:
A model artifact produced from a user-started Training Job against an uploaded Training Dataset.
_Avoid_: user model, runtime model

**Reference Prototype**:
The first shipped prototype classifier used as the baseline demo model and comparison point.
_Avoid_: sample model, starter model

**Model Catalog**:
The shared list of Shipped Models and Custom Models shown in the UI and available for prediction.
_Avoid_: model list, leaderboard data

**Model Registry**:
The project-local record that maps each model in the Model Catalog to its persisted artifact.
_Avoid_: inventory file, JSON list

### Data And Runs

**Training Dataset**:
A labeled MNIST-compatible CSV upload with one label column and 784 pixel columns.
_Avoid_: raw CSV, upload file

**Dataset Split**:
The train, validation, and test partition derived from a Training Dataset for preview and evaluation.
_Avoid_: ratio form, percentages

**Training Job**:
The one-at-a-time asynchronous run that validates a Training Dataset, builds a Custom Model, and reports progress.
_Avoid_: background task, worker job

**Canvas Digit**:
A 20x20 grayscale drawing captured in the browser before backend normalization and scoring.
_Avoid_: doodle, board state

**Prediction Run**:
A single scoring attempt that sends one Canvas Digit to one selected model and returns digit confidences.
_Avoid_: inference call, predict request

**Digit Prototype**:
A per-digit grayscale template used by the prototype classifier to compare Canvas Digits and dataset rows.
_Avoid_: weights, embedding

## Relationships

- A **Model Catalog** contains **Shipped Models** and **Custom Models**
- A **Model Registry** records every model in the **Model Catalog** and points to its artifact
- The shipped classical portion of the **Model Catalog** is regenerated from the labeled root `train.csv`; the unlabeled root `test.csv` is not part of shipped-model evaluation
- A **Training Job** consumes one **Training Dataset** and one **Dataset Split**
- A successful **Training Job** creates one **Custom Model**; an interrupted **Training Job** creates none
- A **Prediction Run** scores one **Canvas Digit** against one selected model from the **Model Catalog**
- The **Reference Prototype** is a **Shipped Model** that uses **Digit Prototypes**

## Example dialogue

> **Dev:** "If I upload a new **Training Dataset** and start a **Training Job**, does it replace the **Reference Prototype**?"
> **Domain expert:** "No. The **Training Job** creates a new **Custom Model**, and both it and the **Reference Prototype** stay in the **Model Catalog**."
>
> **Dev:** "What does the preview step tell me?"
> **Domain expert:** "It validates the **Training Dataset** and shows the **Dataset Split** before the **Training Job** starts."

## Flagged ambiguities

- "built-in" and "shipped" were both used for packaged models - resolved: use **Shipped Model** in conversation; keep `kind="built-in"` only as persisted metadata.
- "model list" was used for both the UI table and the registry file - resolved: use **Model Catalog** for the public list and **Model Registry** for the persisted record.
