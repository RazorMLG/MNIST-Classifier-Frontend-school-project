import { useEffect, useState, type ChangeEvent } from "react";

import "./App.css";

const DRAWING_GRID_SIZE = 20;
const DRAWING_PIXEL_COUNT = DRAWING_GRID_SIZE * DRAWING_GRID_SIZE;

const SORTABLE_METRICS = ["accuracy", "macro_f1", "avg_inference_ms"] as const;

type SortableMetricKey = (typeof SORTABLE_METRICS)[number];

type HealthPayload = {
  status: string;
  service: string;
  storage: {
    ready: boolean;
    root: string;
    directories: string[];
  };
};

type ModelMetrics = {
  accuracy?: number;
  macro_precision?: number;
  macro_recall?: number;
  macro_f1?: number;
  avg_inference_ms?: number;
};

type ModelDataset = {
  source?: string;
  image_shape?: string;
  train_examples?: number;
  validation_examples?: number;
  test_examples?: number;
};

type SamplePrediction = {
  label: number;
  predicted: number;
  confidence: number;
};

type ModelEvaluation = {
  confusion_matrix?: number[][];
  sample_predictions?: SamplePrediction[];
};

type ModelTrainingSnapshot = {
  classifier?: string;
  file_name?: string;
  split?: {
    train_ratio: number;
    validation_ratio: number;
    test_ratio: number;
  };
  hyperparameters?: Record<string, number | string>;
};

type ModelTraining = {
  seed?: number;
  config_snapshot?: ModelTrainingSnapshot;
};

type DeepEpochCurve = {
  epoch: number;
  train_loss?: number;
  validation_loss?: number;
  train_accuracy?: number;
  validation_accuracy?: number;
};

type ModelDeepDetails = {
  architecture_summary?: string[];
  epoch_curves?: DeepEpochCurve[];
};

type ModelSummary = {
  id: string;
  name: string;
  kind: string;
  family?: string;
  description?: string;
  trained_at?: string;
  metrics?: ModelMetrics;
  dataset?: ModelDataset;
  hyperparameters?: Record<string, number | string>;
  evaluation?: ModelEvaluation;
  training?: ModelTraining;
  deep_details?: ModelDeepDetails;
  input?: {
    width: number;
    height: number;
    encoding?: string;
  };
};

type ModelsPayload = {
  models: ModelSummary[];
};

type PredictionPayload = {
  model: {
    id: string;
    name: string;
  };
  prediction: {
    digit: number;
    confidences: number[];
  };
};

type BootstrapState =
  | { kind: "loading" }
  | { kind: "ready"; health: HealthPayload; models: ModelSummary[] }
  | { kind: "error"; message: string };

type PredictionState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ready"; payload: PredictionPayload }
  | { kind: "error"; message: string };

type WorkspaceMode = "predict" | "training";

type ModelDetailsTab = "overview" | "deep";

type TrainingSplitForm = {
  train: number;
  validation: number;
  test: number;
};

type TrainingPreviewPayload = {
  file_name: string;
  dataset: {
    example_count: number;
    feature_count: number;
    label_range: {
      min: number;
      max: number;
    };
  };
  split: {
    ratios: {
      train: number;
      validation: number;
      test: number;
    };
    counts: {
      train: number;
      validation: number;
      test: number;
    };
  };
};

type TrainingPreviewState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ready"; payload: TrainingPreviewPayload }
  | { kind: "error"; message: string };

type TrainingJob = {
  id: string;
  model_id: string;
  model_name: string;
  status: string;
  progress: {
    percent: number;
    stage: string;
  };
  error?: string | null;
  started_at?: string;
  completed_at?: string | null;
};

type TrainingJobPayload = {
  job: TrainingJob;
};

type TrainingModelFamily =
  | "prototype"
  | "knn"
  | "svm"
  | "random-forest"
  | "mlp"
  | "cnn";

type CustomTrainingForm = {
  modelFamily: TrainingModelFamily;
  modelName: string;
  seed: number;
  maxExamplesPerLabel: number;
  prototypeBlend: number;
  temperature: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
  neighbors: number;
  pcaComponents: number;
  regularization: number;
  maxIter: number;
  estimators: number;
  maxDepth: number;
};

type TrainingJobState =
  | { kind: "idle" }
  | { kind: "submitting" }
  | { kind: "running"; job: TrainingJob }
  | { kind: "completed"; job: TrainingJob }
  | { kind: "error"; message: string };

type Language = "en" | "sr";

const DEFAULT_TRAINING_SPLIT: TrainingSplitForm = {
  train: 80,
  validation: 10,
  test: 10,
};

const DEFAULT_CUSTOM_TRAINING_FORM: CustomTrainingForm = {
  modelFamily: "prototype",
  modelName: "",
  seed: 17,
  maxExamplesPerLabel: 4,
  prototypeBlend: 0.35,
  temperature: 18,
  epochs: 4,
  batchSize: 32,
  learningRate: 0.001,
  neighbors: 5,
  pcaComponents: 16,
  regularization: 1,
  maxIter: 5000,
  estimators: 96,
  maxDepth: 16,
};

const TRAINING_POLL_INTERVAL_MS = 150;
const BOOTSTRAP_RETRY_ATTEMPTS = 12;
const BOOTSTRAP_RETRY_DELAY_MS = 250;

const APP_COPY = {
  en: {
    languageButtons: {
      sr: "Srpski",
      en: "English",
    },
    hero: {
      eyebrow: "MNIST model catalog",
      title: "Inspect leaderboard metadata. Score a digit live.",
      summary:
        "This tracer bullet keeps the app bootable while exposing model metrics, dataset lineage, and detail metadata in the UI, then reuses the selected model for the existing prediction flow.",
      statusLoading: "Checking backend",
      statusReady: "Backend ready",
      statusError: "Backend unavailable",
      noteLoading: "Checking backend status and loading model metadata.",
      noteError: (message: string) =>
        `The frontend could not reach the backend: ${message}`,
      noteReady: (count: number) =>
        `${count} model${count === 1 ? "" : "s"} loaded for inspection and live scoring.`,
    },
    workspace: {
      kicker: "Workspace mode",
      predictTitle: "Paint a digit sample",
      trainingTitle: "Upload a labeled MNIST CSV",
      predictModeButton: "Prediction mode",
      trainingModeButton: "Training mode",
    },
    board: {
      clear: "Clear board",
      scoring: "Scoring digit",
      runPrediction: "Run prediction",
      predictHelp:
        "Click or drag across the board. The frontend sends the raw pixel grid and the backend performs centering and scale normalization.",
      trainingIntro:
        "Upload a labeled CSV that matches the bundled MNIST training schema, preview how the split will divide the dataset, then launch a curated prototype, k-NN, SVM, Random Forest, MLP, or CNN run.",
    },
    training: {
      csvLabel: "Training CSV",
      csvSelected: (fileName: string) => `${fileName} selected for preview.`,
      csvPlaceholder:
        "Use the labeled training format: label followed by pixel0 through pixel783.",
      splitTrain: "Train split",
      splitValidation: "Validation split",
      splitTest: "Test split",
      modelLabel: "Training model",
      modelOptionPrototype: "Reference prototype",
      modelOptionKnn: "k-NN classifier",
      modelOptionSvm: "SVM classifier",
      modelOptionRandomForest: "Random Forest classifier",
      modelOptionMlp: "MLP classifier",
      modelOptionCnn: "CNN classifier",
      modelHelper:
        "Completed runs reuse the same shared leaderboard and model picker regardless of family.",
      nameLabel: "Model name",
      nameHelper:
        "Version one requires a manual model name. The app does not suggest defaults, and each run lands in the shared leaderboard as its own entry.",
      seedLabel: "Seed",
      seedHelper:
        "Saved with the model for reproducible split and evaluation behavior.",
      examplesCapLabel: "Examples per label cap",
      examplesCapHelper:
        "Recommended 4. Ceiling 5000 for the portable prototype baseline.",
      prototypeBlendLabel: "Prototype blend",
      prototypeBlendHelper:
        "Recommended 0.35. Higher blend leans more on the shipped reference prototype.",
      confidenceTemperatureLabel: "Confidence temperature",
      confidenceTemperatureHelper:
        "Recommended 18. Ceiling 40 to keep confidence scaling readable.",
      neighborsLabel: "Neighbors",
      neighborsHelper:
        "Recommended 5. Lower values stay sharper on lightly sized classroom splits.",
      pcaComponentsLabel: "PCA components",
      pcaComponentsHelperKnn:
        "Recommended 16. Keeps the feature projection lightweight for fast iteration.",
      regularizationLabel: "Regularization",
      regularizationHelper:
        "Recommended 1.0. Higher values fit more aggressively to the uploaded split.",
      maxIterationsLabel: "Max iterations",
      maxIterationsHelper:
        "Recommended 5000. Longer runs can help convergence on trickier uploaded splits.",
      pcaComponentsHelperSvm:
        "Recommended 16. Keeps the linear margin stable on small classroom datasets.",
      estimatorsLabel: "Estimators",
      estimatorsHelper:
        "Recommended 96. More trees improve stability but extend training time.",
      maxDepthLabel: "Max depth",
      maxDepthHelper:
        "Recommended 16. Shallower trees are easier to keep general on smaller uploads.",
      pcaComponentsHelperRandomForest:
        "Recommended 16. Keeps the projection compact before the forest stage.",
      epochsLabel: "Epochs",
      epochsHelper:
        "Recommended 4. Keep deep custom runs short enough to stay responsive on CPU-only classroom hardware.",
      batchSizeLabel: "Batch size",
      batchSizeHelper:
        "Recommended 32. Smaller batches keep version-one deep training practical on portable hardware.",
      learningRateLabel: "Learning rate",
      learningRateHelper: (modelFamily: TrainingModelFamily) =>
        modelFamily === "mlp"
          ? "Recommended 0.002 for the fixed MLP stack."
          : "Recommended 0.001 for the fixed CNN stack.",
      previewingButton: "Previewing split",
      previewButton: "Preview split",
      startingButton: "Starting training",
      runningButton: "Training in progress",
      startButton: "Start training",
      splitTotalWarning: "Split ratios must total 100 before preview.",
      idleHelp:
        "Choose a labeled CSV and preview the train, validation, and test counts before launching a custom training run.",
      previewError: (message: string) => `CSV preview failed: ${message}`,
      previewLoading: "Validating the CSV schema and calculating split sizes.",
      previewFile: "File",
      previewExamples: "Examples",
      previewFeatures: "Features",
      previewLabels: "Labels",
      previewRangeTo: "to",
      previewTrainExamples: (count: number) => `Train ${count} examples`,
      previewValidationExamples: (count: number) =>
        `Validation ${count} examples`,
      previewTestExamples: (count: number) => `Test ${count} examples`,
      previewReadyHelp:
        "Preview is ready. Start training to create a custom model entry without leaving the shared leaderboard screen.",
      submittingJob: "Submitting the custom training job.",
      completedJob: (modelName: string) =>
        `${modelName} is now available in the shared model list.`,
      failedJob: (message: string) => `Custom training failed: ${message}`,
      deleteButton: "Delete custom model",
      deletingButton: "Deleting custom model",
      validationTrainPositive: "Training split must be greater than zero.",
      validationPreviewFirst:
        "Preview the dataset split before starting training.",
      validationNameRequired:
        "Model name is required before starting training.",
      validationDuplicateName: (modelName: string) =>
        `Model name '${modelName}' is already in use. Choose a unique name.`,
      fallbackPreviewRequest: "CSV preview request failed.",
      fallbackTrainingRequest: "Training job request failed.",
      fallbackTrainingStatus: "Training status request failed.",
      fallbackFailedJob: (modelName: string) =>
        `${modelName} did not complete successfully.`,
      deleteSuccess: (modelName: string) =>
        `Custom model deleted: ${modelName}.`,
      deleteError: (message: string) => `Delete failed: ${message}`,
      fallbackDelete: "Custom model deletion failed.",
      prototypeWarningLargeCap:
        "Large per-label caps can slow the prototype baseline on classroom hardware.",
      prototypeWarningBlend:
        "High prototype blend leans heavily on the shipped baseline and can hide dataset differences.",
      prototypeWarningTemperature:
        "Very high confidence temperature can make score gaps look sharper than the underlying distances.",
      deepWarningEpochs:
        "Longer deep runs can noticeably increase training time on classroom hardware.",
      deepWarningBatchSize:
        "Very large batch sizes can make small uploaded splits less stable during version-one deep training.",
      deepWarningLearningRate:
        "Aggressive learning rates can make custom deep runs unstable or noisy.",
      pcaWarning:
        "Larger PCA projections need more examples and can slow down classroom-size custom runs.",
      knnWarningNeighbors:
        "Higher neighbor counts smooth predictions more aggressively and can blur similar digits.",
      svmWarningRegularization:
        "Higher SVM regularization values can fit tightly to a small uploaded split.",
      svmWarningMaxIter:
        "Longer SVM iteration budgets can noticeably increase version-one training time.",
      forestWarningEstimators:
        "Larger forests improve stability but will push training time up on classroom hardware.",
      forestWarningMaxDepth:
        "Very deep trees can overfit small uploaded datasets.",
    },
    leaderboard: {
      kicker: "Model leaderboard",
      heading: "Compare available entries",
      sortLabel: "Sort leaderboard by",
      accuracy: "Accuracy",
      macroF1: "Macro F1",
      inferenceLatency: "Inference latency",
      modelColumn: "Model",
      latencyColumn: "Latency",
      loading:
        "The model leaderboard will populate after the backend finishes bootstrapping.",
      unavailable:
        "The model leaderboard is unavailable until the backend responds.",
    },
    modelPicker: {
      label: "Model",
      loading: "Loading models",
      unavailable: "Backend unavailable",
      empty: "No models available",
    },
    details: {
      kicker: "Model details",
      emptyTitle: "Select a model",
      builtIn: "Built-in",
      custom: "Custom",
      datasetSource: "Dataset source",
      trainedAt: "Trained at",
      datasetSizes: "Dataset sizes",
      inputShape: "Input shape",
      overviewTab: "Overview",
      deepTab: "Deep details",
      deepKicker: "Deep details",
      architectureSummary: "Architecture summary",
      layer: (index: number) => `Layer ${index + 1}`,
      epochCurvesKicker: "Epoch curves",
      trainingProgression: "Training progression",
      epoch: "Epoch",
      trainLoss: "Train loss",
      validationLoss: "Validation loss",
      trainAccuracy: "Train accuracy",
      validationAccuracy: "Validation accuracy",
      trainingSnapshotKicker: "Training snapshot",
      savedConfigAndSeed: "Saved config and seed",
      seedLabel: "Seed",
      classifierLabel: "Classifier",
      uploadedFile: "Uploaded file",
      splitLabel: "Split",
      metricsKicker: "Metrics",
      leaderboardMetrics: "Leaderboard metrics",
      macroPrecision: "Macro precision",
      macroRecall: "Macro recall",
      hyperparametersKicker: "Hyperparameters",
      referenceConfiguration: "Reference configuration",
      confusionMatrixKicker: "Confusion matrix",
      evaluationBreakdown: "Evaluation breakdown",
      actualPred: "Actual\\Pred",
      samplePredictionsKicker: "Sample predictions",
      heldOutExamples: "Held-out examples",
      samplePrediction: (label: number, predicted: number) =>
        `Actual ${label} predicted ${predicted}`,
      emptyHelp:
        "Select a model from the leaderboard to inspect its metadata and evaluation details.",
    },
    prediction: {
      kicker: "Prediction",
      title: "Confidence profile",
      idleHelp:
        "Paint a digit and run the selected model to see the predicted class and the full digit confidence spread.",
      error: (message: string) => `Prediction failed: ${message}`,
      loading:
        "Normalizing the sketch and scoring it against the selected model.",
      headline: (modelName: string, digit: number) =>
        `${modelName} predicts ${digit}`,
    },
    common: {
      pending: "Pending",
      trainShort: "Train",
      validationShort: "Val",
      testShort: "Test",
    },
  },
  sr: {
    languageButtons: {
      sr: "Srpski",
      en: "English",
    },
    hero: {
      eyebrow: "Katalog MNIST modela",
      title: "Pregledaj metrike modela. Testiraj cifru uzivo.",
      summary:
        "Ovaj tracer bullet odrzava aplikaciju pokrenutom dok u interfejs uvodi metrike modela, poreklo skupa podataka i detaljne metapodatke, a zatim koristi izabrani model u postojecem toku predvidjanja.",
      statusLoading: "Provera backenda",
      statusReady: "Backend spreman",
      statusError: "Backend nedostupan",
      noteLoading: "Proveravam status backenda i ucitavam metapodatke modela.",
      noteError: (message: string) =>
        `Frontend nije mogao da pristupi backendu: ${message}`,
      noteReady: (count: number) =>
        `${count} model${count === 1 ? "" : "a"} ucitan${count === 1 ? "" : "o"} za pregled i testiranje uzivo.`,
    },
    workspace: {
      kicker: "Rezim rada",
      predictTitle: "Nacrtaj uzorak cifre",
      trainingTitle: "Otpremi oznaceni MNIST CSV",
      predictModeButton: "Predvidjanje",
      trainingModeButton: "Obuka",
    },
    board: {
      clear: "Obrisi tablu",
      scoring: "Obradjujem cifru",
      runPrediction: "Pokreni predvidjanje",
      predictHelp:
        "Klikni ili prevuci preko table. Frontend salje sirovu mrezu piksela, a backend radi centriranje i normalizaciju razmere.",
      trainingIntro:
        "Otpremi oznaceni CSV koji odgovara ukljucenoj MNIST semi za obuku, pregledaj kako ce podela rasporediti skup podataka, a zatim pokreni odabrani prototip, k-NN, SVM, Random Forest, MLP ili CNN trening.",
    },
    training: {
      csvLabel: "CSV za obuku",
      csvSelected: (fileName: string) => `${fileName} je izabran za pregled.`,
      csvPlaceholder:
        "Koristi oznaceni format za obuku: label, pa zatim pixel0 do pixel783.",
      splitTrain: "Udeo za obuku",
      splitValidation: "Udeo za validaciju",
      splitTest: "Udeo za test",
      modelLabel: "Model za obuku",
      modelOptionPrototype: "Referentni prototip",
      modelOptionKnn: "k-NN klasifikator",
      modelOptionSvm: "SVM klasifikator",
      modelOptionRandomForest: "Random Forest klasifikator",
      modelOptionMlp: "MLP klasifikator",
      modelOptionCnn: "CNN klasifikator",
      modelHelper:
        "Zavrsena pokretanja koriste istu zajednicku rang-listu i birac modela bez obzira na porodicu.",
      nameLabel: "Naziv modela",
      nameHelper:
        "Prva verzija zahteva rucni naziv modela. Aplikacija ne predlaze podrazumevane nazive, a svako pokretanje se pojavljuje kao zasebna stavka na zajednickoj rang-listi.",
      seedLabel: "Seed",
      seedHelper: "Cuva se uz model radi ponovljivog deljenja i evaluacije.",
      examplesCapLabel: "Maksimalno primera po oznaci",
      examplesCapHelper:
        "Preporuceno 4. Gornja granica 5000 za prenosivi prototipski bazni model.",
      prototypeBlendLabel: "Mesanje prototipa",
      prototypeBlendHelper:
        "Preporuceno 0.35. Veca vrednost se vise oslanja na isporuceni referentni prototip.",
      confidenceTemperatureLabel: "Temperatura pouzdanosti",
      confidenceTemperatureHelper:
        "Preporuceno 18. Gornja granica 40 da skaliranje pouzdanosti ostane citljivo.",
      neighborsLabel: "Susedi",
      neighborsHelper:
        "Preporuceno 5. Nize vrednosti ostaju ostrije na manjim skupovima iz ucionice.",
      pcaComponentsLabel: "PCA komponente",
      pcaComponentsHelperKnn:
        "Preporuceno 16. Odrzava projekciju obelezja laganom za brzo iteriranje.",
      regularizationLabel: "Regularizacija",
      regularizationHelper:
        "Preporuceno 1.0. Vece vrednosti agresivnije prilagodjavaju model otpremljenoj podeli.",
      maxIterationsLabel: "Maksimalan broj iteracija",
      maxIterationsHelper:
        "Preporuceno 5000. Duzе pokretanje moze pomoci konvergenciji na zahtevnijim podelama.",
      pcaComponentsHelperSvm:
        "Preporuceno 16. Odrzava linearnu marginu stabilnom na manjim skupovima iz ucionice.",
      estimatorsLabel: "Broj estimatora",
      estimatorsHelper:
        "Preporuceno 96. Vise stabala poboljsava stabilnost, ali produzava vreme obuke.",
      maxDepthLabel: "Maksimalna dubina",
      maxDepthHelper:
        "Preporuceno 16. Plića stabla je lakse zadrzati opstim na manjim otpremama.",
      pcaComponentsHelperRandomForest:
        "Preporuceno 16. Odrzava projekciju kompaktnom pre faze sume.",
      epochsLabel: "Epohe",
      epochsHelper:
        "Preporuceno 4. Neka prilagodjene duboke obuke ostanu dovoljno kratke za odziv na CPU hardveru u ucionici.",
      batchSizeLabel: "Velicina batch-a",
      batchSizeHelper:
        "Preporuceno 32. Manji batch-evi odrzavaju prakticnost duboke obuke u prvoj verziji na prenosivom hardveru.",
      learningRateLabel: "Stopa ucenja",
      learningRateHelper: (modelFamily: TrainingModelFamily) =>
        modelFamily === "mlp"
          ? "Preporuceno 0.002 za fiksnu MLP arhitekturu."
          : "Preporuceno 0.001 za fiksnu CNN arhitekturu.",
      previewingButton: "Pregledam podelu",
      previewButton: "Pregledaj podelu",
      startingButton: "Pokrecem obuku",
      runningButton: "Obuka je u toku",
      startButton: "Pokreni obuku",
      splitTotalWarning: "Udeli moraju dati zbir 100 pre pregleda.",
      idleHelp:
        "Izaberi oznaceni CSV i pregledaj broj primera za obuku, validaciju i test pre pokretanja prilagodjene obuke.",
      previewError: (message: string) => `Pregled CSV-a nije uspeo: ${message}`,
      previewLoading: "Proveravam CSV semu i racunam velicine podela.",
      previewFile: "Fajl",
      previewExamples: "Primeri",
      previewFeatures: "Obelezja",
      previewLabels: "Oznake",
      previewRangeTo: "do",
      previewTrainExamples: (count: number) => `Obuka ${count} primera`,
      previewValidationExamples: (count: number) =>
        `Validacija ${count} primera`,
      previewTestExamples: (count: number) => `Test ${count} primera`,
      previewReadyHelp:
        "Pregled je spreman. Pokreni obuku da napravis unos za prilagodjeni model bez napustanja zajednicke rang-liste.",
      submittingJob: "Saljem posao za prilagodjenu obuku.",
      completedJob: (modelName: string) =>
        `${modelName} je sada dostupan na zajednickoj listi modela.`,
      failedJob: (message: string) =>
        `Prilagodjena obuka nije uspela: ${message}`,
      deleteButton: "Obrisi prilagodjeni model",
      deletingButton: "Brisem prilagodjeni model",
      validationTrainPositive: "Udeo za obuku mora biti veci od nule.",
      validationPreviewFirst:
        "Pregledaj podelu skupa podataka pre pokretanja obuke.",
      validationNameRequired: "Naziv modela je obavezan pre pokretanja obuke.",
      validationDuplicateName: (modelName: string) =>
        `Naziv modela '${modelName}' je vec zauzet. Izaberi jedinstven naziv.`,
      fallbackPreviewRequest: "Zahtev za pregled CSV-a nije uspeo.",
      fallbackTrainingRequest: "Zahtev za posao obuke nije uspeo.",
      fallbackTrainingStatus: "Zahtev za status obuke nije uspeo.",
      fallbackFailedJob: (modelName: string) =>
        `${modelName} nije uspesno zavrsio obradu.`,
      deleteSuccess: (modelName: string) =>
        `Prilagodjeni model je obrisan: ${modelName}.`,
      deleteError: (message: string) => `Brisanje nije uspelo: ${message}`,
      fallbackDelete: "Brisanje prilagodjenog modela nije uspelo.",
      prototypeWarningLargeCap:
        "Veliki limit po oznaci moze usporiti prototipski bazni model na hardveru iz ucionice.",
      prototypeWarningBlend:
        "Visoka vrednost mesanja prototipa oslanja se na isporuceni bazni model i moze prikriti razlike u skupu podataka.",
      prototypeWarningTemperature:
        "Veoma visoka temperatura pouzdanosti moze uciniti razlike u rezultatima ostrijim nego sto stvarne udaljenosti jesu.",
      deepWarningEpochs:
        "Duze duboke obuke mogu primetno produziti vreme obuke na hardveru iz ucionice.",
      deepWarningBatchSize:
        "Veoma veliki batch-evi mogu uciniti male otpremljene podele manje stabilnim tokom duboke obuke prve verzije.",
      deepWarningLearningRate:
        "Agresivne stope ucenja mogu uciniti prilagodjene duboke obuke nestabilnim ili sumovitim.",
      pcaWarning:
        "Vece PCA projekcije traze vise primera i mogu usporiti prilagodjena pokretanja na skupovima velicine ucionice.",
      knnWarningNeighbors:
        "Veci broj suseda agresivnije izravnava predvidjanja i moze zamutiti slicne cifre.",
      svmWarningRegularization:
        "Vece SVM vrednosti regularizacije mogu se usko prilagoditi maloj otpremljenoj podeli.",
      svmWarningMaxIter:
        "DuzI budzeti iteracija za SVM mogu primetno produziti vreme obuke prve verzije.",
      forestWarningEstimators:
        "Vece sume poboljsavaju stabilnost, ali ce povecati vreme obuke na hardveru iz ucionice.",
      forestWarningMaxDepth:
        "Vrlo duboka stabla mogu preprilagoditi male otpremljene skupove podataka.",
    },
    leaderboard: {
      kicker: "Rang-lista modela",
      heading: "Uporedi dostupne unose",
      sortLabel: "Sortiraj rang-listu po",
      accuracy: "Tacnost",
      macroF1: "Makro F1",
      inferenceLatency: "Latencija inferencije",
      modelColumn: "Model",
      latencyColumn: "Latencija",
      loading:
        "Rang-lista modela ce se popuniti kada backend zavrsi podizanje.",
      unavailable: "Rang-lista modela nije dostupna dok backend ne odgovori.",
    },
    modelPicker: {
      label: "Model",
      loading: "Ucitavam modele",
      unavailable: "Backend nedostupan",
      empty: "Nema dostupnih modela",
    },
    details: {
      kicker: "Detalji modela",
      emptyTitle: "Izaberi model",
      builtIn: "Ugradjen",
      custom: "Prilagodjen",
      datasetSource: "Izvor skupa podataka",
      trainedAt: "Treniran",
      datasetSizes: "Velicine skupa",
      inputShape: "Ulazni oblik",
      overviewTab: "Pregled",
      deepTab: "Duboki detalji",
      deepKicker: "Duboki detalji",
      architectureSummary: "Pregled arhitekture",
      layer: (index: number) => `Sloj ${index + 1}`,
      epochCurvesKicker: "Krive epoha",
      trainingProgression: "Napredak obuke",
      epoch: "Epoha",
      trainLoss: "Gubitak na obuci",
      validationLoss: "Gubitak na validaciji",
      trainAccuracy: "Tacnost na obuci",
      validationAccuracy: "Tacnost na validaciji",
      trainingSnapshotKicker: "Snimak obuke",
      savedConfigAndSeed: "Sacuvana konfiguracija i seed",
      seedLabel: "Seed",
      classifierLabel: "Klasifikator",
      uploadedFile: "Otpremljeni fajl",
      splitLabel: "Podela",
      metricsKicker: "Metrike",
      leaderboardMetrics: "Metrike rang-liste",
      macroPrecision: "Makro preciznost",
      macroRecall: "Makro odziv",
      hyperparametersKicker: "Hiperparametri",
      referenceConfiguration: "Referentna konfiguracija",
      confusionMatrixKicker: "Matrica konfuzije",
      evaluationBreakdown: "Razrada evaluacije",
      actualPred: "Stvarno\\Pred",
      samplePredictionsKicker: "Primeri predvidjanja",
      heldOutExamples: "Zadrzani primeri",
      samplePrediction: (label: number, predicted: number) =>
        `Stvarno ${label} predvidjeno ${predicted}`,
      emptyHelp:
        "Izaberi model sa rang-liste da pregledas njegove metapodatke i detalje evaluacije.",
    },
    prediction: {
      kicker: "Predvidjanje",
      title: "Profil pouzdanosti",
      idleHelp:
        "Nacrtaj cifru i pokreni izabrani model da vidis predvidjenu klasu i celu raspodelu pouzdanosti po ciframa.",
      error: (message: string) => `Predvidjanje nije uspelo: ${message}`,
      loading: "Normalizujem skicu i ocenjujem je izabranim modelom.",
      headline: (modelName: string, digit: number) =>
        `${modelName} predvidja ${digit}`,
    },
    common: {
      pending: "Na cekanju",
      trainShort: "Obuka",
      validationShort: "Validacija",
      testShort: "Test",
    },
  },
} as const;

export function App() {
  const [language, setLanguage] = useState<Language>("sr");
  const [bootstrapState, setBootstrapState] = useState<BootstrapState>({
    kind: "loading",
  });
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>("predict");
  const [leaderboardMetric, setLeaderboardMetric] =
    useState<SortableMetricKey>("accuracy");
  const [selectedModelId, setSelectedModelId] = useState("");
  const [canvasPixels, setCanvasPixels] = useState<number[]>(() =>
    createBlankCanvas(),
  );
  const [predictionState, setPredictionState] = useState<PredictionState>({
    kind: "idle",
  });
  const [isPainting, setIsPainting] = useState(false);
  const [trainingSplit, setTrainingSplit] = useState<TrainingSplitForm>(
    DEFAULT_TRAINING_SPLIT,
  );
  const [trainingFile, setTrainingFile] = useState<File | null>(null);
  const [trainingPreviewState, setTrainingPreviewState] =
    useState<TrainingPreviewState>({ kind: "idle" });
  const [customTrainingForm, setCustomTrainingForm] =
    useState<CustomTrainingForm>(DEFAULT_CUSTOM_TRAINING_FORM);
  const [trainingJobState, setTrainingJobState] = useState<TrainingJobState>({
    kind: "idle",
  });
  const [modelDeleteFeedback, setModelDeleteFeedback] = useState<string | null>(
    null,
  );
  const [isDeletingModel, setIsDeletingModel] = useState(false);
  const [modelDetailsTab, setModelDetailsTab] =
    useState<ModelDetailsTab>("overview");

  function applyBootstrapPayload(
    healthPayload: HealthPayload,
    modelsPayload: ModelsPayload,
    preferredModelId?: string,
  ) {
    setBootstrapState({
      kind: "ready",
      health: healthPayload,
      models: modelsPayload.models,
    });

    setSelectedModelId((currentModelId) => {
      const nextSelectedModelId = preferredModelId ?? currentModelId;

      if (
        nextSelectedModelId &&
        modelsPayload.models.some((model) => model.id === nextSelectedModelId)
      ) {
        return nextSelectedModelId;
      }

      return modelsPayload.models[0]?.id ?? "";
    });
  }

  async function refreshBootstrap(preferredModelId?: string) {
    try {
      const { healthPayload, modelsPayload } = await loadBootstrapPayload();

      applyBootstrapPayload(healthPayload, modelsPayload, preferredModelId);
      return modelsPayload;
    } catch (error) {
      setBootstrapState({
        kind: "error",
        message:
          error instanceof Error ? error.message : "Backend check failed.",
      });
      throw error;
    }
  }

  useEffect(() => {
    let isActive = true;

    async function loadAppBootstrap() {
      try {
        const { healthPayload, modelsPayload } = await loadBootstrapPayload();

        if (!isActive) {
          return;
        }

        applyBootstrapPayload(healthPayload, modelsPayload);
      } catch (error) {
        if (!isActive) {
          return;
        }

        setBootstrapState({
          kind: "error",
          message:
            error instanceof Error ? error.message : "Backend check failed.",
        });
      }
    }

    void loadAppBootstrap();

    return () => {
      isActive = false;
    };
  }, []);

  useEffect(() => {
    setModelDetailsTab("overview");
  }, [selectedModelId]);

  const hasInk = canvasPixels.some((pixel) => pixel > 0);
  const selectedModel =
    bootstrapState.kind === "ready"
      ? bootstrapState.models.find((model) => model.id === selectedModelId)
      : undefined;
  const sortedModels =
    bootstrapState.kind === "ready"
      ? [...bootstrapState.models].sort((leftModel, rightModel) =>
          compareModels(leftModel, rightModel, leaderboardMetric),
        )
      : [];
  const trainingSplitTotal =
    trainingSplit.train + trainingSplit.validation + trainingSplit.test;
  const selectedModelHasDeepDetails = Boolean(
    selectedModel?.deep_details?.architecture_summary?.length ||
    selectedModel?.deep_details?.epoch_curves?.length,
  );
  const canPreviewTrainingSplit =
    bootstrapState.kind === "ready" &&
    trainingFile !== null &&
    trainingPreviewState.kind !== "loading";
  const copy = APP_COPY[language];

  const readinessLabel =
    bootstrapState.kind === "ready" && bootstrapState.health.storage.ready
      ? copy.hero.statusReady
      : bootstrapState.kind === "error"
        ? copy.hero.statusError
        : copy.hero.statusLoading;

  async function handlePredict() {
    if (!selectedModelId || !hasInk) {
      return;
    }

    setPredictionState({ kind: "loading" });

    try {
      const payload = await fetchJson<PredictionPayload>("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model_id: selectedModelId,
          canvas: {
            width: DRAWING_GRID_SIZE,
            height: DRAWING_GRID_SIZE,
            pixels: canvasPixels,
          },
        }),
      });

      setPredictionState({ kind: "ready", payload });
    } catch (error) {
      setPredictionState({
        kind: "error",
        message:
          error instanceof Error ? error.message : "Prediction request failed.",
      });
    }
  }

  function handlePaint(index: number) {
    setCanvasPixels((currentPixels) => {
      if (currentPixels[index] === 255) {
        return currentPixels;
      }

      const nextPixels = currentPixels.slice();
      nextPixels[index] = 255;
      return nextPixels;
    });

    if (predictionState.kind !== "idle") {
      setPredictionState({ kind: "idle" });
    }
  }

  function handleClearBoard() {
    setCanvasPixels(createBlankCanvas());
    setPredictionState({ kind: "idle" });
  }

  function handleTrainingFileChange(event: ChangeEvent<HTMLInputElement>) {
    setTrainingFile(event.target.files?.[0] ?? null);
    setTrainingPreviewState({ kind: "idle" });
    setModelDeleteFeedback(null);

    if (
      trainingJobState.kind !== "running" &&
      trainingJobState.kind !== "submitting"
    ) {
      setTrainingJobState({ kind: "idle" });
    }
  }

  function handleTrainingSplitChange(
    key: keyof TrainingSplitForm,
    value: string,
  ) {
    const numericValue = Number(value);

    setTrainingSplit((currentSplit) => ({
      ...currentSplit,
      [key]: Number.isFinite(numericValue) ? numericValue : 0,
    }));
    setTrainingPreviewState({ kind: "idle" });
    setModelDeleteFeedback(null);

    if (
      trainingJobState.kind !== "running" &&
      trainingJobState.kind !== "submitting"
    ) {
      setTrainingJobState({ kind: "idle" });
    }
  }

  function handleTrainingModelNameChange(value: string) {
    setCustomTrainingForm((currentForm) => ({
      ...currentForm,
      modelName: value,
    }));
    setModelDeleteFeedback(null);

    if (
      trainingJobState.kind !== "running" &&
      trainingJobState.kind !== "submitting"
    ) {
      setTrainingJobState({ kind: "idle" });
    }
  }

  function handleTrainingModelFamilyChange(value: TrainingModelFamily) {
    setCustomTrainingForm((currentForm) => ({
      ...currentForm,
      modelFamily: value,
    }));
    setModelDeleteFeedback(null);

    if (
      trainingJobState.kind !== "running" &&
      trainingJobState.kind !== "submitting"
    ) {
      setTrainingJobState({ kind: "idle" });
    }
  }

  function handleTrainingSettingChange(
    key: Exclude<keyof CustomTrainingForm, "modelName" | "modelFamily">,
    value: string,
  ) {
    const numericValue = Number(value);

    setCustomTrainingForm((currentForm) => ({
      ...currentForm,
      [key]: Number.isFinite(numericValue) ? numericValue : 0,
    }));
    setModelDeleteFeedback(null);

    if (
      trainingJobState.kind !== "running" &&
      trainingJobState.kind !== "submitting"
    ) {
      setTrainingJobState({ kind: "idle" });
    }
  }

  async function handlePreviewTrainingSplit() {
    if (!trainingFile) {
      return;
    }

    if (trainingSplit.train <= 0) {
      setTrainingPreviewState({
        kind: "error",
        message: copy.training.validationTrainPositive,
      });
      return;
    }

    if (Math.abs(trainingSplitTotal - 100) > 0.001) {
      setTrainingPreviewState({
        kind: "error",
        message: copy.training.splitTotalWarning,
      });
      return;
    }

    setTrainingPreviewState({ kind: "loading" });

    try {
      const payload = await fetchJson<TrainingPreviewPayload>(
        "/api/training/csv-preview",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            file_name: trainingFile.name,
            csv_text: await readTextFromFile(trainingFile),
            split: {
              train_ratio: toSplitRatio(trainingSplit.train),
              validation_ratio: toSplitRatio(trainingSplit.validation),
              test_ratio: toSplitRatio(trainingSplit.test),
            },
          }),
        },
      );

      setTrainingPreviewState({ kind: "ready", payload });
    } catch (error) {
      setTrainingPreviewState({
        kind: "error",
        message:
          error instanceof Error
            ? error.message
            : copy.training.fallbackPreviewRequest,
      });
    }
  }

  async function handleStartTraining() {
    if (!trainingFile) {
      return;
    }

    const modelName = customTrainingForm.modelName.trim();
    if (!modelName) {
      setTrainingJobState({
        kind: "error",
        message: copy.training.validationNameRequired,
      });
      return;
    }

    if (trainingPreviewState.kind !== "ready") {
      setTrainingJobState({
        kind: "error",
        message: copy.training.validationPreviewFirst,
      });
      return;
    }

    if (
      bootstrapState.kind === "ready" &&
      bootstrapState.models.some(
        (model) => model.name.toLowerCase() === modelName.toLowerCase(),
      )
    ) {
      setTrainingJobState({
        kind: "error",
        message: copy.training.validationDuplicateName(modelName),
      });
      return;
    }

    setModelDeleteFeedback(null);
    setTrainingJobState({ kind: "submitting" });

    try {
      const payload = await fetchJson<TrainingJobPayload>(
        "/api/training/jobs",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model_name: modelName,
            model_family: customTrainingForm.modelFamily,
            file_name: trainingFile.name,
            csv_text: await readTextFromFile(trainingFile),
            split: {
              train_ratio: toSplitRatio(trainingSplit.train),
              validation_ratio: toSplitRatio(trainingSplit.validation),
              test_ratio: toSplitRatio(trainingSplit.test),
            },
            seed: customTrainingForm.seed,
            hyperparameters:
              buildCustomTrainingHyperparameters(customTrainingForm),
          }),
        },
      );

      if (payload.job.status === "completed") {
        await refreshBootstrap(payload.job.model_id);
        setTrainingJobState({ kind: "completed", job: payload.job });
        return;
      }

      setTrainingJobState({ kind: "running", job: payload.job });
    } catch (error) {
      setTrainingJobState({
        kind: "error",
        message:
          error instanceof Error
            ? error.message
            : copy.training.fallbackTrainingRequest,
      });
    }
  }

  async function handleDeleteSelectedModel() {
    if (!selectedModel || selectedModel.kind !== "custom" || isDeletingModel) {
      return;
    }

    setIsDeletingModel(true);
    setModelDeleteFeedback(null);

    try {
      await fetchNoContent(`/api/models/${selectedModel.id}`, {
        method: "DELETE",
      });
      await refreshBootstrap();

      setModelDeleteFeedback(copy.training.deleteSuccess(selectedModel.name));

      if (
        trainingJobState.kind === "completed" &&
        trainingJobState.job.model_id === selectedModel.id
      ) {
        setTrainingJobState({ kind: "idle" });
      }
    } catch (error) {
      setModelDeleteFeedback(
        copy.training.deleteError(
          error instanceof Error ? error.message : copy.training.fallbackDelete,
        ),
      );
    } finally {
      setIsDeletingModel(false);
    }
  }

  const canChooseModel =
    bootstrapState.kind === "ready" && bootstrapState.models.length > 0;
  const canStartTraining =
    bootstrapState.kind === "ready" &&
    trainingFile !== null &&
    trainingPreviewState.kind === "ready" &&
    customTrainingForm.modelName.trim().length > 0 &&
    trainingJobState.kind !== "submitting" &&
    trainingJobState.kind !== "running";
  const trainingWarnings = getCustomTrainingWarnings(
    customTrainingForm,
    language,
  );
  const activeTrainingJobId =
    trainingJobState.kind === "running" ? trainingJobState.job.id : null;

  useEffect(() => {
    if (!activeTrainingJobId) {
      return;
    }

    let isActive = true;

    async function pollTrainingJob() {
      try {
        const payload = await fetchJson<TrainingJobPayload>(
          `/api/training/jobs/${activeTrainingJobId}`,
        );

        if (!isActive) {
          return;
        }

        if (payload.job.status === "completed") {
          await refreshBootstrap(payload.job.model_id);

          if (!isActive) {
            return;
          }

          setTrainingJobState({ kind: "completed", job: payload.job });
          return;
        }

        if (
          payload.job.status === "failed" ||
          payload.job.status === "cancelled"
        ) {
          setTrainingJobState({
            kind: "error",
            message:
              payload.job.error ??
              copy.training.fallbackFailedJob(payload.job.model_name),
          });
          return;
        }

        setTrainingJobState({ kind: "running", job: payload.job });
      } catch (error) {
        if (!isActive) {
          return;
        }

        setTrainingJobState({
          kind: "error",
          message:
            error instanceof Error
              ? error.message
              : copy.training.fallbackTrainingStatus,
        });
      }
    }

    const intervalId = window.setInterval(() => {
      void pollTrainingJob();
    }, TRAINING_POLL_INTERVAL_MS);

    void pollTrainingJob();

    return () => {
      isActive = false;
      window.clearInterval(intervalId);
    };
  }, [activeTrainingJobId]);

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">{copy.hero.eyebrow}</p>
          <h1>{copy.hero.title}</h1>
        </div>

        <div className="hero-meta" aria-live="polite">
          <div className="mode-toggle" role="group" aria-label="Language">
            <button
              className={`mode-toggle-button ${
                language === "sr" ? "mode-toggle-button--active" : ""
              }`}
              type="button"
              onClick={() => setLanguage("sr")}
            >
              {copy.languageButtons.sr}
            </button>
            <button
              className={`mode-toggle-button ${
                language === "en" ? "mode-toggle-button--active" : ""
              }`}
              type="button"
              onClick={() => setLanguage("en")}
            >
              {copy.languageButtons.en}
            </button>
          </div>
          <span
            className={`status-pill status-pill--${
              bootstrapState.kind === "ready"
                ? "ready"
                : bootstrapState.kind === "error"
                  ? "error"
                  : "loading"
            }`}
          >
            {readinessLabel}
          </span>
          <p className="hero-note">
            {bootstrapState.kind === "loading"
              ? copy.hero.noteLoading
              : bootstrapState.kind === "error"
                ? copy.hero.noteError(bootstrapState.message)
                : copy.hero.noteReady(bootstrapState.models.length)}
          </p>
        </div>
      </section>

      <section className="workspace-grid">
        <div className="workspace-primary">
          <section className="board-card">
            <div className="panel-heading">
              <div>
                <p className="panel-kicker">{copy.workspace.kicker}</p>
                <h2>
                  {workspaceMode === "predict"
                    ? copy.workspace.predictTitle
                    : copy.workspace.trainingTitle}
                </h2>
              </div>
              <div
                className="mode-toggle"
                role="group"
                aria-label="Workspace mode"
              >
                <button
                  className={`mode-toggle-button ${
                    workspaceMode === "predict"
                      ? "mode-toggle-button--active"
                      : ""
                  }`}
                  type="button"
                  onClick={() => setWorkspaceMode("predict")}
                >
                  {copy.workspace.predictModeButton}
                </button>
                <button
                  className={`mode-toggle-button ${
                    workspaceMode === "training"
                      ? "mode-toggle-button--active"
                      : ""
                  }`}
                  type="button"
                  onClick={() => setWorkspaceMode("training")}
                >
                  {copy.workspace.trainingModeButton}
                </button>
              </div>
            </div>

            {workspaceMode === "predict" ? (
              <>
                <div className="control-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={handleClearBoard}
                  >
                    {copy.board.clear}
                  </button>
                  <button
                    className="primary-button"
                    type="button"
                    disabled={
                      bootstrapState.kind !== "ready" ||
                      !selectedModelId ||
                      !hasInk ||
                      predictionState.kind === "loading"
                    }
                    onClick={() => {
                      void handlePredict();
                    }}
                  >
                    {predictionState.kind === "loading"
                      ? copy.board.scoring
                      : copy.board.runPrediction}
                  </button>
                </div>

                <p className="status-copy board-copy">
                  {copy.board.predictHelp}
                </p>

                <div className="drawing-stage">
                  <div
                    className="drawing-frame"
                    onPointerLeave={() => setIsPainting(false)}
                  >
                    <div
                      className="drawing-grid"
                      role="grid"
                      aria-label="Digit drawing board"
                    >
                      {canvasPixels.map((pixel, index) => {
                        const row = Math.floor(index / DRAWING_GRID_SIZE);
                        const column = index % DRAWING_GRID_SIZE;

                        return (
                          <button
                            key={index}
                            aria-label={`Paint row ${row + 1} column ${column + 1}`}
                            aria-pressed={pixel > 0}
                            className={`pixel-button ${
                              pixel > 0 ? "pixel-button--active" : ""
                            }`}
                            type="button"
                            onClick={() => handlePaint(index)}
                            onPointerDown={(event) => {
                              event.preventDefault();
                              setIsPainting(true);
                              handlePaint(index);
                            }}
                            onPointerEnter={() => {
                              if (isPainting) {
                                handlePaint(index);
                              }
                            }}
                            onPointerUp={() => setIsPainting(false)}
                          />
                        );
                      })}
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <>
                <p className="status-copy board-copy">
                  {copy.board.trainingIntro}
                </p>

                <div className="training-intake-grid">
                  <div className="training-field training-field--wide">
                    <label className="field-label" htmlFor="training-csv">
                      {copy.training.csvLabel}
                    </label>
                    <input
                      id="training-csv"
                      className="training-file-input"
                      type="file"
                      accept=".csv,text/csv"
                      onChange={handleTrainingFileChange}
                    />
                    <p className="helper-copy">
                      {trainingFile
                        ? copy.training.csvSelected(trainingFile.name)
                        : copy.training.csvPlaceholder}
                    </p>
                  </div>

                  <div className="training-split-grid">
                    <div className="training-field">
                      <label className="field-label" htmlFor="split-train">
                        {copy.training.splitTrain}
                      </label>
                      <input
                        id="split-train"
                        className="training-input"
                        type="number"
                        min="0"
                        max="100"
                        step="1"
                        value={trainingSplit.train}
                        onChange={(event) => {
                          handleTrainingSplitChange(
                            "train",
                            event.target.value,
                          );
                        }}
                      />
                    </div>

                    <div className="training-field">
                      <label className="field-label" htmlFor="split-validation">
                        {copy.training.splitValidation}
                      </label>
                      <input
                        id="split-validation"
                        className="training-input"
                        type="number"
                        min="0"
                        max="100"
                        step="1"
                        value={trainingSplit.validation}
                        onChange={(event) => {
                          handleTrainingSplitChange(
                            "validation",
                            event.target.value,
                          );
                        }}
                      />
                    </div>

                    <div className="training-field">
                      <label className="field-label" htmlFor="split-test">
                        {copy.training.splitTest}
                      </label>
                      <input
                        id="split-test"
                        className="training-input"
                        type="number"
                        min="0"
                        max="100"
                        step="1"
                        value={trainingSplit.test}
                        onChange={(event) => {
                          handleTrainingSplitChange("test", event.target.value);
                        }}
                      />
                    </div>
                  </div>

                  <div className="training-config-grid">
                    <div className="training-field">
                      <label
                        className="field-label"
                        htmlFor="training-model-family"
                      >
                        {copy.training.modelLabel}
                      </label>
                      <select
                        id="training-model-family"
                        className="model-select"
                        value={customTrainingForm.modelFamily}
                        onChange={(event) => {
                          handleTrainingModelFamilyChange(
                            event.target.value as TrainingModelFamily,
                          );
                        }}
                      >
                        <option value="prototype">
                          {copy.training.modelOptionPrototype}
                        </option>
                        <option value="knn">
                          {copy.training.modelOptionKnn}
                        </option>
                        <option value="svm">
                          {copy.training.modelOptionSvm}
                        </option>
                        <option value="random-forest">
                          {copy.training.modelOptionRandomForest}
                        </option>
                        <option value="mlp">
                          {copy.training.modelOptionMlp}
                        </option>
                        <option value="cnn">
                          {copy.training.modelOptionCnn}
                        </option>
                      </select>
                      <p className="helper-copy">{copy.training.modelHelper}</p>
                    </div>

                    <div className="training-field training-field--wide">
                      <label
                        className="field-label"
                        htmlFor="training-model-name"
                      >
                        {copy.training.nameLabel}
                      </label>
                      <input
                        id="training-model-name"
                        className="training-input"
                        type="text"
                        value={customTrainingForm.modelName}
                        onChange={(event) => {
                          handleTrainingModelNameChange(event.target.value);
                        }}
                      />
                      <p className="helper-copy">{copy.training.nameHelper}</p>
                    </div>

                    <div className="training-field">
                      <label className="field-label" htmlFor="training-seed">
                        {copy.training.seedLabel}
                      </label>
                      <input
                        id="training-seed"
                        className="training-input"
                        type="number"
                        min="0"
                        max="1000000"
                        step="1"
                        value={customTrainingForm.seed}
                        onChange={(event) => {
                          handleTrainingSettingChange(
                            "seed",
                            event.target.value,
                          );
                        }}
                      />
                      <p className="helper-copy">{copy.training.seedHelper}</p>
                    </div>
                  </div>

                  {customTrainingForm.modelFamily === "prototype" ? (
                    <div className="training-config-grid">
                      <div className="training-field">
                        <label
                          className="field-label"
                          htmlFor="training-cap-per-label"
                        >
                          {copy.training.examplesCapLabel}
                        </label>
                        <input
                          id="training-cap-per-label"
                          className="training-input"
                          type="number"
                          min="1"
                          max="5000"
                          step="1"
                          value={customTrainingForm.maxExamplesPerLabel}
                          onChange={(event) => {
                            handleTrainingSettingChange(
                              "maxExamplesPerLabel",
                              event.target.value,
                            );
                          }}
                        />
                        <p className="helper-copy">
                          {copy.training.examplesCapHelper}
                        </p>
                      </div>

                      <div className="training-field">
                        <label
                          className="field-label"
                          htmlFor="training-prototype-blend"
                        >
                          {copy.training.prototypeBlendLabel}
                        </label>
                        <input
                          id="training-prototype-blend"
                          className="training-input"
                          type="number"
                          min="0"
                          max="1"
                          step="0.05"
                          value={customTrainingForm.prototypeBlend}
                          onChange={(event) => {
                            handleTrainingSettingChange(
                              "prototypeBlend",
                              event.target.value,
                            );
                          }}
                        />
                        <p className="helper-copy">
                          {copy.training.prototypeBlendHelper}
                        </p>
                      </div>

                      <div className="training-field">
                        <label
                          className="field-label"
                          htmlFor="training-temperature"
                        >
                          {copy.training.confidenceTemperatureLabel}
                        </label>
                        <input
                          id="training-temperature"
                          className="training-input"
                          type="number"
                          min="1"
                          max="40"
                          step="1"
                          value={customTrainingForm.temperature}
                          onChange={(event) => {
                            handleTrainingSettingChange(
                              "temperature",
                              event.target.value,
                            );
                          }}
                        />
                        <p className="helper-copy">
                          {copy.training.confidenceTemperatureHelper}
                        </p>
                      </div>
                    </div>
                  ) : null}

                  {customTrainingForm.modelFamily === "knn" ? (
                    <div className="training-config-grid">
                      <div className="training-field">
                        <label
                          className="field-label"
                          htmlFor="training-neighbors"
                        >
                          {copy.training.neighborsLabel}
                        </label>
                        <input
                          id="training-neighbors"
                          className="training-input"
                          type="number"
                          min="1"
                          max="15"
                          step="1"
                          value={customTrainingForm.neighbors}
                          onChange={(event) => {
                            handleTrainingSettingChange(
                              "neighbors",
                              event.target.value,
                            );
                          }}
                        />
                        <p className="helper-copy">
                          {copy.training.neighborsHelper}
                        </p>
                      </div>

                      <div className="training-field">
                        <label
                          className="field-label"
                          htmlFor="training-pca-components"
                        >
                          {copy.training.pcaComponentsLabel}
                        </label>
                        <input
                          id="training-pca-components"
                          className="training-input"
                          type="number"
                          min="4"
                          max="64"
                          step="1"
                          value={customTrainingForm.pcaComponents}
                          onChange={(event) => {
                            handleTrainingSettingChange(
                              "pcaComponents",
                              event.target.value,
                            );
                          }}
                        />
                        <p className="helper-copy">
                          {copy.training.pcaComponentsHelperKnn}
                        </p>
                      </div>
                    </div>
                  ) : null}

                  {customTrainingForm.modelFamily === "svm" ? (
                    <div className="training-config-grid">
                      <div className="training-field">
                        <label
                          className="field-label"
                          htmlFor="training-regularization"
                        >
                          {copy.training.regularizationLabel}
                        </label>
                        <input
                          id="training-regularization"
                          className="training-input"
                          type="number"
                          min="0.1"
                          max="10"
                          step="0.1"
                          value={customTrainingForm.regularization}
                          onChange={(event) => {
                            handleTrainingSettingChange(
                              "regularization",
                              event.target.value,
                            );
                          }}
                        />
                        <p className="helper-copy">
                          {copy.training.regularizationHelper}
                        </p>
                      </div>

                      <div className="training-field">
                        <label
                          className="field-label"
                          htmlFor="training-max-iter"
                        >
                          {copy.training.maxIterationsLabel}
                        </label>
                        <input
                          id="training-max-iter"
                          className="training-input"
                          type="number"
                          min="500"
                          max="20000"
                          step="100"
                          value={customTrainingForm.maxIter}
                          onChange={(event) => {
                            handleTrainingSettingChange(
                              "maxIter",
                              event.target.value,
                            );
                          }}
                        />
                        <p className="helper-copy">
                          {copy.training.maxIterationsHelper}
                        </p>
                      </div>

                      <div className="training-field">
                        <label
                          className="field-label"
                          htmlFor="training-pca-components"
                        >
                          {copy.training.pcaComponentsLabel}
                        </label>
                        <input
                          id="training-pca-components"
                          className="training-input"
                          type="number"
                          min="4"
                          max="64"
                          step="1"
                          value={customTrainingForm.pcaComponents}
                          onChange={(event) => {
                            handleTrainingSettingChange(
                              "pcaComponents",
                              event.target.value,
                            );
                          }}
                        />
                        <p className="helper-copy">
                          {copy.training.pcaComponentsHelperSvm}
                        </p>
                      </div>
                    </div>
                  ) : null}

                  {customTrainingForm.modelFamily === "random-forest" ? (
                    <div className="training-config-grid">
                      <div className="training-field">
                        <label
                          className="field-label"
                          htmlFor="training-estimators"
                        >
                          {copy.training.estimatorsLabel}
                        </label>
                        <input
                          id="training-estimators"
                          className="training-input"
                          type="number"
                          min="10"
                          max="400"
                          step="1"
                          value={customTrainingForm.estimators}
                          onChange={(event) => {
                            handleTrainingSettingChange(
                              "estimators",
                              event.target.value,
                            );
                          }}
                        />
                        <p className="helper-copy">
                          {copy.training.estimatorsHelper}
                        </p>
                      </div>

                      <div className="training-field">
                        <label
                          className="field-label"
                          htmlFor="training-max-depth"
                        >
                          {copy.training.maxDepthLabel}
                        </label>
                        <input
                          id="training-max-depth"
                          className="training-input"
                          type="number"
                          min="2"
                          max="64"
                          step="1"
                          value={customTrainingForm.maxDepth}
                          onChange={(event) => {
                            handleTrainingSettingChange(
                              "maxDepth",
                              event.target.value,
                            );
                          }}
                        />
                        <p className="helper-copy">
                          {copy.training.maxDepthHelper}
                        </p>
                      </div>

                      <div className="training-field">
                        <label
                          className="field-label"
                          htmlFor="training-pca-components"
                        >
                          {copy.training.pcaComponentsLabel}
                        </label>
                        <input
                          id="training-pca-components"
                          className="training-input"
                          type="number"
                          min="4"
                          max="64"
                          step="1"
                          value={customTrainingForm.pcaComponents}
                          onChange={(event) => {
                            handleTrainingSettingChange(
                              "pcaComponents",
                              event.target.value,
                            );
                          }}
                        />
                        <p className="helper-copy">
                          {copy.training.pcaComponentsHelperRandomForest}
                        </p>
                      </div>
                    </div>
                  ) : null}

                  {customTrainingForm.modelFamily === "mlp" ||
                  customTrainingForm.modelFamily === "cnn" ? (
                    <div className="training-config-grid">
                      <div className="training-field">
                        <label
                          className="field-label"
                          htmlFor="training-epochs"
                        >
                          {copy.training.epochsLabel}
                        </label>
                        <input
                          id="training-epochs"
                          className="training-input"
                          type="number"
                          min="1"
                          max="12"
                          step="1"
                          value={customTrainingForm.epochs}
                          onChange={(event) => {
                            handleTrainingSettingChange(
                              "epochs",
                              event.target.value,
                            );
                          }}
                        />
                        <p className="helper-copy">
                          {copy.training.epochsHelper}
                        </p>
                      </div>

                      <div className="training-field">
                        <label
                          className="field-label"
                          htmlFor="training-batch-size"
                        >
                          {copy.training.batchSizeLabel}
                        </label>
                        <input
                          id="training-batch-size"
                          className="training-input"
                          type="number"
                          min="4"
                          max="128"
                          step="4"
                          value={customTrainingForm.batchSize}
                          onChange={(event) => {
                            handleTrainingSettingChange(
                              "batchSize",
                              event.target.value,
                            );
                          }}
                        />
                        <p className="helper-copy">
                          {copy.training.batchSizeHelper}
                        </p>
                      </div>

                      <div className="training-field">
                        <label
                          className="field-label"
                          htmlFor="training-learning-rate"
                        >
                          {copy.training.learningRateLabel}
                        </label>
                        <input
                          id="training-learning-rate"
                          className="training-input"
                          type="number"
                          min="0.0001"
                          max="0.05"
                          step="0.0005"
                          value={customTrainingForm.learningRate}
                          onChange={(event) => {
                            handleTrainingSettingChange(
                              "learningRate",
                              event.target.value,
                            );
                          }}
                        />
                        <p className="helper-copy">
                          {copy.training.learningRateHelper(
                            customTrainingForm.modelFamily,
                          )}
                        </p>
                      </div>
                    </div>
                  ) : null}
                </div>

                <div className="control-row">
                  <button
                    className="secondary-button"
                    type="button"
                    disabled={!canPreviewTrainingSplit}
                    onClick={() => {
                      void handlePreviewTrainingSplit();
                    }}
                  >
                    {trainingPreviewState.kind === "loading"
                      ? copy.training.previewingButton
                      : copy.training.previewButton}
                  </button>
                  <button
                    className="primary-button"
                    type="button"
                    disabled={!canStartTraining}
                    onClick={() => {
                      void handleStartTraining();
                    }}
                  >
                    {trainingJobState.kind === "submitting"
                      ? copy.training.startingButton
                      : trainingJobState.kind === "running"
                        ? copy.training.runningButton
                        : copy.training.startButton}
                  </button>
                </div>

                {trainingWarnings.length > 0 ? (
                  <div className="warning-stack" aria-live="polite">
                    {trainingWarnings.map((warning) => (
                      <p className="warning-copy" key={warning}>
                        {warning}
                      </p>
                    ))}
                  </div>
                ) : null}

                {Math.abs(trainingSplitTotal - 100) > 0.001 ? (
                  <p className="status-copy">
                    {copy.training.splitTotalWarning}
                  </p>
                ) : null}

                {trainingPreviewState.kind === "idle" ? (
                  <p className="status-copy">{copy.training.idleHelp}</p>
                ) : null}

                {trainingPreviewState.kind === "error" ? (
                  <p className="status-copy">
                    {copy.training.previewError(trainingPreviewState.message)}
                  </p>
                ) : null}

                {trainingPreviewState.kind === "loading" ? (
                  <p className="status-copy">{copy.training.previewLoading}</p>
                ) : null}

                {trainingPreviewState.kind === "ready" ? (
                  <>
                    <dl className="details-grid details-grid--compact">
                      <div>
                        <dt>{copy.training.previewFile}</dt>
                        <dd>{trainingPreviewState.payload.file_name}</dd>
                      </div>
                      <div>
                        <dt>{copy.training.previewExamples}</dt>
                        <dd>
                          {formatCount(
                            trainingPreviewState.payload.dataset.example_count,
                          )}
                        </dd>
                      </div>
                      <div>
                        <dt>{copy.training.previewFeatures}</dt>
                        <dd>
                          {formatCount(
                            trainingPreviewState.payload.dataset.feature_count,
                          )}
                        </dd>
                      </div>
                      <div>
                        <dt>{copy.training.previewLabels}</dt>
                        <dd>
                          {trainingPreviewState.payload.dataset.label_range.min}{" "}
                          {copy.training.previewRangeTo}{" "}
                          {trainingPreviewState.payload.dataset.label_range.max}
                        </dd>
                      </div>
                    </dl>

                    <div className="preview-stat-grid" aria-live="polite">
                      <article className="preview-stat-card">
                        <p className="preview-stat-copy">
                          {copy.training.previewTrainExamples(
                            trainingPreviewState.payload.split.counts.train,
                          )}
                        </p>
                        <strong>
                          {formatPercent(
                            trainingPreviewState.payload.split.ratios.train,
                          )}
                        </strong>
                      </article>
                      <article className="preview-stat-card">
                        <p className="preview-stat-copy">
                          {copy.training.previewValidationExamples(
                            trainingPreviewState.payload.split.counts
                              .validation,
                          )}
                        </p>
                        <strong>
                          {formatPercent(
                            trainingPreviewState.payload.split.ratios
                              .validation,
                          )}
                        </strong>
                      </article>
                      <article className="preview-stat-card">
                        <p className="preview-stat-copy">
                          {copy.training.previewTestExamples(
                            trainingPreviewState.payload.split.counts.test,
                          )}
                        </p>
                        <strong>
                          {formatPercent(
                            trainingPreviewState.payload.split.ratios.test,
                          )}
                        </strong>
                      </article>
                    </div>
                  </>
                ) : null}

                {trainingPreviewState.kind === "ready" &&
                trainingJobState.kind === "idle" ? (
                  <p className="status-copy">
                    {copy.training.previewReadyHelp}
                  </p>
                ) : null}

                {trainingJobState.kind === "submitting" ? (
                  <p className="status-copy">{copy.training.submittingJob}</p>
                ) : null}

                {trainingJobState.kind === "running" ? (
                  <div className="training-progress-card" aria-live="polite">
                    <div className="training-progress-row">
                      <p className="status-copy">
                        {trainingJobState.job.progress.stage}
                      </p>
                      <strong>
                        {Math.round(
                          trainingJobState.job.progress.percent * 100,
                        )}
                        %
                      </strong>
                    </div>
                    <div
                      aria-label="Training progress"
                      aria-valuemax={100}
                      aria-valuemin={0}
                      aria-valuenow={Math.round(
                        trainingJobState.job.progress.percent * 100,
                      )}
                      className="training-progress-bar"
                      role="progressbar"
                    >
                      <span
                        className="training-progress-fill"
                        style={{
                          width: `${Math.round(
                            trainingJobState.job.progress.percent * 100,
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                ) : null}

                {trainingJobState.kind === "completed" ? (
                  <p className="status-copy">
                    {copy.training.completedJob(
                      trainingJobState.job.model_name,
                    )}
                  </p>
                ) : null}

                {trainingJobState.kind === "error" ? (
                  <p className="status-copy">
                    {copy.training.failedJob(trainingJobState.message)}
                  </p>
                ) : null}

                {modelDeleteFeedback ? (
                  <p className="status-copy">{modelDeleteFeedback}</p>
                ) : null}
              </>
            )}
          </section>

          <section
            className="leaderboard-panel"
            aria-labelledby="leaderboard-heading"
          >
            <div className="panel-heading panel-heading--tight">
              <div>
                <p className="panel-kicker">{copy.leaderboard.kicker}</p>
                <h2 id="leaderboard-heading">{copy.leaderboard.heading}</h2>
              </div>
              <div className="sort-control">
                <label className="field-label" htmlFor="leaderboard-sort">
                  {copy.leaderboard.sortLabel}
                </label>
                <select
                  id="leaderboard-sort"
                  className="model-select"
                  disabled={bootstrapState.kind !== "ready"}
                  value={leaderboardMetric}
                  onChange={(event) => {
                    setLeaderboardMetric(
                      event.target.value as SortableMetricKey,
                    );
                  }}
                >
                  <option value="accuracy">{copy.leaderboard.accuracy}</option>
                  <option value="macro_f1">{copy.leaderboard.macroF1}</option>
                  <option value="avg_inference_ms">
                    {copy.leaderboard.inferenceLatency}
                  </option>
                </select>
              </div>
            </div>

            {bootstrapState.kind === "ready" ? (
              <div className="leaderboard-table-frame">
                <table
                  className="leaderboard-table"
                  aria-label="Model leaderboard"
                >
                  <thead>
                    <tr>
                      <th scope="col">{copy.leaderboard.modelColumn}</th>
                      <th scope="col">{copy.leaderboard.accuracy}</th>
                      <th scope="col">{copy.leaderboard.macroF1}</th>
                      <th scope="col">{copy.leaderboard.latencyColumn}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedModels.map((model, index) => {
                      const isSelected = model.id === selectedModelId;

                      return (
                        <tr
                          key={model.id}
                          className={`leaderboard-table-row ${
                            isSelected ? "leaderboard-table-row--selected" : ""
                          }`}
                          aria-selected={isSelected}
                          tabIndex={0}
                          onClick={() => {
                            setSelectedModelId(model.id);
                            setPredictionState({ kind: "idle" });
                          }}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              setSelectedModelId(model.id);
                              setPredictionState({ kind: "idle" });
                            }
                          }}
                        >
                          <th scope="row">
                            <div className="leaderboard-model-cell">
                              <span className="leaderboard-rank">
                                #{index + 1}
                              </span>
                              <div className="leaderboard-model-copy">
                                <span className="leaderboard-name">
                                  {model.name}
                                </span>
                                <span className="leaderboard-subline">
                                  {formatModelKind(model.kind, language)}
                                </span>
                              </div>
                            </div>
                          </th>
                          <td>
                            {formatMetric(model.metrics?.accuracy, "percent")}
                          </td>
                          <td>
                            {formatMetric(model.metrics?.macro_f1, "percent")}
                          </td>
                          <td>
                            {formatMetric(
                              model.metrics?.avg_inference_ms,
                              "ms",
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="status-copy">
                {bootstrapState.kind === "loading"
                  ? copy.leaderboard.loading
                  : copy.leaderboard.unavailable}
              </p>
            )}
          </section>
        </div>

        <section className="prediction-card" aria-live="polite">
          <div className="model-picker-block">
            <label className="field-label" htmlFor="model-select">
              {copy.modelPicker.label}
            </label>
            <select
              id="model-select"
              className="model-select"
              disabled={!canChooseModel}
              value={canChooseModel ? selectedModelId : ""}
              onChange={(event) => {
                setSelectedModelId(event.target.value);
                setPredictionState({ kind: "idle" });
              }}
            >
              {!canChooseModel ? (
                <option value="">
                  {bootstrapState.kind === "loading"
                    ? copy.modelPicker.loading
                    : bootstrapState.kind === "error"
                      ? copy.modelPicker.unavailable
                      : copy.modelPicker.empty}
                </option>
              ) : (
                bootstrapState.models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {`${model.name} (${formatModelKind(model.kind, language)})`}
                  </option>
                ))
              )}
            </select>
            {bootstrapState.kind === "ready" && selectedModel?.description ? (
              <p className="helper-copy">{selectedModel.description}</p>
            ) : null}
          </div>

          <div className="panel-heading">
            <div>
              <p className="panel-kicker">{copy.details.kicker}</p>
              <h2>{selectedModel?.name ?? copy.details.emptyTitle}</h2>
            </div>
            {selectedModel ? (
              <div className="panel-actions">
                <span
                  className={`kind-badge ${
                    selectedModel.kind === "custom"
                      ? "kind-badge--custom"
                      : "kind-badge--built-in"
                  }`}
                >
                  {formatModelKind(selectedModel.kind, language)}
                </span>
                {selectedModel.kind === "custom" ? (
                  <button
                    className="secondary-button secondary-button--compact"
                    type="button"
                    disabled={isDeletingModel}
                    onClick={() => {
                      void handleDeleteSelectedModel();
                    }}
                  >
                    {isDeletingModel
                      ? copy.training.deletingButton
                      : copy.training.deleteButton}
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>

          {selectedModel ? (
            <>
              <dl className="details-grid details-grid--compact">
                <div>
                  <dt>{copy.details.datasetSource}</dt>
                  <dd>
                    {selectedModel.dataset?.source ?? copy.common.pending}
                  </dd>
                </div>
                <div>
                  <dt>{copy.details.trainedAt}</dt>
                  <dd>{formatTimestamp(selectedModel.trained_at, language)}</dd>
                </div>
                <div>
                  <dt>{copy.details.datasetSizes}</dt>
                  <dd>{formatDatasetSizes(selectedModel.dataset, language)}</dd>
                </div>
                <div>
                  <dt>{copy.details.inputShape}</dt>
                  <dd>
                    {selectedModel.dataset?.image_shape ??
                      formatInputShape(selectedModel.input, language)}
                  </dd>
                </div>
              </dl>

              {selectedModelHasDeepDetails ? (
                <div
                  aria-label="Model detail sections"
                  className="detail-tabs"
                  role="tablist"
                >
                  <button
                    aria-controls="model-details-overview-panel"
                    aria-selected={modelDetailsTab === "overview"}
                    className={`detail-tab ${
                      modelDetailsTab === "overview" ? "detail-tab--active" : ""
                    }`}
                    id="model-details-overview-tab"
                    onClick={() => {
                      setModelDetailsTab("overview");
                    }}
                    role="tab"
                    type="button"
                  >
                    {copy.details.overviewTab}
                  </button>
                  <button
                    aria-controls="model-details-deep-panel"
                    aria-selected={modelDetailsTab === "deep"}
                    className={`detail-tab ${
                      modelDetailsTab === "deep" ? "detail-tab--active" : ""
                    }`}
                    id="model-details-deep-tab"
                    onClick={() => {
                      setModelDetailsTab("deep");
                    }}
                    role="tab"
                    type="button"
                  >
                    {copy.details.deepTab}
                  </button>
                </div>
              ) : null}

              {modelDetailsTab === "deep" && selectedModelHasDeepDetails ? (
                <div
                  aria-labelledby="model-details-deep-tab"
                  className="detail-section"
                  id="model-details-deep-panel"
                  role="tabpanel"
                >
                  <section className="detail-block">
                    <div className="panel-heading panel-heading--tight">
                      <div>
                        <p className="panel-kicker">
                          {copy.details.deepKicker}
                        </p>
                        <h3>{copy.details.architectureSummary}</h3>
                      </div>
                    </div>
                    <div className="sample-grid">
                      {(
                        selectedModel.deep_details?.architecture_summary ?? []
                      ).map((line, index) => (
                        <article
                          className="sample-card"
                          key={`${line}-${index}`}
                        >
                          <span className="sample-chip">
                            {copy.details.layer(index)}
                          </span>
                          <strong>{line}</strong>
                        </article>
                      ))}
                    </div>
                  </section>

                  <section className="detail-block">
                    <div className="panel-heading panel-heading--tight">
                      <div>
                        <p className="panel-kicker">
                          {copy.details.epochCurvesKicker}
                        </p>
                        <h3>{copy.details.trainingProgression}</h3>
                      </div>
                    </div>
                    <div className="matrix-frame">
                      <table className="matrix-table">
                        <thead>
                          <tr>
                            <th scope="col">{copy.details.epoch}</th>
                            <th scope="col">{copy.details.trainLoss}</th>
                            <th scope="col">{copy.details.validationLoss}</th>
                            <th scope="col">{copy.details.trainAccuracy}</th>
                            <th scope="col">
                              {copy.details.validationAccuracy}
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {(selectedModel.deep_details?.epoch_curves ?? []).map(
                            (curve) => (
                              <tr key={curve.epoch}>
                                <th scope="row">{curve.epoch}</th>
                                <td>{formatLoss(curve.train_loss)}</td>
                                <td>{formatLoss(curve.validation_loss)}</td>
                                <td>
                                  {formatMetric(
                                    curve.train_accuracy,
                                    "percent",
                                  )}
                                </td>
                                <td>
                                  {formatMetric(
                                    curve.validation_accuracy,
                                    "percent",
                                  )}
                                </td>
                              </tr>
                            ),
                          )}
                        </tbody>
                      </table>
                    </div>
                  </section>
                </div>
              ) : (
                <div
                  aria-labelledby={
                    selectedModelHasDeepDetails
                      ? "model-details-overview-tab"
                      : undefined
                  }
                  className="detail-section"
                  id={
                    selectedModelHasDeepDetails
                      ? "model-details-overview-panel"
                      : undefined
                  }
                  role={selectedModelHasDeepDetails ? "tabpanel" : undefined}
                >
                  {selectedModel.training ? (
                    <section className="detail-block">
                      <div className="panel-heading panel-heading--tight">
                        <div>
                          <p className="panel-kicker">
                            {copy.details.trainingSnapshotKicker}
                          </p>
                          <h3>{copy.details.savedConfigAndSeed}</h3>
                        </div>
                      </div>
                      <dl className="details-grid details-grid--compact">
                        <div>
                          <dt>{copy.details.seedLabel}</dt>
                          <dd>
                            {selectedModel.training.seed ?? copy.common.pending}
                          </dd>
                        </div>
                        <div>
                          <dt>{copy.details.classifierLabel}</dt>
                          <dd>
                            {formatClassifierLabel(
                              selectedModel.training.config_snapshot
                                ?.classifier,
                              language,
                            )}
                          </dd>
                        </div>
                        <div>
                          <dt>{copy.details.uploadedFile}</dt>
                          <dd>
                            {selectedModel.training.config_snapshot
                              ?.file_name ?? copy.common.pending}
                          </dd>
                        </div>
                        <div>
                          <dt>{copy.details.splitLabel}</dt>
                          <dd>
                            {formatConfiguredSplit(
                              selectedModel.training.config_snapshot?.split,
                              language,
                            )}
                          </dd>
                        </div>
                      </dl>
                    </section>
                  ) : null}

                  <section className="detail-block">
                    <div className="panel-heading panel-heading--tight">
                      <div>
                        <p className="panel-kicker">
                          {copy.details.metricsKicker}
                        </p>
                        <h3>{copy.details.leaderboardMetrics}</h3>
                      </div>
                    </div>
                    <dl className="details-grid details-grid--compact">
                      <div>
                        <dt>{copy.leaderboard.accuracy}</dt>
                        <dd>
                          {formatMetric(
                            selectedModel.metrics?.accuracy,
                            "percent",
                          )}
                        </dd>
                      </div>
                      <div>
                        <dt>{copy.details.macroPrecision}</dt>
                        <dd>
                          {formatMetric(
                            selectedModel.metrics?.macro_precision,
                            "percent",
                          )}
                        </dd>
                      </div>
                      <div>
                        <dt>{copy.details.macroRecall}</dt>
                        <dd>
                          {formatMetric(
                            selectedModel.metrics?.macro_recall,
                            "percent",
                          )}
                        </dd>
                      </div>
                      <div>
                        <dt>{copy.leaderboard.macroF1}</dt>
                        <dd>
                          {formatMetric(
                            selectedModel.metrics?.macro_f1,
                            "percent",
                          )}
                        </dd>
                      </div>
                      <div>
                        <dt>{copy.leaderboard.inferenceLatency}</dt>
                        <dd>
                          {formatMetric(
                            selectedModel.metrics?.avg_inference_ms,
                            "ms",
                          )}
                        </dd>
                      </div>
                    </dl>
                  </section>

                  <section className="detail-block">
                    <div className="panel-heading panel-heading--tight">
                      <div>
                        <p className="panel-kicker">
                          {copy.details.hyperparametersKicker}
                        </p>
                        <h3>{copy.details.referenceConfiguration}</h3>
                      </div>
                    </div>
                    <div className="tag-grid">
                      {Object.entries(selectedModel.hyperparameters ?? {}).map(
                        ([key, value]) => (
                          <div className="tag-card" key={key}>
                            <span className="tag-label">
                              {formatKeyLabel(key, language)}
                            </span>
                            <strong>{String(value)}</strong>
                          </div>
                        ),
                      )}
                    </div>
                  </section>

                  <section className="detail-block">
                    <div className="panel-heading panel-heading--tight">
                      <div>
                        <p className="panel-kicker">
                          {copy.details.confusionMatrixKicker}
                        </p>
                        <h3>{copy.details.evaluationBreakdown}</h3>
                      </div>
                    </div>
                    <div className="matrix-frame">
                      <table className="matrix-table">
                        <thead>
                          <tr>
                            <th scope="col">{copy.details.actualPred}</th>
                            {renderDigitHeaders()}
                          </tr>
                        </thead>
                        <tbody>
                          {(
                            selectedModel.evaluation?.confusion_matrix ?? []
                          ).map((row, rowIndex) => (
                            <tr key={rowIndex}>
                              <th scope="row">{rowIndex}</th>
                              {row.map((value, columnIndex) => (
                                <td key={columnIndex}>{value}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>

                  <section className="detail-block">
                    <div className="panel-heading panel-heading--tight">
                      <div>
                        <p className="panel-kicker">
                          {copy.details.samplePredictionsKicker}
                        </p>
                        <h3>{copy.details.heldOutExamples}</h3>
                      </div>
                    </div>
                    <div className="sample-grid">
                      {(selectedModel.evaluation?.sample_predictions ?? []).map(
                        (sample, index) => (
                          <article
                            className="sample-card"
                            key={`${sample.label}-${index}`}
                          >
                            <span className="sample-chip">
                              {copy.details.samplePrediction(
                                sample.label,
                                sample.predicted,
                              )}
                            </span>
                            <strong>
                              {formatConfidence(sample.confidence)}
                            </strong>
                          </article>
                        ),
                      )}
                    </div>
                  </section>
                </div>
              )}
            </>
          ) : (
            <p className="status-copy">{copy.details.emptyHelp}</p>
          )}

          <div className="section-divider" />

          <div className="panel-heading panel-heading--tight">
            <div>
              <p className="panel-kicker">{copy.prediction.kicker}</p>
              <h3>{copy.prediction.title}</h3>
            </div>
          </div>

          {predictionState.kind === "idle" ? (
            <p className="status-copy">{copy.prediction.idleHelp}</p>
          ) : null}

          {predictionState.kind === "error" ? (
            <p className="status-copy">
              {copy.prediction.error(predictionState.message)}
            </p>
          ) : null}

          {predictionState.kind === "loading" ? (
            <p className="status-copy">{copy.prediction.loading}</p>
          ) : null}

          {predictionState.kind === "ready" ? (
            <>
              <p className="prediction-headline">
                {copy.prediction.headline(
                  predictionState.payload.model.name,
                  predictionState.payload.prediction.digit,
                )}
              </p>
              <div className="confidence-grid">
                {predictionState.payload.prediction.confidences.map(
                  (confidence, digit) => (
                    <div className="confidence-row" key={digit}>
                      <span className="confidence-digit">{digit}</span>
                      <div className="confidence-bar">
                        <span
                          className="confidence-fill"
                          style={{ width: `${Math.max(confidence * 100, 3)}%` }}
                        />
                      </div>
                      <span className="confidence-value">
                        {formatConfidence(confidence)}
                      </span>
                    </div>
                  ),
                )}
              </div>
            </>
          ) : null}
        </section>
      </section>
    </main>
  );
}

export default App;

function createBlankCanvas() {
  return Array.from({ length: DRAWING_PIXEL_COUNT }, () => 0);
}

async function fetchJson<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);

  if (!response.ok) {
    throw new Error(await readFetchErrorMessage(response, input));
  }

  return (await response.json()) as T;
}

async function loadBootstrapPayload() {
  let lastError: unknown;

  for (let attempt = 0; attempt < BOOTSTRAP_RETRY_ATTEMPTS; attempt += 1) {
    try {
      const [healthPayload, modelsPayload] = await Promise.all([
        fetchJson<HealthPayload>("/api/health"),
        fetchJson<ModelsPayload>("/api/models"),
      ]);

      return { healthPayload, modelsPayload };
    } catch (error) {
      lastError = error;

      if (
        !(error instanceof TypeError) ||
        attempt === BOOTSTRAP_RETRY_ATTEMPTS - 1
      ) {
        break;
      }

      await delay(BOOTSTRAP_RETRY_DELAY_MS);
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error("Backend check failed.");
}

function delay(milliseconds: number) {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}

async function fetchNoContent(input: string, init?: RequestInit) {
  const response = await fetch(input, init);

  if (!response.ok) {
    throw new Error(await readFetchErrorMessage(response, input));
  }
}

function formatConfidence(value: number) {
  return `${Math.round(value * 100)}%`;
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function readTextFromFile(file: File) {
  if (typeof file.text === "function") {
    return file.text();
  }

  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
        return;
      }

      reject(new Error("Could not read the selected CSV file."));
    };
    reader.onerror = () => {
      reject(
        reader.error ?? new Error("Could not read the selected CSV file."),
      );
    };

    reader.readAsText(file);
  });
}

function toSplitRatio(value: number) {
  return value / 100;
}

function compareModels(
  leftModel: ModelSummary,
  rightModel: ModelSummary,
  metric: SortableMetricKey,
) {
  const leftValue = getMetricValue(leftModel, metric);
  const rightValue = getMetricValue(rightModel, metric);

  if (metric === "avg_inference_ms") {
    return (
      leftValue - rightValue || leftModel.name.localeCompare(rightModel.name)
    );
  }

  return (
    rightValue - leftValue || leftModel.name.localeCompare(rightModel.name)
  );
}

function getMetricValue(model: ModelSummary, metric: SortableMetricKey) {
  const value = model.metrics?.[metric];

  if (value == null) {
    return metric === "avg_inference_ms"
      ? Number.POSITIVE_INFINITY
      : Number.NEGATIVE_INFINITY;
  }

  return value;
}

function formatMetric(value: number | undefined, mode: "percent" | "ms") {
  if (value == null) {
    return "n/a";
  }

  if (mode === "ms") {
    return `${value.toFixed(1)} ms`;
  }

  return `${(value * 100).toFixed(1)}%`;
}

function formatLoss(value: number | undefined) {
  if (value == null) {
    return "n/a";
  }

  return value.toFixed(3);
}

function formatTimestamp(value: string | undefined, language: Language) {
  if (!value) {
    return APP_COPY[language].common.pending;
  }

  return value.replace("T", " ").replace("Z", " UTC");
}

function formatDatasetSizes(
  dataset: ModelDataset | undefined,
  language: Language,
) {
  if (!dataset) {
    return APP_COPY[language].common.pending;
  }

  const copy = APP_COPY[language];

  return [
    `${copy.common.trainShort} ${formatCount(dataset.train_examples)}`,
    `${copy.common.validationShort} ${formatCount(
      dataset.validation_examples,
    )}`,
    `${copy.common.testShort} ${formatCount(dataset.test_examples)}`,
  ].join(" | ");
}

function formatCount(value: number | undefined) {
  if (value == null) {
    return "-";
  }

  return value.toLocaleString();
}

function formatInputShape(
  input:
    | {
        width: number;
        height: number;
        encoding?: string;
      }
    | undefined,
  language: Language,
) {
  if (!input) {
    return APP_COPY[language].common.pending;
  }

  return `${input.width}x${input.height}${
    input.encoding ? ` ${input.encoding}` : ""
  }`;
}

function formatKeyLabel(key: string, language: Language) {
  if (language === "sr") {
    const translatedLabel = {
      prototype_grid_size: "Velicina mreze prototipa",
      distance_metric: "Metrika distance",
      epochs: "Broj epoha",
      batch_size: "Velicina batch-a",
      learning_rate: "Stopa ucenja",
      neighbors: "Broj suseda",
      pca_components: "PCA komponente",
      regularization: "Regularizacija",
      max_iter: "Maksimalan broj iteracija",
      estimators: "Broj estimatora",
      max_depth: "Maksimalna dubina",
      max_examples_per_label: "Maksimalno primera po oznaci",
      prototype_blend: "Mesanje prototipa",
      temperature: "Temperatura",
    }[key];

    if (translatedLabel) {
      return translatedLabel;
    }
  }

  return key
    .split(/[_-]/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatModelKind(kind: string, language: Language) {
  return kind === "custom"
    ? APP_COPY[language].details.custom
    : APP_COPY[language].details.builtIn;
}

function formatClassifierLabel(
  classifier: string | undefined,
  language: Language,
) {
  return classifier
    ? formatTrainingModelLabel(classifier, language)
    : APP_COPY[language].common.pending;
}

function formatConfiguredSplit(
  split:
    | {
        train_ratio: number;
        validation_ratio: number;
        test_ratio: number;
      }
    | undefined,
  language: Language,
) {
  if (!split) {
    return APP_COPY[language].common.pending;
  }

  const copy = APP_COPY[language];

  return [
    `${copy.common.trainShort} ${formatPercent(split.train_ratio)}`,
    `${copy.common.validationShort} ${formatPercent(split.validation_ratio)}`,
    `${copy.common.testShort} ${formatPercent(split.test_ratio)}`,
  ].join(" | ");
}

function getCustomTrainingWarnings(
  form: CustomTrainingForm,
  language: Language,
) {
  const copy = APP_COPY[language];
  const warnings: string[] = [];

  if (form.modelFamily === "prototype") {
    if (form.maxExamplesPerLabel > 1000) {
      warnings.push(copy.training.prototypeWarningLargeCap);
    }

    if (form.prototypeBlend > 0.75) {
      warnings.push(copy.training.prototypeWarningBlend);
    }

    if (form.temperature > 30) {
      warnings.push(copy.training.prototypeWarningTemperature);
    }

    return warnings;
  }

  if (form.modelFamily === "mlp" || form.modelFamily === "cnn") {
    if (form.epochs > 6) {
      warnings.push(copy.training.deepWarningEpochs);
    }

    if (form.batchSize > 64) {
      warnings.push(copy.training.deepWarningBatchSize);
    }

    if (form.learningRate > 0.01) {
      warnings.push(copy.training.deepWarningLearningRate);
    }

    return warnings;
  }

  if (form.pcaComponents > 32) {
    warnings.push(copy.training.pcaWarning);
  }

  if (form.modelFamily === "knn" && form.neighbors > 7) {
    warnings.push(copy.training.knnWarningNeighbors);
  }

  if (form.modelFamily === "svm") {
    if (form.regularization > 3) {
      warnings.push(copy.training.svmWarningRegularization);
    }

    if (form.maxIter > 10000) {
      warnings.push(copy.training.svmWarningMaxIter);
    }
  }

  if (form.modelFamily === "random-forest") {
    if (form.estimators > 200) {
      warnings.push(copy.training.forestWarningEstimators);
    }

    if (form.maxDepth > 24) {
      warnings.push(copy.training.forestWarningMaxDepth);
    }
  }

  return warnings;
}

function buildCustomTrainingHyperparameters(form: CustomTrainingForm) {
  if (form.modelFamily === "mlp" || form.modelFamily === "cnn") {
    return {
      epochs: form.epochs,
      batch_size: form.batchSize,
      learning_rate: form.learningRate,
    };
  }

  if (form.modelFamily === "knn") {
    return {
      neighbors: form.neighbors,
      pca_components: form.pcaComponents,
    };
  }

  if (form.modelFamily === "svm") {
    return {
      regularization: form.regularization,
      max_iter: form.maxIter,
      pca_components: form.pcaComponents,
    };
  }

  if (form.modelFamily === "random-forest") {
    return {
      estimators: form.estimators,
      max_depth: form.maxDepth,
      pca_components: form.pcaComponents,
    };
  }

  return {
    max_examples_per_label: form.maxExamplesPerLabel,
    prototype_blend: form.prototypeBlend,
    temperature: form.temperature,
  };
}

function formatTrainingModelLabel(modelFamily: string, language: Language) {
  const copy = APP_COPY[language];

  if (modelFamily === "prototype" || modelFamily === "reference-prototype") {
    return copy.training.modelOptionPrototype;
  }

  if (modelFamily === "knn") {
    return language === "sr" ? "k-NN" : "k-NN";
  }

  if (modelFamily === "svm") {
    return "SVM";
  }

  if (modelFamily === "random-forest") {
    return language === "sr" ? "Random Forest" : "Random Forest";
  }

  if (modelFamily === "mlp") {
    return "MLP";
  }

  if (modelFamily === "cnn") {
    return "CNN";
  }

  return formatKeyLabel(modelFamily, language);
}

async function readFetchErrorMessage(response: Response, input: string) {
  let message = `Request to ${input} failed with ${response.status}`;

  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) {
      message = payload.detail;
    }
  } catch {
    // Ignore malformed error payloads and keep the default message.
  }

  return message;
}

function renderDigitHeaders() {
  return Array.from({ length: 10 }, (_, digit) => (
    <th key={digit} scope="col">
      {digit}
    </th>
  ));
}
