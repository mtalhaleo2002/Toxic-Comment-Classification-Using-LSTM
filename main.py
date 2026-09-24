import argparse
import html
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

if Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve():
    raise SystemExit("Run this project with .venv/bin/python or bash setup_and_run.sh.")

for variable, directory in {
    "MPLCONFIGDIR": "matplotlib",
    "KERAS_HOME": "keras",
    "XDG_CACHE_HOME": "xdg",
    "TMPDIR": "tmp",
}.items():
    location = PROJECT_ROOT / ".cache" / directory
    location.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(location)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the five toxic comment classification tasks.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "toxic_comments_dataset.csv")
    parser.add_argument("--text-column")
    parser.add_argument("--label-column")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--epochs", type=int, choices=range(10, 21), default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-words", type=int, default=20000)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--predict", action="append", help="A new comment to classify after training; can be repeated.")
    arguments = parser.parse_args()
    for name in ("batch_size", "max_length", "threads"):
        if getattr(arguments, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if arguments.max_words < 3:
        parser.error("--max-words must be at least 3.")
    if not 0 <= arguments.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 4294967295.")
    return arguments


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def task_one_load_and_explore(path):
    print("\nTASK 1: Dataset loading and initial exploration")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Supply the CSV using --data.")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("The dataset contains no records.")
    print("\nFirst 10 records:\n", frame.head(10).to_string(index=False))
    print("\nLast 8 records:\n", frame.tail(8).to_string(index=False))
    print("\nDataset shape:", frame.shape)
    print("\nColumn names:", frame.columns.tolist())
    print("\nData types:\n", frame.dtypes.to_string())
    numeric = frame.select_dtypes(include="number")
    print("\nNumerical summaries:\n", numeric.describe().to_string() if not numeric.empty else "No numerical columns.")
    return frame


def normalized_name(value):
    return re.sub(r"[\s-]+", "_", str(value).strip().lower())


def select_column(frame, explicit, candidates, option):
    if explicit:
        if explicit not in frame.columns:
            raise ValueError(f"Column {explicit!r} does not exist. Available columns: {frame.columns.tolist()}")
        return explicit
    matches = [column for column in frame.columns if normalized_name(column) in candidates]
    if len(matches) != 1:
        raise ValueError(f"Cannot choose {option} unambiguously. Supply {option} explicitly. Available columns: {frame.columns.tolist()}")
    return matches[0]


def clean_text(value):
    text = unicodedata.normalize("NFKC", html.unescape(str(value))).lower()
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = "".join(character if character.isalpha() or character.isspace() else " " for character in text)
    return " ".join(text.split())


def task_two_prepare(frame, arguments, output):
    print("\nTASK 2: Cleaning, missing values, duplicates, and target analysis")
    missing = frame.isna().sum().rename("missing_count")
    print("\nMissing values per column:\n", missing.to_string())
    missing.to_csv(output / "missing_values.csv")
    duplicates = frame[frame.duplicated(keep=False)]
    print(f"\nExact duplicate records beyond the first occurrence: {int(frame.duplicated().sum())}")
    print(f"Rows participating in exact duplicates: {len(duplicates)}")
    print(duplicates.to_string(index=False) if not duplicates.empty else "No exact duplicate records.")
    duplicates.to_csv(output / "duplicate_records.csv", index=False)
    text_column = select_column(frame, arguments.text_column, {"comment_text", "text", "comment", "comments", "content"}, "--text-column")
    flag_names = {"toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"}
    flags = [column for column in frame.columns if normalized_name(column) in flag_names]
    target_names = {"label", "class", "target", "category", "toxicity", "toxicity_label", "toxicity_level", "class_label"}
    if not arguments.label_column and len(flags) > 1 and not any(normalized_name(column) in target_names for column in frame.columns):
        raise ValueError("Multiple toxicity flag columns suggest multilabel data. This assignment needs one mutually exclusive class per comment. Supply an appropriate class column with --label-column; no automatic class conversion is performed.")
    label_candidates = target_names if any(normalized_name(column) in target_names for column in frame.columns) else {"toxic"}
    label_column = select_column(frame, arguments.label_column, label_candidates, "--label-column")
    if text_column == label_column:
        raise ValueError("Text and label columns must be different.")
    print(f"\nText column: {text_column}; target column: {label_column}")
    print(f"Duplicate comments beyond the first occurrence: {int(frame[text_column].duplicated().sum())}")
    raw_counts = frame[label_column].value_counts(dropna=False).rename("count")
    print("\nOriginal target distribution:\n", raw_counts.to_string())
    raw_counts.to_csv(output / "original_class_distribution.csv")
    working = frame.drop_duplicates().loc[:, [text_column, label_column]].copy()
    working.columns = ["raw_text", "label"]
    working.insert(0, "source_row", working.index + 2)
    count_before = len(working)
    working = working.dropna(subset=["raw_text", "label"]).copy()
    print(f"Rows removed for missing text or labels: {count_before - len(working)}")
    if pd.api.types.is_numeric_dtype(working["label"]):
        values = working["label"].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError("The target appears to contain continuous scores. Supply discrete class labels for softmax classification.")
    working["label"] = working["label"].astype(str).str.strip()
    working["cleaned_text"] = working["raw_text"].map(clean_text)
    empty = working["cleaned_text"].eq("") | working["label"].eq("")
    print(f"Rows removed for empty cleaned text or blank labels: {int(empty.sum())}")
    working = working.loc[~empty].copy()
    conflicts = working.groupby("cleaned_text")["label"].transform("nunique").gt(1)
    working.loc[conflicts].to_csv(output / "conflicting_labels.csv", index=False)
    print(f"Rows removed because identical cleaned text has conflicting labels: {int(conflicts.sum())}")
    working = working.loc[~conflicts].copy()
    repeated = working.duplicated(subset="cleaned_text")
    print(f"Additional repeated cleaned comments removed: {int(repeated.sum())}")
    working = working.loc[~repeated].reset_index(drop=True)
    counts = working["label"].value_counts().sort_index()
    if len(counts) < 2:
        raise ValueError("At least two distinct target classes must remain after cleaning.")
    distribution = pd.DataFrame({"count": counts, "percentage": counts / counts.sum() * 100})
    print("\nCleaned dataset shape:", working.shape)
    print("\nTarget distribution:\n", distribution.to_string())
    print("\nCleaned examples:\n", working[["cleaned_text", "label"]].head(10).to_string(index=False))
    distribution.to_csv(output / "class_distribution.csv")
    working.to_csv(output / "cleaned_dataset.csv", index=False)
    return working, text_column, label_column


def task_three_tokenize(frame, arguments, output):
    print("\nTASK 3: Stratified splitting, training-only tokenization, and padding")
    try:
        train_validation, test = train_test_split(frame, test_size=0.2, random_state=arguments.seed, stratify=frame["label"])
        train, validation = train_test_split(train_validation, test_size=0.2, random_state=arguments.seed, stratify=train_validation["label"])
        if any(set(partition["label"]) != set(frame["label"]) for partition in (train, validation, test)):
            raise ValueError("A class is absent from a partition.")
    except ValueError as error:
        counts = frame["label"].value_counts()
        if counts.min() < 3:
            raise ValueError(f"Cannot form stratified train/validation/test sets: each class needs at least three distinct comments. Counts: {counts.to_dict()}") from error
        print("The dataset is too small for the requested proportions; allocating at least one validation and test comment per class.")
        training_groups, validation_groups, testing_groups = [], [], []
        for _, group in frame.groupby("label", sort=True):
            test_count = max(1, int(np.ceil(len(group) * 0.2)))
            remaining, test_group = train_test_split(group, test_size=test_count, random_state=arguments.seed)
            validation_count = max(1, int(np.ceil(len(remaining) * 0.2)))
            train_group, validation_group = train_test_split(remaining, test_size=validation_count, random_state=arguments.seed)
            training_groups.append(train_group)
            validation_groups.append(validation_group)
            testing_groups.append(test_group)
        train = pd.concat(training_groups).sample(frac=1, random_state=arguments.seed)
        validation = pd.concat(validation_groups).sample(frac=1, random_state=arguments.seed)
        test = pd.concat(testing_groups).sample(frac=1, random_state=arguments.seed)
    partitions = {"train": train, "validation": validation, "test": test}
    expected_classes = set(frame["label"])
    if any(set(partition["label"]) != expected_classes for partition in partitions.values()):
        raise ValueError("Every class must occur in every partition. Supply more distinct examples for the rare classes.")
    encoder = LabelEncoder().fit(train["label"])
    print("Class encoding:", {label: index for index, label in enumerate(encoder.classes_)})
    tokenizer = Tokenizer(num_words=arguments.max_words, oov_token="<OOV>", filters="", lower=False)
    tokenizer.fit_on_texts(train["cleaned_text"].tolist())
    arrays = {}
    for name, partition in partitions.items():
        sequences = tokenizer.texts_to_sequences(partition["cleaned_text"].tolist())
        inputs = pad_sequences(sequences, maxlen=arguments.max_length, padding="post", truncating="post", dtype="int32")
        targets = encoder.transform(partition["label"]).astype("int32")
        arrays[name] = (inputs, targets)
        print(f"{name}: {len(partition)} comments, padded shape {inputs.shape}, class counts {partition['label'].value_counts().to_dict()}")
        print(f"{name} example integer sequence: {sequences[0]}")
        print(f"{name} example padded sequence: {inputs[0].tolist()}")
        partition[["source_row", "label"]].to_csv(output / f"{name}_rows.csv", index=False)
    vocabulary_size = min(arguments.max_words, len(tokenizer.word_index) + 1)
    print(f"Embedding vocabulary size: {vocabulary_size}; fixed sequence length: {arguments.max_length}")
    (output / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    write_json(output / "classes.json", encoder.classes_.tolist())
    return arrays, encoder.classes_.tolist(), vocabulary_size


def make_dataset(inputs, targets, arguments, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if training:
        dataset = dataset.shuffle(len(targets), seed=arguments.seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.batch(arguments.batch_size).with_options(options).prefetch(1)


def task_four_train(arrays, class_names, vocabulary_size, arguments, output):
    print("\nTASK 4: Model architecture and training")
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(arguments.max_length,), dtype="int32"),
        tf.keras.layers.Embedding(input_dim=vocabulary_size, output_dim=64, mask_zero=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(class_names)), y=arrays["train"][1])
    class_weights = {index: float(weight) for index, weight in enumerate(weights)}
    print("\nTraining class weights:", class_weights)
    checkpoint = output / "best_model.keras"
    history = model.fit(
        make_dataset(*arrays["train"], arguments, training=True),
        validation_data=make_dataset(*arrays["validation"], arguments),
        epochs=arguments.epochs,
        class_weight=class_weights,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True)],
        verbose=2,
    )
    history_frame = pd.DataFrame(history.history)
    history_frame.index = np.arange(1, len(history_frame) + 1)
    history_frame.index.name = "epoch"
    history_frame.to_csv(output / "training_history.csv")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric in zip(axes, ("loss", "accuracy")):
        axis.plot(history_frame.index, history_frame[metric], label="Training")
        axis.plot(history_frame.index, history_frame[f"val_{metric}"], label="Validation")
        axis.set(xlabel="Epoch", ylabel=metric.capitalize(), title=f"Training and validation {metric}")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=180)
    plt.close(figure)
    best_epoch = int(history_frame["val_loss"].idxmin())
    print(f"\nUsing the checkpoint from epoch {best_epoch}, selected by validation loss.")
    return tf.keras.models.load_model(checkpoint), class_weights, best_epoch


def task_five_evaluate(model, arrays, class_names, arguments, output, best_epoch):
    print("\nTASK 5: Held-out test evaluation and visualization")
    test_inputs, test_targets = arrays["test"]
    dataset = make_dataset(test_inputs, test_targets, arguments)
    metrics = model.evaluate(dataset, return_dict=True, verbose=0)
    probabilities = model.predict(dataset, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    class_ids = np.arange(len(class_names))
    matrix = confusion_matrix(test_targets, predictions, labels=class_ids)
    matrix_frame = pd.DataFrame(matrix, index=class_names, columns=class_names)
    matrix_frame.index.name = "Actual"
    matrix_frame.columns.name = "Predicted"
    report_text = classification_report(test_targets, predictions, labels=class_ids, target_names=class_names, digits=4, zero_division=0)
    report = classification_report(test_targets, predictions, labels=class_ids, target_names=class_names, output_dict=True, zero_division=0)
    print(f"\nTest loss: {metrics['loss']:.4f}\nTest accuracy: {metrics['accuracy']:.4f}")
    print("\nConfusion matrix (rows: actual; columns: predicted):\n", matrix_frame.to_string())
    print("\nClassification report:\n", report_text)
    matrix_frame.to_csv(output / "confusion_matrix.csv")
    (output / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame({name: values for name, values in report.items() if isinstance(values, dict)}).T.to_csv(output / "classification_report.csv")
    write_json(output / "metrics.json", {
        "test_loss": float(metrics["loss"]),
        "test_accuracy": float(metrics["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "best_epoch": best_epoch,
        "classification_report": report,
    })
    prediction_frame = pd.DataFrame({"actual": np.asarray(class_names)[test_targets], "predicted": np.asarray(class_names)[predictions]})
    for index, name in enumerate(class_names):
        prediction_frame[f"probability_{name}"] = probabilities[:, index]
    prediction_frame.to_csv(output / "test_predictions.csv", index=False)
    side = max(7, len(class_names) * 0.9)
    figure, axis = plt.subplots(figsize=(side, side))
    sns.heatmap(matrix_frame, annot=True, fmt="d", cmap="Blues", ax=axis, cbar=False)
    axis.set(title="Toxic comment classification: test confusion matrix", xlabel="Predicted class", ylabel="Actual class")
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    print(f"\nConfusion matrix image: {output / 'confusion_matrix.png'}")


def predict_new_comments(model, class_names, arguments, output):
    print("\nNEW COMMENTS: Predictions using the saved training vocabulary")
    comments = arguments.predict or [
        "Thank you for your thoughtful explanation.",
        "Your reply is rude and insulting.",
        "This is extremely abusive and hateful.",
        "I will hurt you if you keep posting.",
        "Attacking people because of their identity is unacceptable.",
    ]
    cleaned = [clean_text(comment) for comment in comments]
    if any(not comment for comment in cleaned):
        raise ValueError("A new comment is empty after cleaning; supply comments containing words.")
    tokenizer = tokenizer_from_json((output / "tokenizer.json").read_text(encoding="utf-8"))
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=arguments.max_length, padding="post", truncating="post", dtype="int32")
    probabilities = model(tf.convert_to_tensor(padded), training=False).numpy()
    predictions = probabilities.argmax(axis=1)
    oov_index = tokenizer.word_index[tokenizer.oov_token]
    results = pd.DataFrame({
        "comment": comments,
        "cleaned_text": cleaned,
        "predicted_label": [class_names[index] for index in predictions],
        "model_probability": probabilities.max(axis=1),
        "oov_fraction": [sequence.count(oov_index) / len(sequence) for sequence in sequences],
    })
    for index, label in enumerate(class_names):
        results[f"probability_{label}"] = probabilities[:, index]
    results.to_csv(output / "new_comment_predictions.csv", index=False)
    print(results[["comment", "predicted_label", "model_probability", "oov_fraction"]].to_string(index=False))
    print("These comments have no supplied ground-truth labels and are not part of test accuracy. Softmax scores are model outputs, not calibrated confidence.")


def main():
    arguments = parse_arguments()
    if not arguments.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {arguments.data}. Place the assignment CSV in the project or use --data.")
    output = arguments.output_dir.resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("--output-dir must be inside the project folder to keep generated files local.")
    output.mkdir(parents=True, exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(arguments.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(arguments.seed)
    tf.config.experimental.enable_op_determinism()
    frame = task_one_load_and_explore(arguments.data)
    cleaned, text_column, label_column = task_two_prepare(frame, arguments, output)
    arrays, class_names, vocabulary_size = task_three_tokenize(cleaned, arguments, output)
    model, class_weights, best_epoch = task_four_train(arrays, class_names, vocabulary_size, arguments, output)
    task_five_evaluate(model, arrays, class_names, arguments, output, best_epoch)
    predict_new_comments(model, class_names, arguments, output)
    write_json(output / "run_config.json", {
        "data": str(arguments.data.resolve()),
        "dataset_sha256": hashlib.sha256(arguments.data.read_bytes()).hexdigest(),
        "main_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "text_column": text_column,
        "label_column": label_column,
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "max_words": arguments.max_words,
        "max_length": arguments.max_length,
        "padding": "post",
        "truncating": "post",
        "cleaning": "HTML unescape, Unicode NFKC, lowercase, remove HTML and URLs, retain letters and spaces, collapse whitespace",
        "vocabulary_size": vocabulary_size,
        "raw_rows": len(frame),
        "cleaned_unique_rows": len(cleaned),
        "evaluation_note": "Identical cleaned comments never cross partitions. Small test sets give unstable estimates; example predictions are unlabeled demonstrations.",
        "classes": class_names,
        "class_weights": class_weights,
        "split_counts": {name: len(targets) for name, (_, targets) in arrays.items()},
        "tensorflow_version": tf.__version__,
    })
    print(f"\nAll five tasks completed. Results saved to {output}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise SystemExit(f"Error: {error}") from error
