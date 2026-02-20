"""Hybrid CNN-LSTM Autoencoder experiment pipeline.

This module adds a proposal-oriented workflow:
1) train on CIC-IDS2017 benign traffic only,
2) zero-shot evaluation on CSE-CIC-IDS2018,
3) few-shot unsupervised adaptation using benign target data,
4) post-adaptation evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
from random import Random
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass
class HybridExperimentConfig:
    sequence_length: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 15
    validation_split: float = 0.1
    early_stopping_patience: int = 3
    threshold_std_factor: float = 2.0
    few_shot_benign_ratio: float = 0.01
    random_seed: int = 1234


@dataclass
class HybridExperimentState:
    model: Any
    scaler: "MinMaxFeatureScaler"
    feature_columns: List[str]
    threshold: float
    config: HybridExperimentConfig
    adaptation_steps: int = 0
    training_history: Optional[Dict[str, List[float]]] = None


class MinMaxFeatureScaler:
    """Simple min-max scaler that avoids external dependencies."""

    def __init__(self) -> None:
        self.minimums: Dict[str, float] = {}
        self.maximums: Dict[str, float] = {}

    def fit(self, rows: Sequence[Mapping[str, Any]], feature_columns: Sequence[str]) -> None:
        if not rows:
            raise ValueError("Cannot fit scaler on empty rows.")

        for column in feature_columns:
            column_values = [_to_float(row.get(column, 0.0)) for row in rows]
            self.minimums[column] = min(column_values)
            self.maximums[column] = max(column_values)

    def transform(self, rows: Sequence[Mapping[str, Any]], feature_columns: Sequence[str]) -> List[List[float]]:
        matrix: List[List[float]] = []
        for row in rows:
            transformed_row: List[float] = []
            for column in feature_columns:
                value = _to_float(row.get(column, 0.0))
                min_value = self.minimums[column]
                max_value = self.maximums[column]
                scale = max_value - min_value
                if scale == 0:
                    transformed_row.append(0.0)
                else:
                    transformed_row.append((value - min_value) / scale)
            matrix.append(transformed_row)
        return matrix


def train_cic2017_normal(
    cic2017_data: Any,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    label_column: str = "Label",
    benign_token: str = "BENIGN",
    config: Optional[HybridExperimentConfig] = None,
    model_builder: Optional[Any] = None,
) -> HybridExperimentState:
    """Train hybrid model on benign-only data from source dataset."""
    resolved_config = config or HybridExperimentConfig()
    rows = _to_rows(cic2017_data)
    selected_features = list(feature_columns or infer_feature_columns(rows, label_column))
    aligned_rows = align_feature_columns(rows, selected_features)
    attack_labels = to_attack_labels(rows, label_column=label_column, benign_token=benign_token)

    benign_rows = [row for row, is_attack in zip(aligned_rows, attack_labels) if is_attack == 0]
    if len(benign_rows) < resolved_config.sequence_length:
        raise ValueError("Not enough benign rows for selected sequence length.")

    scaler = MinMaxFeatureScaler()
    scaler.fit(benign_rows, selected_features)

    benign_matrix = scaler.transform(benign_rows, selected_features)
    benign_sequences = create_sequences(benign_matrix, resolved_config.sequence_length)

    builder = model_builder or build_default_hybrid_model
    model, callbacks = builder(len(selected_features), resolved_config.sequence_length, resolved_config)
    fit_kwargs = {
        "batch_size": resolved_config.batch_size,
        "epochs": resolved_config.epochs,
        "validation_split": resolved_config.validation_split,
        "verbose": 0,
    }
    if callbacks:
        fit_kwargs["callbacks"] = callbacks

    history = model.fit(benign_sequences, benign_sequences, **fit_kwargs)
    train_errors = reconstruction_errors(model, benign_sequences)
    threshold = compute_threshold(train_errors, resolved_config.threshold_std_factor)

    return HybridExperimentState(
        model=model,
        scaler=scaler,
        feature_columns=selected_features,
        threshold=threshold,
        config=resolved_config,
        training_history=getattr(history, "history", None),
    )


def evaluate_zero_shot_cic2018(
    state: HybridExperimentState,
    cse_cic2018_data: Any,
    *,
    label_column: str = "Label",
    benign_token: str = "BENIGN",
) -> Dict[str, Any]:
    """Evaluate target dataset without any adaptation."""
    return _evaluate_dataset(state, cse_cic2018_data, label_column=label_column, benign_token=benign_token)


def adapt_few_shot_benign_1pct(
    state: HybridExperimentState,
    cse_cic2018_data: Any,
    *,
    benign_ratio: Optional[float] = None,
    adaptation_epochs: int = 3,
    label_column: str = "Label",
    benign_token: str = "BENIGN",
) -> HybridExperimentState:
    """Apply unsupervised adaptation using a small benign subset from target data."""
    ratio = benign_ratio if benign_ratio is not None else state.config.few_shot_benign_ratio
    if ratio <= 0:
        raise ValueError("benign_ratio must be greater than 0.")

    rows = _to_rows(cse_cic2018_data)
    aligned_rows = align_feature_columns(rows, state.feature_columns)
    attack_labels = to_attack_labels(rows, label_column=label_column, benign_token=benign_token)
    benign_rows = [row for row, is_attack in zip(aligned_rows, attack_labels) if is_attack == 0]
    if not benign_rows:
        raise ValueError("No benign rows available for adaptation.")

    minimum_rows = state.config.sequence_length
    sample_size = max(minimum_rows, int(len(benign_rows) * ratio))
    if len(benign_rows) < minimum_rows:
        raise ValueError("Not enough benign rows to build adaptation sequences.")
    if sample_size >= len(benign_rows):
        selected_rows = benign_rows
    else:
        chooser = Random(state.config.random_seed)
        selected_rows = chooser.sample(benign_rows, sample_size)

    adapted_matrix = state.scaler.transform(selected_rows, state.feature_columns)
    adapted_sequences = create_sequences(adapted_matrix, state.config.sequence_length)
    state.model.fit(
        adapted_sequences,
        adapted_sequences,
        batch_size=state.config.batch_size,
        epochs=adaptation_epochs,
        verbose=0,
    )

    adapted_errors = reconstruction_errors(state.model, adapted_sequences)
    adapted_threshold = compute_threshold(adapted_errors, state.config.threshold_std_factor)

    return replace(
        state,
        threshold=adapted_threshold,
        adaptation_steps=state.adaptation_steps + 1,
    )


def evaluate_post_adaptation(
    state: HybridExperimentState,
    cse_cic2018_data: Any,
    *,
    label_column: str = "Label",
    benign_token: str = "BENIGN",
) -> Dict[str, Any]:
    """Evaluate target dataset after few-shot adaptation."""
    return _evaluate_dataset(state, cse_cic2018_data, label_column=label_column, benign_token=benign_token)


def train_and_evaluate_cross_dataset(
    cic2017_data: Any,
    cse_cic2018_data: Any,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    label_column: str = "Label",
    benign_token: str = "BENIGN",
    config: Optional[HybridExperimentConfig] = None,
    model_builder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run full proposal workflow and return summary metrics."""
    state = train_cic2017_normal(
        cic2017_data,
        feature_columns=feature_columns,
        label_column=label_column,
        benign_token=benign_token,
        config=config,
        model_builder=model_builder,
    )
    source_eval = _evaluate_dataset(
        state,
        cic2017_data,
        label_column=label_column,
        benign_token=benign_token,
    )
    zero_shot_eval = evaluate_zero_shot_cic2018(
        state,
        cse_cic2018_data,
        label_column=label_column,
        benign_token=benign_token,
    )
    adapted_state = adapt_few_shot_benign_1pct(
        state,
        cse_cic2018_data,
        label_column=label_column,
        benign_token=benign_token,
    )
    few_shot_eval = evaluate_post_adaptation(
        adapted_state,
        cse_cic2018_data,
        label_column=label_column,
        benign_token=benign_token,
    )

    return {
        "state": adapted_state,
        "source_metrics": source_eval["metrics"],
        "zero_shot_metrics": zero_shot_eval["metrics"],
        "few_shot_metrics": few_shot_eval["metrics"],
        "generalization_gap_zero_shot": compute_generalization_gap(
            source_eval["metrics"],
            zero_shot_eval["metrics"],
        ),
        "generalization_gap_few_shot": compute_generalization_gap(
            source_eval["metrics"],
            few_shot_eval["metrics"],
        ),
    }


def align_feature_columns(
    data: Any,
    feature_columns: Sequence[str],
) -> List[Dict[str, float]]:
    """Align feature columns and fill missing values with 0."""
    rows = _to_rows(data)
    aligned: List[Dict[str, float]] = []
    for row in rows:
        aligned_row: Dict[str, float] = {}
        for column in feature_columns:
            aligned_row[column] = _to_float(row.get(column, 0.0))
        aligned.append(aligned_row)
    return aligned


def infer_feature_columns(rows: Sequence[Mapping[str, Any]], label_column: str = "Label") -> List[str]:
    ignored = {label_column, "Full Label", "Timestamp"}
    columns: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in ignored and key not in columns:
                columns.append(key)
    return columns


def to_attack_labels(
    data: Any,
    *,
    label_column: str = "Label",
    benign_token: str = "BENIGN",
) -> List[int]:
    """Convert labels to binary attack labels: benign=0, attack=1."""
    rows = _to_rows(data)
    labels: List[int] = []
    normalized_benign = str(benign_token).strip().upper()
    benign_aliases = {"BENIGN", "NORMAL", "0", "FALSE", normalized_benign}

    for row in rows:
        raw_label = row.get(label_column, "")
        if isinstance(raw_label, bool):
            labels.append(1 if raw_label else 0)
            continue
        if isinstance(raw_label, (int, float)):
            labels.append(0 if float(raw_label) == 0 else 1)
            continue

        label_text = str(raw_label).strip().upper()
        labels.append(0 if label_text in benign_aliases else 1)
    return labels


def create_sequences(matrix: Sequence[Sequence[float]], sequence_length: int) -> List[List[List[float]]]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1.")
    if len(matrix) < sequence_length:
        raise ValueError("Not enough rows to build sequences.")

    sequences: List[List[List[float]]] = []
    last_index = len(matrix) - sequence_length + 1
    for start_index in range(last_index):
        window = [list(row) for row in matrix[start_index : start_index + sequence_length]]
        sequences.append(window)
    return sequences


def labels_for_sequences(labels: Sequence[int], sequence_length: int) -> List[int]:
    if len(labels) < sequence_length:
        raise ValueError("Not enough labels to map sequences.")
    return list(labels[sequence_length - 1 :])


def reconstruction_errors(model: Any, sequences: Sequence[Sequence[Sequence[float]]]) -> List[float]:
    predictions = model.predict(sequences, verbose=0)
    errors: List[float] = []
    for original_sequence, predicted_sequence in zip(sequences, predictions):
        absolute_sum = 0.0
        count = 0
        for original_row, predicted_row in zip(original_sequence, predicted_sequence):
            for original_value, predicted_value in zip(original_row, predicted_row):
                absolute_sum += abs(float(predicted_value) - float(original_value))
                count += 1
        errors.append(absolute_sum / max(count, 1))
    return errors


def compute_threshold(errors: Sequence[float], std_factor: float = 2.0) -> float:
    if not errors:
        raise ValueError("errors cannot be empty.")
    average_error = mean(errors)
    variance = mean([(error - average_error) ** 2 for error in errors])
    return average_error + std_factor * sqrt(variance)


def apply_threshold(errors: Sequence[float], threshold: float) -> List[int]:
    return [1 if value > threshold else 0 for value in errors]


def compute_binary_metrics(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    *,
    error_scores: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have same length.")
    if not true_labels:
        raise ValueError("Cannot compute metrics on empty labels.")

    true_values = [1 if value else 0 for value in true_labels]
    predicted_values = [1 if value else 0 for value in predicted_labels]

    true_positive = sum(1 for t, p in zip(true_values, predicted_values) if t == 1 and p == 1)
    true_negative = sum(1 for t, p in zip(true_values, predicted_values) if t == 0 and p == 0)
    false_positive = sum(1 for t, p in zip(true_values, predicted_values) if t == 0 and p == 1)
    false_negative = sum(1 for t, p in zip(true_values, predicted_values) if t == 1 and p == 0)

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1_score = _safe_divide(2 * precision * recall, precision + recall)
    accuracy = _safe_divide(true_positive + true_negative, len(true_values))
    fpr = _safe_divide(false_positive, false_positive + true_negative)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "fpr": fpr,
    }
    if error_scores is not None:
        metrics["auc_roc"] = compute_auc_roc(true_values, error_scores)
    return metrics


def compute_auc_roc(true_labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(true_labels) != len(scores):
        raise ValueError("true_labels and scores must have same length.")
    positives = [(score, label) for score, label in zip(scores, true_labels) if label == 1]
    negatives = [(score, label) for score, label in zip(scores, true_labels) if label == 0]
    if not positives or not negatives:
        return 0.0

    sorted_pairs = sorted(zip(scores, true_labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(sorted_pairs):
        tie_end = index
        while tie_end + 1 < len(sorted_pairs) and sorted_pairs[tie_end + 1][0] == sorted_pairs[index][0]:
            tie_end += 1
        average_rank = (index + tie_end + 2) / 2.0
        for tie_index in range(index, tie_end + 1):
            if sorted_pairs[tie_index][1] == 1:
                rank_sum += average_rank
        index = tie_end + 1

    pos_count = float(len(positives))
    neg_count = float(len(negatives))
    return (rank_sum - (pos_count * (pos_count + 1) / 2.0)) / (pos_count * neg_count)


def compute_generalization_gap(
    in_distribution_metrics: Mapping[str, float],
    out_distribution_metrics: Mapping[str, float],
    *,
    metric_name: str = "f1_score",
) -> float:
    in_value = float(in_distribution_metrics.get(metric_name, 0.0))
    out_value = float(out_distribution_metrics.get(metric_name, 0.0))
    return in_value - out_value


def build_default_hybrid_model(
    n_features: int,
    sequence_length: int,
    config: HybridExperimentConfig,
) -> Tuple[Any, List[Any]]:
    """Build TensorFlow CNN-BiLSTM autoencoder model."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for default model building. "
            "Provide a custom model_builder if TensorFlow is unavailable."
        ) from exc

    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Conv1D(64, kernel_size=3, activation="relu", padding="same", input_shape=(sequence_length, n_features)),
            layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
            layers.Dropout(0.1),
            layers.Dense(64, activation="relu"),
            layers.RepeatVector(sequence_length),
            layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.TimeDistributed(layers.Dense(n_features, activation="sigmoid")),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mae",
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        )
    ]
    return model, callbacks


def _evaluate_dataset(
    state: HybridExperimentState,
    dataset: Any,
    *,
    label_column: str,
    benign_token: str,
) -> Dict[str, Any]:
    rows = _to_rows(dataset)
    aligned_rows = align_feature_columns(rows, state.feature_columns)
    transformed_matrix = state.scaler.transform(aligned_rows, state.feature_columns)
    sequences = create_sequences(transformed_matrix, state.config.sequence_length)

    attack_labels = to_attack_labels(rows, label_column=label_column, benign_token=benign_token)
    sequence_labels = labels_for_sequences(attack_labels, state.config.sequence_length)
    errors = reconstruction_errors(state.model, sequences)
    predictions = apply_threshold(errors, state.threshold)
    metrics = compute_binary_metrics(sequence_labels, predictions, error_scores=errors)

    return {
        "metrics": metrics,
        "threshold": state.threshold,
        "errors": errors,
        "predictions": predictions,
        "labels": sequence_labels,
    }


def _to_rows(data: Any) -> List[Dict[str, Any]]:
    if data is None:
        raise ValueError("Input data cannot be None.")

    if isinstance(data, list):
        if not data:
            return []
        if not isinstance(data[0], Mapping):
            raise TypeError("List input must contain mapping rows.")
        return [dict(row) for row in data]

    to_dict = getattr(data, "to_dict", None)
    if callable(to_dict):
        try:
            records = data.to_dict("records")
        except TypeError:
            records = data.to_dict()
        if isinstance(records, list):
            return [dict(row) for row in records]
        if isinstance(records, Mapping):
            keys = list(records.keys())
            row_count = len(records[keys[0]]) if keys else 0
            rows: List[Dict[str, Any]] = []
            for index in range(row_count):
                rows.append({key: records[key][index] for key in keys})
            return rows

    raise TypeError("Unsupported data format. Use list[dict] or pandas DataFrame-like input.")


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
