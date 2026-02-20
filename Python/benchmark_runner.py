"""CLI runner to benchmark hybrid pipeline with standardized metrics output."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import hybrid_pipeline as hp


IGNORED_COLUMNS = {"timestamp", "full label"}


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_code_hash(paths: Sequence[str | Path]) -> str:
    hasher = hashlib.sha256()
    for raw_path in sorted(str(Path(p)) for p in paths):
        path = Path(raw_path)
        hasher.update(raw_path.encode("utf-8"))
        if not path.exists():
            hasher.update(b"<missing>")
            continue
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def list_csv_files(path: str | Path) -> list[Path]:
    target = Path(path)
    if target.is_file() and target.suffix.lower() == ".csv":
        return [target]
    if target.is_dir():
        return sorted(target.rglob("*.csv"))
    return []


def _read_csv_rows(csv_path: Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            rows.append(dict(row))
    return rows


def load_rows_from_path(
    path: str | Path,
    *,
    max_rows_per_file: int | None = None,
    max_total_rows: int | None = None,
) -> list[dict[str, Any]]:
    files = list_csv_files(path)
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {path}")

    rows: list[dict[str, Any]] = []
    for csv_path in files:
        remaining = None
        if max_total_rows is not None:
            remaining = max(0, max_total_rows - len(rows))
            if remaining == 0:
                break
        file_limit = max_rows_per_file
        if remaining is not None:
            file_limit = remaining if file_limit is None else min(file_limit, remaining)
        rows.extend(_read_csv_rows(csv_path, max_rows=file_limit))

        if max_total_rows is not None and len(rows) >= max_total_rows:
            rows = rows[:max_total_rows]
            break
    return rows


def infer_shared_feature_columns(
    cic_rows: Sequence[Mapping[str, Any]],
    cse_rows: Sequence[Mapping[str, Any]],
    *,
    label_column: str = "Label",
) -> list[str]:
    if not cic_rows or not cse_rows:
        raise ValueError("Cannot infer features from empty rows.")

    label_lower = label_column.strip().lower()
    cic_cols = {k for row in cic_rows for k in row.keys()}
    cse_cols = {k for row in cse_rows for k in row.keys()}
    shared = cic_cols.intersection(cse_cols)

    features = []
    for col in sorted(shared):
        lower = col.strip().lower()
        if lower == label_lower or lower in IGNORED_COLUMNS:
            continue
        features.append(col)
    if not features:
        raise ValueError("No shared feature columns found.")
    return features


def _normalize_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "accuracy": _to_float(metrics.get("accuracy")),
        "precision": _to_float(metrics.get("precision")),
        "recall": _to_float(metrics.get("recall")),
        "f1": _to_float(metrics.get("f1_score", metrics.get("f1"))),
        "roc_auc": _to_float(metrics.get("auc_roc", metrics.get("roc_auc"))),
        "fpr": _to_float(metrics.get("fpr")),
    }


def build_standardized_record(
    *,
    dataset: str,
    mode: str,
    threshold_method: str,
    threshold: float,
    seed: int,
    metrics: Mapping[str, Any],
    model_path: str,
    code_hash: str,
    config_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "mode": mode,
        "threshold_method": threshold_method,
        "threshold": float(threshold),
        "seed": int(seed),
        "model_path": model_path,
        "code_hash": code_hash,
        "config_snapshot": dict(config_snapshot),
    }
    record.update(_normalize_metrics(metrics))
    return record


def _load_feature_columns(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict) and "features" in payload and isinstance(payload["features"], list):
        return [str(x) for x in payload["features"]]
    raise ValueError("Unsupported feature columns payload. Use list or {'features': [...]} format.")


def _save_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _save_state(state: hp.HybridExperimentState, state_dir: Path) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    model_path = state_dir / "model.keras"
    if not hasattr(state.model, "save"):
        raise RuntimeError("State model does not support save().")
    state.model.save(model_path)

    payload = {
        "feature_columns": state.feature_columns,
        "threshold": state.threshold,
        "adaptation_steps": state.adaptation_steps,
        "config": asdict(state.config),
        "training_history": state.training_history,
        "scaler": {
            "minimums": state.scaler.minimums,
            "maximums": state.scaler.maximums,
        },
    }
    _save_json(state_dir / "state.json", payload)
    return model_path


def _load_state(state_dir: Path) -> hp.HybridExperimentState:
    payload = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    model_path = state_dir / "model.keras"

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to load saved state model.") from exc

    model = tf.keras.models.load_model(model_path)
    scaler = hp.MinMaxFeatureScaler()
    scaler.minimums = {k: float(v) for k, v in payload["scaler"]["minimums"].items()}
    scaler.maximums = {k: float(v) for k, v in payload["scaler"]["maximums"].items()}
    config = hp.HybridExperimentConfig(**payload["config"])
    return hp.HybridExperimentState(
        model=model,
        scaler=scaler,
        feature_columns=list(payload["feature_columns"]),
        threshold=float(payload["threshold"]),
        config=config,
        adaptation_steps=int(payload.get("adaptation_steps", 0)),
        training_history=payload.get("training_history"),
    )


def _metrics_output_name(dataset: str, mode: str) -> str:
    safe_dataset = dataset.lower().replace("-", "_")
    return f"{safe_dataset}_{mode}_standardized.json"


def _save_standardized_record(output_dir: Path, dataset: str, mode: str, record: Mapping[str, Any]) -> Path:
    out_path = output_dir / _metrics_output_name(dataset, mode)
    _save_json(out_path, record)
    return out_path


def cmd_train(args: argparse.Namespace) -> None:
    cic_rows = load_rows_from_path(
        args.cic_data,
        max_rows_per_file=args.max_rows_per_file,
        max_total_rows=args.max_total_rows,
    )
    cse_rows = load_rows_from_path(
        args.cse_data,
        max_rows_per_file=args.max_rows_per_file,
        max_total_rows=args.max_total_rows,
    )

    feature_columns = _load_feature_columns(args.feature_columns_json)
    if feature_columns is None:
        feature_columns = infer_shared_feature_columns(cic_rows, cse_rows, label_column=args.label_column)

    config = hp.HybridExperimentConfig(
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        threshold_std_factor=args.threshold_std_factor,
        few_shot_benign_ratio=args.few_shot_ratio,
        random_seed=args.seed,
    )
    state = hp.train_cic2017_normal(
        cic_rows,
        feature_columns=feature_columns,
        label_column=args.label_column,
        benign_token=args.benign_token,
        config=config,
    )
    model_path = _save_state(state, Path(args.state_dir))
    logging.info("Training complete. State saved to %s", args.state_dir)
    logging.info("Model path: %s", model_path)


def cmd_eval_zero_shot(args: argparse.Namespace) -> None:
    state = _load_state(Path(args.state_dir))
    cse_rows = load_rows_from_path(
        args.cse_data,
        max_rows_per_file=args.max_rows_per_file,
        max_total_rows=args.max_total_rows,
    )
    zero = hp.evaluate_zero_shot_cic2018(
        state,
        cse_rows,
        label_column=args.label_column,
        benign_token=args.benign_token,
    )
    code_hash = compute_code_hash([Path(__file__), Path(__file__).parent / "hybrid_pipeline.py"])
    record = build_standardized_record(
        dataset="CSE-CIC-IDS2018",
        mode="zero_shot",
        threshold_method="mean_plus_k_std",
        threshold=state.threshold,
        seed=state.config.random_seed,
        metrics=zero["metrics"],
        model_path=str(Path(args.state_dir) / "model.keras"),
        code_hash=code_hash,
        config_snapshot=asdict(state.config),
    )
    out_dir = Path(args.output_dir)
    out_path = _save_standardized_record(out_dir, "CSE-CIC-IDS2018", "zero_shot", record)
    logging.info("Zero-shot metrics saved: %s", out_path)


def cmd_eval_few_shot(args: argparse.Namespace) -> None:
    state = _load_state(Path(args.state_dir))
    cse_rows = load_rows_from_path(
        args.cse_data,
        max_rows_per_file=args.max_rows_per_file,
        max_total_rows=args.max_total_rows,
    )
    adapted = hp.adapt_few_shot_benign_1pct(
        state,
        cse_rows,
        benign_ratio=args.few_shot_ratio,
        adaptation_epochs=args.few_shot_epochs,
        label_column=args.label_column,
        benign_token=args.benign_token,
    )
    few = hp.evaluate_post_adaptation(
        adapted,
        cse_rows,
        label_column=args.label_column,
        benign_token=args.benign_token,
    )
    code_hash = compute_code_hash([Path(__file__), Path(__file__).parent / "hybrid_pipeline.py"])
    record = build_standardized_record(
        dataset="CSE-CIC-IDS2018",
        mode="few_shot",
        threshold_method="mean_plus_k_std",
        threshold=adapted.threshold,
        seed=adapted.config.random_seed,
        metrics=few["metrics"],
        model_path=str(Path(args.state_dir) / "model.keras"),
        code_hash=code_hash,
        config_snapshot=asdict(adapted.config),
    )
    out_dir = Path(args.output_dir)
    out_path = _save_standardized_record(out_dir, "CSE-CIC-IDS2018", "few_shot", record)
    logging.info("Few-shot metrics saved: %s", out_path)


def cmd_run_all(args: argparse.Namespace) -> None:
    cic_rows = load_rows_from_path(
        args.cic_data,
        max_rows_per_file=args.max_rows_per_file,
        max_total_rows=args.max_total_rows,
    )
    cse_rows = load_rows_from_path(
        args.cse_data,
        max_rows_per_file=args.max_rows_per_file,
        max_total_rows=args.max_total_rows,
    )
    feature_columns = _load_feature_columns(args.feature_columns_json)
    if feature_columns is None:
        feature_columns = infer_shared_feature_columns(cic_rows, cse_rows, label_column=args.label_column)

    config = hp.HybridExperimentConfig(
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        threshold_std_factor=args.threshold_std_factor,
        few_shot_benign_ratio=args.few_shot_ratio,
        random_seed=args.seed,
    )

    state = hp.train_cic2017_normal(
        cic_rows,
        feature_columns=feature_columns,
        label_column=args.label_column,
        benign_token=args.benign_token,
        config=config,
    )
    source_eval = hp.evaluate_zero_shot_cic2018(
        state,
        cic_rows,
        label_column=args.label_column,
        benign_token=args.benign_token,
    )
    zero_eval = hp.evaluate_zero_shot_cic2018(
        state,
        cse_rows,
        label_column=args.label_column,
        benign_token=args.benign_token,
    )
    adapted_state = hp.adapt_few_shot_benign_1pct(
        state,
        cse_rows,
        benign_ratio=args.few_shot_ratio,
        adaptation_epochs=args.few_shot_epochs,
        label_column=args.label_column,
        benign_token=args.benign_token,
    )
    few_eval = hp.evaluate_post_adaptation(
        adapted_state,
        cse_rows,
        label_column=args.label_column,
        benign_token=args.benign_token,
    )

    output_dir = Path(args.output_dir)
    model_path = _save_state(adapted_state, output_dir / "state")
    code_hash = compute_code_hash([Path(__file__), Path(__file__).parent / "hybrid_pipeline.py"])

    cic_record = build_standardized_record(
        dataset="CIC-IDS2017",
        mode="in_domain",
        threshold_method="mean_plus_k_std",
        threshold=state.threshold,
        seed=config.random_seed,
        metrics=source_eval["metrics"],
        model_path=str(model_path),
        code_hash=code_hash,
        config_snapshot=asdict(config),
    )
    cse_zero_record = build_standardized_record(
        dataset="CSE-CIC-IDS2018",
        mode="zero_shot",
        threshold_method="mean_plus_k_std",
        threshold=state.threshold,
        seed=config.random_seed,
        metrics=zero_eval["metrics"],
        model_path=str(model_path),
        code_hash=code_hash,
        config_snapshot=asdict(config),
    )
    cse_few_record = build_standardized_record(
        dataset="CSE-CIC-IDS2018",
        mode="few_shot",
        threshold_method="mean_plus_k_std",
        threshold=adapted_state.threshold,
        seed=config.random_seed,
        metrics=few_eval["metrics"],
        model_path=str(model_path),
        code_hash=code_hash,
        config_snapshot=asdict(config),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_standardized_record(output_dir, "CIC-IDS2017", "in_domain", cic_record)
    _save_standardized_record(output_dir, "CSE-CIC-IDS2018", "zero_shot", cse_zero_record)
    _save_standardized_record(output_dir, "CSE-CIC-IDS2018", "few_shot", cse_few_record)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": config.random_seed,
        "feature_count": len(feature_columns),
        "cic_record": cic_record,
        "cse_zero_shot_record": cse_zero_record,
        "cse_few_shot_record": cse_few_record,
        "generalization_gap_zero_shot_f1": cic_record["f1"] - cse_zero_record["f1"],
        "generalization_gap_few_shot_f1": cic_record["f1"] - cse_few_record["f1"],
    }
    _save_json(output_dir / "summary.json", summary)
    logging.info("Run-all complete. Results saved in %s", output_dir)


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--label-column", default="Label")
    parser.add_argument("--benign-token", default="BENIGN")
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--max-total-rows", type=int, default=None)


def _add_train_like_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--threshold-std-factor", type=float, default=2.0)
    parser.add_argument("--few-shot-ratio", type=float, default=0.01)
    parser.add_argument("--few-shot-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--feature-columns-json", default=None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--cic-data", required=True)
    train.add_argument("--cse-data", required=True)
    train.add_argument("--state-dir", required=True)
    _add_data_args(train)
    _add_train_like_args(train)
    train.set_defaults(func=cmd_train)

    eval_zero = subparsers.add_parser("eval-zero-shot")
    eval_zero.add_argument("--state-dir", required=True)
    eval_zero.add_argument("--cse-data", required=True)
    eval_zero.add_argument("--output-dir", required=True)
    _add_data_args(eval_zero)
    eval_zero.set_defaults(func=cmd_eval_zero_shot)

    eval_few = subparsers.add_parser("eval-few-shot")
    eval_few.add_argument("--state-dir", required=True)
    eval_few.add_argument("--cse-data", required=True)
    eval_few.add_argument("--output-dir", required=True)
    eval_few.add_argument("--few-shot-ratio", type=float, default=0.01)
    eval_few.add_argument("--few-shot-epochs", type=int, default=3)
    _add_data_args(eval_few)
    eval_few.set_defaults(func=cmd_eval_few_shot)

    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--cic-data", required=True)
    run_all.add_argument("--cse-data", required=True)
    run_all.add_argument("--output-dir", required=True)
    _add_data_args(run_all)
    _add_train_like_args(run_all)
    run_all.set_defaults(func=cmd_run_all)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s | %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
