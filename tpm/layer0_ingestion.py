"""
Layer 0: Data Ingestion
-----------------------
Loads raw data from various file formats into a standardized internal
representation: a pandas DataFrame with numeric columns and a time/row index.

Domain-agnostic: handles CSV, TSV, DAT (whitespace-delimited), and RData files.
Auto-detects delimiters, headers, and timestamp columns.

For multi-run datasets (like TEP's te_process.csv), detects the run/group
column, strips label/metadata columns, and returns a dict of per-run
DataFrames so each run is an independent time series.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


# Columns that are known ground-truth labels — never fed to analysis.
# Keyed by dataset filename pattern for auto-detection.
KNOWN_LABEL_SCHEMAS = {
    "te_process": {
        "label_cols": {"faultNumber", "source", "fault_status"},
        "run_col": "simulationRun",
        "sample_col": "sample",
    },
}


class DataIngestion:
    """Generic data loader that normalizes any tabular file into a clean DataFrame."""

    SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".dat", ".txt", ".rdata", ".rda", ".xlsx", ".xls"}

    def __init__(self):
        self.metadata: Dict[str, Any] = {}

    def load(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load a file and return a clean numeric DataFrame.

        Parameters
        ----------
        filepath : str
            Path to the data file.
        **kwargs : dict
            Optional overrides:
            - delimiter: str — force a specific delimiter
            - timestamp_col: str or int — which column is the timestamp
            - header: bool — whether the file has a header row
            - sheet_name: str or int — for Excel files
            - label_cols: set of str — columns to strip (ground-truth labels)
            - run_col: str — column that identifies independent runs/groups
            - sample_col: str — column to use as within-run index

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with numeric columns and a meaningful index.
        """
        ext = os.path.splitext(filepath)[1].lower()

        if ext in (".csv", ".tsv", ".dat", ".txt"):
            df = self._load_delimited(filepath, **kwargs)
        elif ext in (".rdata", ".rda"):
            df = self._load_rdata(filepath, **kwargs)
        elif ext in (".xlsx", ".xls"):
            df = self._load_excel(filepath, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        df = self._normalize(df, **kwargs)
        self._record_metadata(df, filepath)
        return df

    def load_multi_run(self, filepath: str, **kwargs) -> Dict[str, Any]:
        """
        Load a multi-run dataset. Auto-detects or accepts explicit config
        for which columns are labels, which column groups the runs, and
        which column is the within-run time index.

        Returns
        -------
        dict with keys:
            - 'runs': Dict[int, pd.DataFrame] — run_id → value-only DataFrame
            - 'labels': pd.DataFrame — ground-truth columns kept aside
            - 'columns': List[str] — the analysis columns
            - 'run_ids': List[int] — sorted list of run IDs
            - 'metadata': dict — file-level metadata
        """
        # ── 1. Raw load (no normalization yet) ──────────────────────
        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".csv", ".tsv", ".dat", ".txt"):
            raw = self._load_delimited(filepath, **kwargs)
        elif ext in (".xlsx", ".xls"):
            raw = self._load_excel(filepath, **kwargs)
        else:
            raw = self._load_delimited(filepath, **kwargs)

        # ── 2. Detect or use provided schema ────────────────────────
        schema = self._detect_schema(filepath, raw, **kwargs)
        label_cols = schema["label_cols"]
        run_col = schema["run_col"]
        sample_col = schema["sample_col"]

        # ── 3. Separate labels from analysis data ────────────────────
        label_cols_present = [c for c in raw.columns if c in label_cols]
        meta_cols = label_cols | {run_col, sample_col} - {None}
        columns = [c for c in raw.columns if c not in meta_cols]

        labels = raw[label_cols_present].copy() if label_cols_present else pd.DataFrame()
        if run_col and run_col in raw.columns:
            labels[run_col] = raw[run_col]
        if sample_col and sample_col in raw.columns:
            labels[sample_col] = raw[sample_col]

        # ── 4. Split into per-run DataFrames ────────────────────────
        runs: Dict[int, pd.DataFrame] = {}

        if run_col and run_col in raw.columns:
            grouped = raw.groupby(run_col)
            for run_id, group in grouped:
                run_df = group[columns].copy()

                # Set within-run index
                if sample_col and sample_col in group.columns:
                    run_df.index = group[sample_col].values.astype(int)
                    run_df.index.name = "sample"
                else:
                    run_df.index = pd.RangeIndex(len(run_df), name="sample")

                run_df = run_df.sort_index()

                # Convert to numeric
                for col in run_df.columns:
                    run_df[col] = pd.to_numeric(run_df[col], errors="coerce")

                runs[int(run_id)] = run_df
        else:
            # Single run — whole file
            run_df = raw[columns].copy()
            for col in run_df.columns:
                run_df[col] = pd.to_numeric(run_df[col], errors="coerce")
            run_df.index = pd.RangeIndex(len(run_df), name="sample")
            runs[1] = run_df

        # ── 5. Record metadata ──────────────────────────────────────
        self.metadata = {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "total_rows": len(raw),
            "n_runs": len(runs),
            "columns": columns,
            "n_columns": len(columns),
            "label_columns": label_cols_present,
            "run_col": run_col,
            "sample_col": sample_col,
            "run_sizes": {rid: len(df) for rid, df in runs.items()},
        }

        return {
            "runs": runs,
            "labels": labels,
            "columns": columns,
            "run_ids": sorted(runs.keys()),
            "metadata": self.metadata,
        }

    def load_multiple(self, filepaths: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """Load multiple files and return a dict of {filename: DataFrame}."""
        results = {}
        for fp in filepaths:
            name = os.path.splitext(os.path.basename(fp))[0]
            results[name] = self.load(fp, **kwargs)
        return results

    # ── Schema detection ────────────────────────────────────────────────

    def _detect_schema(self, filepath: str, df: pd.DataFrame,
                       **kwargs) -> Dict[str, Any]:
        """
        Detect or accept explicit label/run/sample column configuration.

        Priority:
        1. Explicit kwargs (label_cols, run_col, sample_col)
        2. Known dataset patterns (filename matching)
        3. Auto-detection heuristics
        """
        # Start with defaults
        schema = {
            "label_cols": set(),
            "run_col": None,
            "sample_col": None,
        }

        # Priority 1: Explicit overrides
        if "label_cols" in kwargs:
            schema["label_cols"] = set(kwargs["label_cols"])
        if "run_col" in kwargs:
            schema["run_col"] = kwargs["run_col"]
        if "sample_col" in kwargs:
            schema["sample_col"] = kwargs["sample_col"]

        # If all specified, return early
        if schema["label_cols"] or schema["run_col"]:
            return schema

        # Priority 2: Known dataset patterns
        basename = os.path.splitext(os.path.basename(filepath))[0].lower()
        for pattern, known_schema in KNOWN_LABEL_SCHEMAS.items():
            if pattern in basename:
                schema["label_cols"] = known_schema["label_cols"]
                schema["run_col"] = known_schema["run_col"]
                schema["sample_col"] = known_schema["sample_col"]
                return schema

        # Priority 3: Auto-detection heuristics
        schema["label_cols"] = self._detect_label_columns(df)
        schema["run_col"] = self._detect_run_column(df)
        schema["sample_col"] = self._detect_sample_column(df)

        return schema

    def _detect_label_columns(self, df: pd.DataFrame) -> set:
        """
        Heuristic: columns that look like ground-truth labels.
        - String/categorical columns with few unique values
        - Column names containing 'label', 'class', 'fault', 'target', 'status'
        """
        labels = set()
        label_hints = {"label", "class", "fault", "target", "status",
                       "category", "group", "type", "source", "split"}

        for col in df.columns:
            name_lower = str(col).lower()

            # Name-based detection
            if any(hint in name_lower for hint in label_hints):
                # But only if it has few unique values (label-like)
                n_unique = df[col].nunique()
                if n_unique <= 50 or df[col].dtype == object:
                    labels.add(col)
                    continue

            # String columns with few unique values
            if df[col].dtype == object:
                n_unique = df[col].nunique()
                if n_unique <= 20:
                    labels.add(col)

        return labels

    def _detect_run_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Heuristic: a column that groups rows into independent runs.
        Look for 'run', 'simulation', 'batch', 'trial', 'experiment', 'session'.
        """
        run_hints = {"run", "simulation", "batch", "trial", "experiment",
                     "session", "episode", "group_id"}

        for col in df.columns:
            name_lower = str(col).lower()
            if any(hint in name_lower for hint in run_hints):
                # Must have multiple groups but not too many (not continuous)
                n_unique = df[col].nunique()
                if 2 <= n_unique <= len(df) / 5:
                    return col
        return None

    def _detect_sample_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Heuristic: a column that serves as within-run time index.
        Look for 'sample', 'step', 'tick', 'time_step', 'index'.
        """
        sample_hints = {"sample", "step", "tick", "time_step", "timestep",
                        "frame", "observation"}

        for col in df.columns:
            name_lower = str(col).lower()
            if any(hint in name_lower for hint in sample_hints):
                # Should be numeric and monotonically increasing within groups
                if pd.api.types.is_numeric_dtype(df[col]):
                    return col
        return None

    # ── Format-specific loaders ──────────────────────────────────────────

    def _load_delimited(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load CSV/TSV/DAT files with auto-detected delimiter."""
        delimiter = kwargs.get("delimiter", None)
        header = kwargs.get("header", "infer")

        if delimiter is None:
            delimiter = self._detect_delimiter(filepath)

        # Try loading with header first
        if header == "infer":
            df = pd.read_csv(filepath, delimiter=delimiter, header=0, engine="python")
            # If first row looks numeric, it's probably not a header
            if self._first_row_is_numeric(df):
                df = pd.read_csv(filepath, delimiter=delimiter, header=None, engine="python")
                df.columns = [f"col_{i}" for i in range(len(df.columns))]
        elif header:
            df = pd.read_csv(filepath, delimiter=delimiter, header=0, engine="python")
        else:
            df = pd.read_csv(filepath, delimiter=delimiter, header=None, engine="python")
            df.columns = [f"col_{i}" for i in range(len(df.columns))]

        return df

    def _load_rdata(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load R .RData files using pyreadr."""
        try:
            import pyreadr
        except ImportError:
            raise ImportError(
                "pyreadr is required to read .RData files. "
                "Install it with: pip install pyreadr"
            )

        result = pyreadr.read_r(filepath)
        # RData files can contain multiple objects — take the first or specified
        key = kwargs.get("rdata_key", None)
        if key and key in result:
            df = result[key]
        else:
            # Take the first (and usually only) DataFrame
            first_key = list(result.keys())[0]
            df = result[first_key]
            if len(result) > 1:
                self.metadata["rdata_keys"] = list(result.keys())
                self.metadata["rdata_selected"] = first_key

        return df

    def _load_excel(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load Excel files."""
        sheet_name = kwargs.get("sheet_name", 0)
        return pd.read_excel(filepath, sheet_name=sheet_name)

    # ── Normalization ────────────────────────────────────────────────────

    def _normalize(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Normalize a raw DataFrame:
        1. Detect and set timestamp index (or create synthetic one)
        2. Convert columns to numeric where possible
        3. Drop fully empty columns
        4. Standardize column names
        """
        df = df.copy()

        # Step 1: Handle timestamp/index
        timestamp_col = kwargs.get("timestamp_col", None)
        if timestamp_col is not None:
            df = self._set_timestamp_index(df, timestamp_col)
        else:
            # Auto-detect timestamp column
            ts_col = self._detect_timestamp_column(df)
            if ts_col is not None:
                df = self._set_timestamp_index(df, ts_col)
            else:
                # Create synthetic integer index
                df.index = pd.RangeIndex(len(df), name="sample_index")

        # Step 2: Convert to numeric, coercing errors to NaN
        for col in df.columns:
            if df[col].dtype == object:
                converted = pd.to_numeric(df[col], errors="coerce")
                # Only convert if we didn't lose too much data
                non_null_before = df[col].notna().sum()
                non_null_after = converted.notna().sum()
                if non_null_after >= 0.5 * non_null_before:
                    df[col] = converted

        # Step 3: Keep only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Step 4: Drop fully empty columns
        numeric_df = numeric_df.dropna(axis=1, how="all")

        # Step 5: Clean column names (strip whitespace)
        numeric_df.columns = [str(c).strip() for c in numeric_df.columns]

        return numeric_df

    def _set_timestamp_index(self, df: pd.DataFrame, col) -> pd.DataFrame:
        """Set a column as the datetime index."""
        if isinstance(col, int):
            col = df.columns[col]

        try:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            df.index.name = "timestamp"
        except (ValueError, TypeError):
            # If it can't be parsed as datetime, try as numeric index
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.set_index(col)
            df.index.name = "time_index"

        return df

    # ── Auto-detection helpers ───────────────────────────────────────────

    def _detect_delimiter(self, filepath: str) -> str:
        """Sniff the delimiter from the first few lines."""
        with open(filepath, "r", errors="replace") as f:
            sample = ""
            for i, line in enumerate(f):
                sample += line
                if i >= 10:
                    break

        # Count occurrences of common delimiters
        counts = {
            ",": sample.count(","),
            "\t": sample.count("\t"),
            ";": sample.count(";"),
            " ": 0,  # handled separately for whitespace
        }

        # Check for consistent whitespace-delimited (multiple spaces)
        import re
        ws_matches = re.findall(r"  +", sample)
        counts[r"\s+"] = len(ws_matches)

        # Pick the most common delimiter
        best = max(counts, key=counts.get)
        if best == r"\s+" and counts[best] > 0:
            return r"\s+"
        elif counts[best] == 0:
            return r"\s+"  # fallback to whitespace
        return best

    def _detect_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Try to find a column that looks like a timestamp."""
        for col in df.columns:
            if df[col].dtype == object:
                # Try parsing a sample as datetime
                sample = df[col].dropna().head(20)
                try:
                    parsed = pd.to_datetime(sample)
                    if parsed.notna().sum() >= 0.8 * len(sample):
                        return col
                except (ValueError, TypeError):
                    continue

            # Check column name hints
            name_lower = str(col).lower()
            if any(hint in name_lower for hint in ["time", "date", "timestamp", "datetime"]):
                return col

        return None

    def _first_row_is_numeric(self, df: pd.DataFrame) -> bool:
        """Check if the column names (first row if header=0) look numeric."""
        numeric_count = 0
        for col in df.columns:
            try:
                float(str(col))
                numeric_count += 1
            except ValueError:
                pass
        return numeric_count > len(df.columns) * 0.5

    # ── Metadata ─────────────────────────────────────────────────────────

    def _record_metadata(self, df: pd.DataFrame, filepath: str):
        """Record metadata about the loaded dataset."""
        self.metadata.update({
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "index_type": type(df.index).__name__,
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
        })

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata from the last load operation."""
        return self.metadata.copy()

    def summary(self, df: pd.DataFrame) -> str:
        """Return a human-readable summary of the loaded data."""
        lines = [
            f"Dataset: {self.metadata.get('filename', 'unknown')}",
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"Index: {self.metadata.get('index_type', 'unknown')}",
            f"Memory: {self.metadata.get('memory_mb', 0):.2f} MB",
            f"Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}",
        ]
        return "\n".join(lines)