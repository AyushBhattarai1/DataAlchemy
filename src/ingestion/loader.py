"""Data Ingestion Loader Module for DataAlchemy.

Provides a standardized, robust interface for loading tabular data files
(CSV, Excel, JSON, Parquet) into an immutable Dataset container with validation
and structured application-level error handling.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd


# -----------------------------------------------------------------------------
# Custom Domain Exceptions
# -----------------------------------------------------------------------------

class IngestionError(Exception):
    """Base exception for all ingestion engine failures."""
    pass


class UnsupportedFileFormatError(IngestionError):
    """Raised when an unsupported file extension is provided."""
    pass


class DatasetNotFoundError(IngestionError, FileNotFoundError):
    """Raised when the specified dataset file does not exist."""
    pass


class EmptyDatasetError(IngestionError):
    """Raised when a dataset file is empty (0 bytes) or has 0 rows / 0 columns."""
    pass


class CorruptDatasetError(IngestionError):
    """Raised when a dataset file cannot be parsed or is corrupted."""
    pass


# -----------------------------------------------------------------------------
# Standardized Dataset Representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset:
    """Standardized representation of an ingested dataset.

    Attributes:
        dataframe: The underlying pandas DataFrame containing the raw data.
        filename: Name of the source file.
        file_type: Detected file format (e.g., 'csv', 'xlsx', 'json', 'parquet').
        row_count: Number of rows in the dataset.
        column_count: Number of columns in the dataset.
        metadata: Additional metadata including file size, columns, and timestamp.
    """
    dataframe: pd.DataFrame
    filename: str
    file_type: str
    row_count: int
    column_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def columns(self) -> List[str]:
        """Return the list of column names."""
        return list(self.dataframe.columns)

    def __repr__(self) -> str:
        return (
            f"Dataset(filename='{self.filename}', "
            f"file_type='{self.file_type}', "
            f"rows={self.row_count}, "
            f"columns={self.column_count})"
        )


# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------

class DataLoader:
    """Handles discovery, validation, and loading of tabular data files."""

    SUPPORTED_FORMATS: Dict[str, str] = {
        ".csv": "csv",
        ".xlsx": "xlsx",
        ".xls": "xls",
        ".json": "json",
        ".parquet": "parquet",
    }

    def __init__(self) -> None:
        pass

    @classmethod
    def get_supported_extensions(cls) -> Set[str]:
        """Return the set of supported file extensions."""
        return set(cls.SUPPORTED_FORMATS.keys())

    def validate_path(self, path: Union[str, Path]) -> Path:
        """Validate that the file path exists and is a regular file.

        Args:
            path: Path to the dataset file.

        Returns:
            Resolved Path object.

        Raises:
            DatasetNotFoundError: If the path does not exist or is a directory.
        """
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise DatasetNotFoundError(f"File does not exist: {path}")
        if not resolved.is_file():
            raise DatasetNotFoundError(f"Path is not a regular file: {path}")
        return resolved

    def detect_format(self, path: Path) -> str:
        """Detect and validate the file format from its extension.

        Args:
            path: Resolved Path object.

        Returns:
            Format identifier string ('csv', 'xlsx', 'xls', 'json', 'parquet').

        Raises:
            UnsupportedFileFormatError: If the extension is not supported.
        """
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            supported_str = ", ".join(sorted(self.SUPPORTED_FORMATS.keys()))
            raise UnsupportedFileFormatError(
                f"Unsupported file format: {ext if ext else '(no extension)'}\n\n"
                f"Supported formats:\n{supported_str}"
            )
        return self.SUPPORTED_FORMATS[ext]

    def _load_raw_dataframe(self, path: Path, file_type: str) -> pd.DataFrame:
        """Parse raw file into a pandas DataFrame based on file type.

        Args:
            path: Resolved path to the file.
            file_type: Canonical file format identifier.

        Returns:
            Parsed pandas DataFrame.

        Raises:
            EmptyDatasetError: If parsing indicates an empty dataset.
            CorruptDatasetError: If file content is corrupt or unparseable.
        """
        try:
            if file_type == "csv":
                return pd.read_csv(path)

            elif file_type in ("xlsx", "xls"):
                return pd.read_excel(path)

            elif file_type == "json":
                return pd.read_json(path)

            elif file_type == "parquet":
                return pd.read_parquet(path)

            else:
                raise UnsupportedFileFormatError(f"Unsupported file format: {file_type}")

        except pd.errors.EmptyDataError as err:
            raise EmptyDatasetError(f"Dataset contains no data: {path}") from err
        except pd.errors.ParserError as err:
            raise CorruptDatasetError(f"Failed to parse CSV file: {path}. Cause: {err}") from err
        except UnicodeDecodeError as err:
            raise CorruptDatasetError(f"Encoding error reading file: {path}. Cause: {err}") from err
        except (ValueError, KeyError) as err:
            # Typically JSON parsing or schema mismatch errors
            raise CorruptDatasetError(f"Malformed data file: {path}. Cause: {err}") from err
        except Exception as err:
            # Handle PyArrow / Excel / Corrupt binary errors gracefully
            err_name = type(err).__name__
            if "Arrow" in err_name or "BadZipFile" in err_name or "InvalidFile" in err_name:
                raise CorruptDatasetError(f"Corrupt {file_type.upper()} file: {path}. Cause: {err}") from err
            raise CorruptDatasetError(f"Failed to read {file_type.upper()} file: {path}. Cause: {err}") from err

    def load(self, file_path: Union[str, Path]) -> Dataset:
        """Validate and load a tabular dataset into a standardized Dataset container.

        Args:
            file_path: Path or string pointing to the tabular data file.

        Returns:
            Dataset: Standardized container with the loaded DataFrame and metadata.

        Raises:
            DatasetNotFoundError: If file does not exist.
            UnsupportedFileFormatError: If file format is not supported.
            EmptyDatasetError: If file is 0 bytes, or DataFrame has 0 rows or 0 columns.
            CorruptDatasetError: If file is corrupt or cannot be parsed.
        """
        # 1. Path validation
        resolved_path = self.validate_path(file_path)

        # 2. Check extension & format detection
        file_type = self.detect_format(resolved_path)

        # 3. Check for 0-byte file
        file_size = resolved_path.stat().st_size
        if file_size == 0:
            raise EmptyDatasetError(f"File is empty (0 bytes): {file_path}")

        # 4. Load raw DataFrame
        df = self._load_raw_dataframe(resolved_path, file_type)

        # 5. Sanity checks
        if not isinstance(df, pd.DataFrame):
            raise CorruptDatasetError(f"Loaded object is not a valid tabular DataFrame: {type(df)}")

        row_count, col_count = df.shape
        if row_count == 0:
            raise EmptyDatasetError(f"Dataset contains zero rows: {file_path}")
        if col_count == 0:
            raise EmptyDatasetError(f"Dataset contains zero columns: {file_path}")

        # 6. Build metadata
        try:
            memory_bytes = int(df.memory_usage(deep=True).sum())
        except Exception:
            memory_bytes = 0

        metadata: Dict[str, Any] = {
            "source_path": str(resolved_path),
            "file_size_bytes": file_size,
            "memory_usage_bytes": memory_bytes,
            "columns": [str(c) for c in df.columns],
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

        # 7. Construct standardized Dataset
        return Dataset(
            dataframe=df,
            filename=resolved_path.name,
            file_type=file_type,
            row_count=row_count,
            column_count=col_count,
            metadata=metadata,
        )


def load_dataset(file_path: Union[str, Path]) -> Dataset:
    """Convenience functional API to validate and load a dataset.

    Args:
        file_path: Path or string pointing to the dataset file.

    Returns:
        Dataset: Standardized container with raw DataFrame and metadata.
    """
    return DataLoader().load(file_path)
