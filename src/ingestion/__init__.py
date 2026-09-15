"""DataAlchemy Ingestion Package.

Provides data loading and semantic schema detection for tabular datasets.
"""

from src.ingestion.loader import (
    CorruptDatasetError,
    DataLoader,
    Dataset,
    DatasetNotFoundError,
    EmptyDatasetError,
    IngestionError,
    UnsupportedFileFormatError,
    load_dataset,
)
from src.ingestion.schema import (
    DatasetSchema,
    SchemaDetector,
    SemanticType,
    detect_schema,
)

__all__ = [
    # Loader & Containers
    "Dataset",
    "DataLoader",
    "load_dataset",
    # Schema & Types
    "SemanticType",
    "DatasetSchema",
    "SchemaDetector",
    "detect_schema",
    # Exceptions
    "IngestionError",
    "UnsupportedFileFormatError",
    "DatasetNotFoundError",
    "EmptyDatasetError",
    "CorruptDatasetError",
]
