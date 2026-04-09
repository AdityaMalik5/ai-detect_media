"""Shared utilities used across the deepfake detection pipeline and server."""

import os

# Allowed image extensions for upload validation
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
MAX_FILE_SIZE_MB = 10
INPUT_SIZE = 128


def get_filename_only(file_path: str) -> str:
    """Return the filename without extension from a full path."""
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split(".")[0]
    return filename_only


def allowed_file(filename: str) -> bool:
    """Check if the filename has an allowed image extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS
