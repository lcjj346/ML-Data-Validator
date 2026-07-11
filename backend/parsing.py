"""
Upload parsing - tolerant CSV/Excel reading with human-friendly errors.

Strategy: try a strict parse first. If the file has malformed rows, re-parse
tolerantly (skipping bad rows) and surface a warning so the user knows exactly
what was dropped. Only fail hard when even the tolerant parse cannot produce
usable data - and then with a message a non-technical user can act on.
"""

import csv
import io
import re
from typing import Optional, Tuple

import pandas as pd
from fastapi import HTTPException


def _friendly_parser_error(err: Exception) -> str:
    """Translate pandas tokenizer errors into plain language."""
    msg = str(err)
    m = re.search(r"Expected (\d+) fields in line (\d+), saw (\d+)", msg)
    if m:
        expected, line, saw = m.groups()
        return (
            f"Row {line} has {saw} values but the header has {expected} columns - "
            f"check that row for unquoted commas or missing values."
        )
    return f"Could not read the file: {msg}"


def parse_upload(filename: str, contents: bytes) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Parse uploaded CSV/XLSX bytes into a DataFrame.

    Returns (df, warning). warning is None for a clean parse, or a
    human-readable note when malformed rows had to be skipped.
    Raises HTTPException(400) with a friendly message when parsing fails.
    """
    if filename.endswith(".xlsx"):
        try:
            df = pd.read_excel(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read the Excel file: {e}")
        if df.empty:
            raise HTTPException(status_code=400, detail="The Excel file contains no data rows.")
        return df, None

    # 1. Strict parse - the happy path
    try:
        df = pd.read_csv(io.BytesIO(contents))
        if df.empty:
            raise HTTPException(status_code=400, detail="The CSV file contains no data rows.")
        return df, None
    except pd.errors.ParserError as strict_error:
        friendly = _friendly_parser_error(strict_error)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The CSV file is empty.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read the file: {e}")

    # 2. Tolerant re-parse - skip malformed rows and tell the user
    try:
        df = pd.read_csv(io.BytesIO(contents), engine="python", on_bad_lines="skip")
    except Exception:
        # Even tolerant parsing failed - report the original, translated error
        raise HTTPException(status_code=400, detail=friendly)

    if df.empty:
        raise HTTPException(status_code=400, detail=friendly)

    # Count the rows pandas dropped (rows with MORE fields than the header -
    # short rows are padded with NaN, not dropped). csv.reader respects quoting,
    # so quoted commas are not miscounted.
    skipped = []
    try:
        reader = csv.reader(io.StringIO(contents.decode("utf-8", errors="replace")))
        rows = [r for r in reader if r]
        header_len = len(rows[0]) if rows else 0
        skipped = [(line_no, len(r)) for line_no, r in enumerate(rows[1:], start=2) if len(r) > header_len]
    except csv.Error:
        pass  # counting is best-effort; the parse itself already succeeded

    example = f" (e.g. row {skipped[0][0]} has {skipped[0][1]} values instead of {len(df.columns)})" if skipped else ""
    warning = (
        f"{len(skipped)} malformed row(s) were skipped{example}. "
        f"{len(df)} valid rows were loaded."
    )
    return df, warning
