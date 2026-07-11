"""
In-memory session store - replaces Streamlit's st.session_state.

Each session holds a DataFrame, validation results, corrections, etc.
Sessions expire after 30 minutes of inactivity.
"""

import time
import uuid
import threading
import pandas as pd
from typing import Dict, Optional


class Session:
    """A single user session with all validation/training state."""

    def __init__(self):
        self.created_at: float = time.time()
        self.last_accessed: float = time.time()

        # Upload state
        self.filename: Optional[str] = None
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None

        # Validation state
        self.cell_validity: Dict[str, bool] = {}  # "row_col" -> bool
        self.cell_confidence: Dict[str, float] = {}  # "row_col" -> confidence
        self.modified_cells: set = set()  # set of "row_col" keys
        self.column_mappings: Dict[str, str] = {}
        self.corrections: list = []
        self.audit_log: list = []  # chronological record of every cell change

        # Training state
        self.training_df: Optional[pd.DataFrame] = None
        self.training_filename: Optional[str] = None
        self.training_metrics: Optional[dict] = None

    def touch(self):
        self.last_accessed = time.time()


class SessionStore:
    """Thread-safe in-memory session store with TTL cleanup."""

    TTL_SECONDS = 30 * 60  # 30 minutes
    CLEANUP_INTERVAL = 5 * 60  # 5 minutes

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self._start_cleanup()

    def create(self) -> str:
        session_id = str(uuid.uuid4())
        with self._lock:
            self._sessions[session_id] = Session()
        return session_id

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session

    def delete(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

    def _cleanup(self):
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items()
                if now - s.last_accessed > self.TTL_SECONDS
            ]
            for sid in expired:
                del self._sessions[sid]

    def _start_cleanup(self):
        def run():
            while True:
                time.sleep(self.CLEANUP_INTERVAL)
                self._cleanup()

        t = threading.Thread(target=run, daemon=True)
        t.start()


# Global singleton
store = SessionStore()
