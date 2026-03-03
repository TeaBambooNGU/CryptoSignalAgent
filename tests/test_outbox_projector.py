"""Outbox 投影器测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.conversation.projector import OutboxProjector
from app.conversation.store import SQLiteConversationTruthStore


class _DummyMemoryService:
    def __init__(self) -> None:
        self.preference_writes: list[dict] = []
        self.correction_writes: list[dict] = []

    def apply_preference_write(self, **kwargs) -> None:
        self.preference_writes.append(kwargs)

    def apply_tool_correction_write(self, **kwargs) -> None:
        self.correction_writes.append(kwargs)


class OutboxProjectorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "conversation_state.db"
        self.store = SQLiteConversationTruthStore(str(db_path))
        self.memory = _DummyMemoryService()
        self.projector = OutboxProjector(
            truth_store=self.store,
            memory_service=self.memory,
            poll_interval_seconds=0.1,
            batch_size=10,
            max_attempts=3,
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_run_once_projects_memory_preference_event(self) -> None:
        self.store.enqueue_outbox_event(
            conversation_id="conv-1",
            turn_id="turn-1",
            request_id="req-1",
            event_type="memory.save_preference",
            payload={
                "user_id": "u-1",
                "preference": {"watchlist": ["BTC"]},
                "confidence": 0.9,
                "updated_at": 1,
            },
        )

        handled = self.projector.run_once()

        self.assertEqual(handled, 1)
        self.assertEqual(len(self.memory.preference_writes), 1)
        self.assertEqual(self.memory.preference_writes[0]["user_id"], "u-1")
        self.assertEqual(self.store.fetch_pending_outbox(limit=10), [])

    def test_run_once_ignores_unknown_event_and_marks_done(self) -> None:
        self.store.enqueue_outbox_event(
            conversation_id="conv-2",
            turn_id="turn-1",
            request_id="req-2",
            event_type="unknown.event",
            payload={"hello": "world"},
        )

        handled = self.projector.run_once()

        self.assertEqual(handled, 1)
        self.assertEqual(self.store.fetch_pending_outbox(limit=10), [])


if __name__ == "__main__":
    unittest.main()

