# letta_memory.py
import os
import uuid
import sys
import psycopg2
from psycopg2 import pool
from typing import Dict, Any, Type
from ai_memory_sdk import Memory as LettaMemory
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_module.ArkModelNew import (
    Message,
    UserMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

ROLE_TO_CLASS: Dict[str, Type[Message]] = {
    "system": SystemMessage,
    "user": UserMessage,
    "assistant": AIMessage,
    "tool": ToolMessage,
}

CLASS_TO_ROLE: Dict[Type[Message], str] = {
    SystemMessage: "system",
    UserMessage: "user",
    AIMessage: "assistant",
    ToolMessage: "tool",
}

load_dotenv()

LETTA_KEY = os.getenv("LETTA_KEY")
LETTA_BASE_URL = os.getenv("LETTA_BASE_URL")
LETTA_MODEL = os.getenv("LETTA_MODEL", "openai-proxy/Qwen/Qwen3-8B")

_connection_pool = None
_pool_lock = threading.Lock()

# TODO: condition??
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="letta_bg")


def _get_pool(db_url: str):
    """Get or create the global connection pool."""
    global _connection_pool
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                _connection_pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=db_url
                )
    return _connection_pool


class Memory:
    """
    Connects agent to supabase backend for long
    and short term memories

    """

    def __init__(self, user_id: str, session_id: str, db_url: str, use_long_term: bool = True):
        self.user_id = user_id
        self.db_url = db_url
        self.use_long_term = use_long_term

        self._pool = _get_pool(db_url)

        self._letta = None
        if self.use_long_term:
            self._letta = LettaMemory(
                api_key=LETTA_KEY,
                base_url=LETTA_BASE_URL,
                model=LETTA_MODEL,
            )

        self.session_id = session_id if session_id is not None else str(uuid.uuid4())

    def start_new_session(self):
        """Start a new chat session."""
        self.session_id = str(uuid.uuid4())
        return self.session_id

    def serialize(self, message: Message) -> str:
        """
        Convert a Message subclass into the string stored in Postgres.
        Store role separately in the role column.
        """
        return message.model_dump_json()

    def deserialize(self, message: str, role: str) -> Message:
        """
        Convert the stored Postgres string back into the correct Message subclass.
        Requires the role column value.
        """
        cls = ROLE_TO_CLASS.get(role)
        if cls is None:
            raise ValueError(f"Unknown role: {role}")
        return cls.model_validate_json(message)

    def _add_to_letta_background(self, role: str, content: str):
        """Background task to send a message to Letta (non-blocking)."""
        try:
            if self._letta:
                self._letta.add_messages(
                    self.user_id,
                    messages=[{"role": role, "content": content}],
                    skip_vector_storage=False,
                )
        except Exception as e:
            print(f"[letta background] Error: {e}")

    def add_memory(self, message) -> bool:
        """Add a single turn to Postgres (fast) + Letta in background."""
        try:
            role = CLASS_TO_ROLE[type(message)]

            conn = self._pool.getconn()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO conversation_context (user_id, session_id, role, message)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (self.user_id, self.session_id, role, self.serialize(message)),
                )
                conn.commit()
                cur.close()
            finally:
                self._pool.putconn(conn)

            if self.use_long_term and self._letta:
                _executor.submit(self._add_to_letta_background, role, message.content)

            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False

    def retrieve_long_memory(self, context: list = []) -> SystemMessage:
        """Retrieve relevant long-term memories for the current user."""
        if not self.use_long_term or not self._letta:
            return SystemMessage(content="")

        try:
            parts = []

            user_memory = self._letta.get_user_memory(self.user_id, prompt_formatted=True)
            if user_memory:
                parts.append(user_memory)

            summary = self._letta.get_summary(self.user_id, prompt_formatted=True)
            if summary:
                parts.append(summary)

            query = " ".join(m.content for m in context[-2:] if hasattr(m, "content"))
            if query.strip():
                results = self._letta.search(self.user_id, query=query)
                if results:
                    parts.append("relevant passages:\n" + "\n".join(f"- {r}" for r in results))

            if not parts:
                return SystemMessage(content="")

            return SystemMessage(content="\n\n".join(parts))

        except Exception as e:
            print(f"[retrieve_long_memory] Error: {e}")
            return SystemMessage(content="")

    def retrieve_short_memory(self, turns):
        """Retrieve relevant short term memories for the current user"""
        try:
            conn = self._pool.getconn()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT role, message
                    FROM (
                        SELECT id, role, message
                        FROM conversation_context
                        WHERE user_id = %s
                        ORDER BY id DESC
                        LIMIT %s
                    ) sub
                    ORDER BY id ASC
                    """,
                    (self.user_id, turns),
                )
                rows = cur.fetchall()
                cur.close()
            finally:
                self._pool.putconn(conn)

            return [self.deserialize(message=msg, role=role) for role, msg in rows]

        except Exception as e:
            print(f"[retrieve_short_memory] Error: {e}")
            return []