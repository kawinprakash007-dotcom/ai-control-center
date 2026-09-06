from typing import Optional, Union, Dict, Any, List
import re

from core.interfaces.memory_interface import MemoryServiceInterface
from core.models.task import Task


class MemoryCapability:
    """
    Adapter between AI Control Center's Tool protocol and MemoryServiceInterface.
    Executes explicit long-term memory operations: SAVE, READ, FORGET.
    Does not manipulate SQLite directly or expose storage internals.
    """

    def __init__(self, memory_service: Optional[MemoryServiceInterface] = None):
        """
        Initialize MemoryCapability with an injected or lazily-retrieved MemoryService.
        """
        if memory_service is not None:
            self.memory_service = memory_service
        else:
            # Retrieve standard memory store through history compatibility shim
            # without directly importing or binding to SQLite internals
            try:
                from memory.history import get_store
                self.memory_service = get_store()
            except ImportError:
                self.memory_service = None

    def __call__(self, task: Optional[Union[Task, str, Dict[str, Any]]] = None) -> str:
        """
        Execute explicit memory operation for the given task.

        Args:
            task: Task instance, dict of parameters, or raw query string.

        Returns:
            Deterministic response string.
        """
        params: Dict[str, Any] = {}

        if isinstance(task, dict):
            params = task
        elif task is not None and hasattr(task, "parameters") and isinstance(task.parameters, dict):
            params = task.parameters
        elif isinstance(task, str):
            params = self._extract_parameters_from_string(task)

        # Service resolution: task-injected service takes precedence over default
        service = params.get("memory_service") or self.memory_service
        if service is None:
            raise RuntimeError("Memory service is unavailable.")

        user_id = str(params.get("user_id") or "default_user").strip()
        if not user_id:
            user_id = "default_user"

        action = str(params.get("action") or params.get("memory_action") or "").strip().lower()
        key = str(params.get("key") or "").strip()
        value = params.get("value")

        if not action:
            return "No memory operation specified."

        clean_key = key.replace("_", " ")

        # ------------------------------------------------------------------
        # 1. SAVE / REMEMBER
        # ------------------------------------------------------------------
        if action == "save":
            if not key:
                return "Cannot save memory: missing preference key."
            if value is None or (isinstance(value, str) and not value.strip()):
                return "Cannot save memory: missing preference value."

            service.save_preference(
                user_id=user_id,
                key=key,
                value=value,
                category="general",
            )
            return f"Got it — I'll remember that your {clean_key} is {value}."

        # ------------------------------------------------------------------
        # 2. READ / RECALL
        # ------------------------------------------------------------------
        if action == "read":
            if key in ("", "all", "memories", "preferences"):
                entries = service.list_preferences(user_id=user_id)
                if not entries:
                    return "I don't have any memories stored about you yet."
                formatted_lines = [f"- {e.key.replace('_', ' ')}: {e.value}" for e in entries]
                return "I currently remember:\n" + "\n".join(formatted_lines)

            entry = service.get_preference(user_id=user_id, key=key)
            if entry is None and not key.endswith("_name"):
                entry = service.get_preference(user_id=user_id, key=f"{key}_name")
            elif entry is None and key.endswith("_name"):
                entry = service.get_preference(user_id=user_id, key=key[:-5])

            if entry is not None:
                display_key = entry.key.replace("_", " ")
                return f"Your {display_key} is {entry.value}."

            return f"I don't have anything remembered for your {clean_key}."

        # ------------------------------------------------------------------
        # 3. FORGET / DELETE
        # ------------------------------------------------------------------
        if action == "forget":
            if not key or key in ("all", "everything"):
                return "Cannot forget memory: missing or ambiguous preference key."

            deleted = service.delete_preference(user_id=user_id, key=key)
            if not deleted and not key.endswith("_name"):
                deleted = service.delete_preference(user_id=user_id, key=f"{key}_name")
            elif not deleted and key.endswith("_name"):
                deleted = service.delete_preference(user_id=user_id, key=key[:-5])

            if deleted:
                return "Okay — I've forgotten that preference."

            return f"I couldn't find any memory for your {clean_key} to forget."

        # ------------------------------------------------------------------
        # 4. UNSUPPORTED / INVALID
        # ------------------------------------------------------------------
        return f"Unsupported memory operation: '{action}'."

    def _extract_parameters_from_string(self, text: str) -> Dict[str, Any]:
        """
        Fallback deterministic parameter extraction from raw string command.
        """
        raw = text.strip()

        # SAVE pattern
        save_match = re.match(
            r"^(?:remember|save|store\s+preference|store)[:\s]+(?:that\s+)?(?:the\s+preference\s+that\s+)?(?:my\s+)?(.+?)\s+(?:is\s+called|is\s+named|is\s+set\s+to|is|was|to\s+be|as)\s+(.+?)[.!?]?$",
            raw,
            re.IGNORECASE,
        )
        if save_match:
            k = self._normalize_key(save_match.group(1))
            v = save_match.group(2).strip()
            return {"action": "save", "key": k, "value": v}

        # FORGET pattern
        forget_match1 = re.match(
            r"^(?:forget|delete|remove|clear)\s+(?:that\s+)?(?:my\s+)?(.+?)\s+(?:is\s+called|is\s+named|is|was|to\s+be|as)\s+(.+?)[.!?]?$",
            raw,
            re.IGNORECASE,
        )
        if forget_match1:
            k = self._normalize_key(forget_match1.group(1))
            return {"action": "forget", "key": k}

        forget_match2 = re.match(
            r"^(?:forget|delete|remove|clear)\s+(?:my\s+)?(?:saved\s+)?preference\s+for\s+(.+?)[.!?]?$",
            raw,
            re.IGNORECASE,
        )
        if forget_match2:
            k = self._normalize_key(forget_match2.group(1))
            return {"action": "forget", "key": k}

        forget_match3 = re.match(
            r"^(?:forget|delete|remove|clear)\s+(?:my\s+)?(.+?)[.!?]?$",
            raw,
            re.IGNORECASE,
        )
        if forget_match3:
            k = self._normalize_key(forget_match3.group(1))
            return {"action": "forget", "key": k}

        # READ broad
        if re.search(
            r"\b(?:what\s+do\s+you\s+(?:remember|know)\s+about\s+me|what\s+do\s+you\s+remember|what\s+memories\s+do\s+you\s+have|list\s+my\s+(?:memories|preferences)|show\s+my\s+(?:memories|preferences))\b",
            raw,
            re.IGNORECASE,
        ):
            return {"action": "read", "key": "all"}

        # READ specific
        read_match1 = re.match(
            r"^(?:what\s+is|what\'s)\s+my\s+(.+?)[?.!]?$",
            raw,
            re.IGNORECASE,
        )
        if read_match1:
            k = self._normalize_key(read_match1.group(1))
            return {"action": "read", "key": k}

        read_match2 = re.match(
            r"^what\s+did\s+i\s+tell\s+you\s+(?:that\s+)?my\s+(.+?)\s+was[?.!]?$",
            raw,
            re.IGNORECASE,
        )
        if read_match2:
            k = self._normalize_key(read_match2.group(1))
            return {"action": "read", "key": k}

        read_match3 = re.match(
            r"^(?:recall|tell\s+me)\s+(?:my\s+)?(.+?)[?.!]?$",
            raw,
            re.IGNORECASE,
        )
        if read_match3:
            k = self._normalize_key(read_match3.group(1))
            return {"action": "read", "key": k}

        return {}

    @staticmethod
    def _normalize_key(raw_key: str) -> str:
        k = raw_key.strip()
        k = re.sub(r"^(?:my|the|preference\s+for|saved\s+preference\s+for)\s+", "", k, flags=re.IGNORECASE)
        k = re.sub(r"[^\w\s-]", "", k).strip().lower()
        k = re.sub(r"[\s-]+", "_", k)
        return k
