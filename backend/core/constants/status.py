from enum import Enum


class Status(Enum):

    PENDING = "pending"

    RUNNING = "running"

    SUCCESS = "success"

    FAILED = "failed"