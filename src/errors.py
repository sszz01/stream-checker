from enum import Enum, auto

class StreamError(Enum):
    BLUR = auto()
    FREEZE = auto()
    CONNECTION_LOST = auto()