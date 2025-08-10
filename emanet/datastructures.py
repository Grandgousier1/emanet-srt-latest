from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Word:
    """Represents a single word with start and end timestamps."""
    text: str
    start: float
    end: float

@dataclass
class Segment:
    """Represents a segment of transcribed text."""
    start: float
    end: float
    text: str
    words: Optional[List[Word]] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None

@dataclass
class Cue:
    """Represents a single subtitle cue."""
    idx: int
    start: float
    end: float
    text: str
