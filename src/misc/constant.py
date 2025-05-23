from enum import Enum


class DataType(str, Enum):
    MATH = "math"
    MULTIHOPQA = "multi-hop qa"
    SINGLECHOICE = "single-choice"
    KBQA = "kbqa"
    TABLEQA = "tableqa"
    TEXTGAME = "text-based game"
