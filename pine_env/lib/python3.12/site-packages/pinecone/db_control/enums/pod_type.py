from enum import Enum


class PodType(Enum):
    """
    PodType represents the available pod types for a pod index.
    """

    P1_X1 = "p1.x1"
    P1_X2 = "p1.x2"
    P1_X4 = "p1.x4"
    P1_X8 = "p1.x8"
    S1_X1 = "s1.x1"
    S1_X2 = "s1.x2"
    S1_X4 = "s1.x4"
    S1_X8 = "s1.x8"
    P2_X1 = "p2.x1"
    P2_X2 = "p2.x2"
    P2_X4 = "p2.x4"
    P2_X8 = "p2.x8"
