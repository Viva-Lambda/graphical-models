"""!
\file valtype.py Value type
"""
from typing import Any, Callable, Dict, FrozenSet, List, Set, Tuple

NumericValue = float

DomainVal = Any

MeasureValue = float


class Outcome:
    def __init__(self, v):
        self.data = v


class DomainValue:
    """
    """

    def __init__(self, v):
        self.data = v


class CodomainValue:
    """
    """

    def __init__(self, v):
        self.data = v


class PossibleOutcomes:
    """!
    \brief set of possible outcomes from Koller, Friedman 2009, p. 15, 20

    This is simply a frozenset. We assume that possible outcomes contained in
    this object are measurable.
    """

    def __init__(self, omega: FrozenSet[Outcome]):
        self.data = omega
