"""
\file rvaltype.py Random Variable type
"""

from abc import abstractmethod


import math
from random import choice
from typing import Any, Callable, Dict, FrozenSet, List, Set, Tuple
from uuid import uuid4

from pygmodels.value.valtype import (
    CodomainValue,
    NumericValue,
    Outcome,
    PossibleOutcomes,
    DomainValue,
    MeasureValue,
)
from pygmodels.randvar.abstractrval import AbstractRandomVariable
from pygmodels.graph.gtype.node import Node


class BaseRandomVar(AbstractRandomVariable, Node):
    """!
    \brief a Random Variable as defined by Koller, Friedman 2009, p. 20

    Citing from Koller, Friedman:
    <blockquote>
    Formally, a random variable, such as Grade, is defined by a function that
    associates with each outcome in \f$\Omega\f$ a value.
    </blockquote>

    It is important to note that domain and codomain of random variables are
    quite ambiguous. The \f$\Omega\f$ in the definition is set of possible
    outcomes, \see PossibleOutcomes object. In the context of probabilistic
    graphical models each random variable is also considered as a \see Node of
    a \see Graph. This object is meant to be a base class for further needs.
    It lacks quite a bit of methods. Hence it can not be used directly in a
    \see PGModel.
    """

    def __init__(
        self,
        node_id: str,
        data: Any,
        f: Callable[[Outcome], CodomainValue] = lambda x: x,
        marginal_distribution: Callable[[CodomainValue], MeasureValue] = lambda x: 1.0,
    ):
        """!
        \brief Constructor of a random variable

        \param data The data associated to random variable can be anything
        \param node_id identifier of random variable. Same identifier is used
        as node identifier in a graph.
        \param f a function who takes data or from data, and outputs anything.

        \returns a random variable instance
        """
        ndata = {}
        ndata.update(data)
        if "possible-outcomes" in data:
            ndata["outcome-values"] = frozenset(
                [f(v) for v in data["possible-outcomes"].data]
            )

        super().__init__(node_id=node_id, data=ndata)
        self._f = f
        self._dist = marginal_distribution

    @property
    def f(self):
        """!
        The associated function of the random variable that is supposed to be
        measured.
        """
        if self._f is None:
            raise ValueError("Random variable has not an associated function")
        return self._f

    @property
    def distribution(self):
        """!
        The associated measure function for random variable. It measures the
        associated function of the random variable
        """
        if self._dist is None:
            raise ValueError("Random variable has not an associated distribution")
        return self._dist

    def p(self, value: CodomainValue) -> MeasureValue:
        """!
        Outputs the measure value, i.e. probability measure of the random
        variable.
        """
        return self.dist(value)
