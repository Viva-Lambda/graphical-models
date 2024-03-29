"""!
\file factoralg.py Factor algebra operations
"""

from functools import reduce as freduce
from itertools import combinations
from typing import Callable, FrozenSet, List, Optional, Set, Tuple, Union
from uuid import uuid4

from pygmodels.factor.factorf.factorops import FactorFactorableOps, FactorOps
from pygmodels.factor.ftype.abstractfactor import (
    AbstractFactor,
    DomainSliceSet,
    DomainSubset,
    FactorDomain,
    FactorScope,
)
from pygmodels.factor.ftype.basefactor import BaseFactor
from pygmodels.randvar.rtype.abstractrandvar import AbstractRandomVariable


class FactorAlgebra:
    """
    Namespace for factor algebra, ie, operations that takes factors as input
    and output
    """

    @staticmethod
    def product(
        f: AbstractFactor,
        other: AbstractFactor,
        product_fn=lambda x, y: x * y,
        accumulator=lambda added, accumulated: added * accumulated,
    ) -> Tuple[AbstractFactor, float]:
        """!
        Wrapper of FactorOps.cls_product
        """
        ((scope, phi), prod) = FactorOps.product(
            f=f,
            other=other,
            product_fn=product_fn,
            accumulator=accumulator,
        )
        return (
            BaseFactor(gid=str(uuid4()), scope_vars=scope, factor_fn=phi),
            prod,
        )

    @staticmethod
    def reduced(
        f: AbstractFactor, assignments: DomainSubset
    ) -> AbstractFactor:
        """!
        Wrapper of FactorOps.cls_reduced
        """
        (scope, phi) = FactorOps.reduced(f=f, assignments=assignments)
        return BaseFactor(gid=str(uuid4()), scope_vars=scope, factor_fn=phi)

    @staticmethod
    def reduced_by_value(
        f: AbstractFactor, assignments: DomainSubset
    ) -> AbstractFactor:
        """!
        Wrapper of FactorOps.reduced_by_value
        """
        (scope, phi) = FactorFactorableOps.reduced_by_value(
            f=f, assignments=assignments
        )
        return BaseFactor(gid=str(uuid4()), scope_vars=scope, factor_fn=phi)

    @staticmethod
    def filter_assignments(
        f: AbstractFactor, assignments: DomainSubset
    ) -> AbstractFactor:
        """!
        Wrapper of FactorOps.cls_filter_assignments
        """
        (scope, phi) = FactorOps.filter_assignments(
            f=f, assignments=assignments
        )
        return BaseFactor(gid=str(uuid4()), scope_vars=scope, factor_fn=phi)

    @staticmethod
    def reduced_by_vars(
        f: AbstractFactor, assignments: DomainSubset
    ) -> AbstractFactor:
        """!
        Wrapper of FactorOps.reduced_by_vars
        """
        (scope, phi) = FactorFactorableOps.reduced_by_vars(
            f=f, assignments=assignments
        )
        return BaseFactor(gid=str(uuid4()), scope_vars=scope, factor_fn=phi)

    @staticmethod
    def maxout_var(
        f: AbstractFactor, Y: AbstractRandomVariable
    ) -> AbstractFactor:
        """!
        Wrapper of FactorOps.maxout_var
        """
        (scope, phi) = FactorFactorableOps.maxout_var(f=f, Y=Y)
        return BaseFactor(gid=str(uuid4()), scope_vars=scope, factor_fn=phi)

    @staticmethod
    def sumout_var(
        f: AbstractFactor, Y: AbstractRandomVariable
    ) -> AbstractFactor:
        """!
        Wrapper of FactorOps.cls_sumout_var
        """
        (scope, phi) = FactorFactorableOps.sumout_var(f=f, Y=Y)
        return BaseFactor(gid=str(uuid4()), scope_vars=scope, factor_fn=phi)

    @staticmethod
    def sumout_vars(
        f: AbstractFactor, Ys: Set[AbstractRandomVariable]
    ) -> AbstractFactor:
        """!
        \brief Sum the variable out of factor as per Koller, Friedman 2009, p. 297

        \see Factor.sumout_var(Y)

        \return Factor
        """
        if len(Ys) == 0:
            raise ValueError("variables not be an empty set")
        if len(Ys) == 1:
            v = Ys.pop()
            return FactorAlgebra.sumout_var(f, v)
        ylst = list(Ys)
        fac = FactorAlgebra.sumout_var(f, ylst[0])
        for i in range(1, len(ylst)):
            fac = FactorAlgebra.sumout_var(fac, ylst[i])
        return fac
