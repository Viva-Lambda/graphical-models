"""
\file abstractfactor.py Abstract Factor object
"""
from abc import abstractmethod
from pygmodels.gtype.abstractobj import (
    AbstractGraphObj,
    AbstractNode,
)
from pygmodels.gtype.graphobj import GraphObject
from pygmodels.factortype.abstractfactor import AbstractFactor


class BaseFactor(AbstractFactor, GraphObject):
    """
    """

    def __init__(
        self,
        gid: str,
        scope_vars: Set[NumCatRVariable],
        factor_fn: Optional[Callable[[Set[Tuple[str, NumCatRVariable]]], float]] = None,
        data={},
    ):
        """"""
        super().__init__(oid=gid, odata=data)
        for svar in scope_vars:
            vs = svar.values()
            if any([v < 0 for v in vs]):
                msg = "Scope variables contain a negative value."
                msg += " Negative factors are not allowed"
                raise ValueError(msg)

        ## random variables belonging to this factor
        self.svars = scope_vars
