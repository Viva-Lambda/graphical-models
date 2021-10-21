"""
\file abstractrval.py Abstract Random Variable
"""
from abc import abstractmethod


from pygmodels.graph.gtype.abstractobj import AbstractNode


class AbstractRandomVariable(AbstractNode):
    """!
    Abstract random variable
    """

    @abstractmethod
    def p(self, out):
        """!
        Measure the probability of the given outcome
        """
        raise NotImplementedError
