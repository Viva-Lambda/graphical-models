"""
\file catrandomval.py Categorical Random Variable
"""
from typing import Any, Callable, Dict, FrozenSet, List, Set, Tuple

from pygmodels.value.valtype import (
    CodomainValue,
    NumericValue,
    Outcome,
    PossibleOutcomes,
    DomainValue,
    MeasureValue,
)


from pygmodels.randvar.rvaltype import BaseRandomVar


class CatRandomVariable(BaseRandomVar):
    """!
    \brief A discrete/categorical random variable \see RandomVariable
    """

    def __init__(
        self,
        node_id: str,
        input_data: Dict[str, Any],
        f: Callable[[Outcome], CodomainValue] = lambda x: x,
        marginal_distribution: Callable[[CodomainValue], float] = lambda x: 1.0,
    ):
        """!
        \brief Constructor for categorical/discrete random variable

        \param marginal_distribution a function that takes in a value from
        codomain of the random variable and outputs a value in the range [0,1].
        Notice that is not a local distribution, it should be the marginal
        distribution that is independent of local structure.

        \throws ValueError We raise a value error if the probability values
        associated to outcomes add up to a value bigger than one.

        For other parameters and the definition of a random variable \see
        RandomVariable .

        A simple data specification is provided for passing evidences and
        input.
        The possible outcomes key holds a set of values belonging to space of
        possible outcomes. If the input data just contains a key as
        'possible-outcomes', we suppose that it contains a PossibleOutcomes
        object which represents the space of all possible outcomes of the
        measurable event set associated to random variable.

        The function associated to our random variable transforms the set of
        possible outcomes to values as per its definition in Koller, Friedman,
        2009, p. 20. Lastly we check whether obtained, or associated
        outcome-values satisfy the probability rule by checking if the
        probabilities associated to these values add up to one.

        \code{.py}

        >>> students = PossibleOutcomes(frozenset(["student_1", "student_2"]))
        >>> grade_f = lambda x: "F" if x == "student_1" else "A"
        >>> grade_distribution = lambda x: 0.1 if x == "F" else 0.9
        >>> indata = {"possible-outcomes": students}
        >>> rvar = CatRandomVariable(
        >>>    input_data=indata,
        >>>    node_id="myrandomvar",
        >>>    f=grade_f,
        >>>    marginal_distribution=grade_distribution
        >>> )

        \endcode
        """
        super().__init__(node_id=node_id, data=input_data, f=f)
        if "outcome-values" in input_data:
            psum = sum(list(map(marginal_distribution, input_data["outcome-values"])))
            if psum > 1 and psum < 0:
                raise ValueError("probability sum bigger than 1 or smaller than 0")
        self.dist = marginal_distribution

    def p(self, value: CodomainValue) -> float:
        """!
        \brief probability of given outcome value as per the associated
        distribution

        \param value a member of \f$\Omega\f$ set of possible outcomes.

        \returns probability value associated to the outcome
        """
        return self.dist(value)

    def marginal(self, value: CodomainValue) -> float:
        """!
        \brief marginal distribution that is the probability of an outcome

        from Biagini, Campanino, 2016, p. 35
        <blockquote>
        Marginal distribution of X is the function: \f$p_1(x_i) = P(X=x_i)\f$
        </blockquote>

        \see CatRandomVariable.p

        \returns probability value associated to value
        """
        return self.p(value)

    def values(self):
        """!
        \brief outcome values of the random variable

        \see CatRandomVariable constructor for more explanation about outcome
        values and their relation to random variables. \see
        CatRandomVariable.value_set for a more functional version of this
        function which let's you associate several transformations and filters
        before obtaining outcomes.

        \throws KeyError We raise a key error if there are no values associated
        to this random variable.

        \returns possible outcomes associated to this random variable.

        \code{.py}
        >>> students = PossibleOutcomes(frozenset(["student_1", "student_2"]))
        >>> grade_f = lambda x: "F" if x == "student_1" else "A"
        >>> grade_distribution = lambda x: 0.1 if x == "F" else 0.9
        >>> indata = {"possible-outcomes": students}
        >>> rvar = CatRandomVariable(
        >>>    input_data=indata,
        >>>    node_id="myrandomvar",
        >>>    f=grade_f,
        >>>    marginal_distribution=grade_distribution
        >>> )
        >>> rvar.values()
        >>> frozenset(["A", "F"])

        \endcode
        """
        vdata = self.data()
        if "outcome-values" not in vdata:
            raise KeyError("This random variable has no associated set of values")
        return vdata["outcome-values"]

    def value_set(
        self, value_filter=lambda x: True, value_transform=lambda x: x,
    ) -> FrozenSet[Tuple[str, NumericValue]]:
        """!
        \brief the outcome value set of the random variable.

        \param value_filter function for filtering out values during the
        retrieval.

        \param value_transfom function for transforming values during the
        retrieval

        \returns codomain of random variable, that is possible outcomes
        associated to random variable

        This is basically the codomain of the function associated to random
        variable. Notice that this is completely different from probabilities
        and other statistical discussion.
        We also brand each value with the identifier of this random variable.
        When we are dealing with categorical random variables, this function
        should work, however for continuous codomains it would not really work.

        \code{.py}
        >>> students = PossibleOutcomes(frozenset(["student_1", "student_2"]))
        >>> grade_f = lambda x: "F" if x == "student_1" else "A"
        >>> grade_distribution = lambda x: 0.1 if x == "F" else 0.9
        >>> indata = {"possible-outcomes": students}
        >>> rvar = CatRandomVariable(
        >>>    input_data=indata,
        >>>    node_id="myrandomvar",
        >>>    f=grade_f,
        >>>    marginal_distribution=grade_distribution
        >>> )
        >>> rvar.value_set(
        >>>         value_transform=lambda x: x.lower(),
        >>>         value_filter=lambda x: x != "A"
        >>> )
        >>> frozenset([("myrandomvar","f")])

        \endcode
        """
        sid = self.id()
        return frozenset(
            [
                (sid, value_transform(v))
                for v in self.values()
                if value_filter(v) is True
            ]
        )
