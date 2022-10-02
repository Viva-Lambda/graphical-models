"""!
\file numericops.py Operations outputting numeric values with numeric categorical
variables
"""

import math
from pygmodels.randvar.randvarmodel.categorical import NumCatRandomVariable
from pygmodels.randvar.randvarmodel.categorical import CatRandomVariable
from pygmodels.randvar.randvarops.categoricalops import NumericOps as CNumericOps
from pygmodels.randvar.randvarops.numeric.boolops import BoolOps
from typing import Any, Callable, FrozenSet, List, Optional, Set, Tuple
from pygmodels.value.value import NumericValue
from pygmodels.randvar.randvartype.abstractrandvar import (
    AbstractEvidence,
    AbstractRandomVariable,
    AssociatedValueSet,
)
from pygmodels.value.codomain import CodomainValue
from pygmodels.utils import is_type, type_check


class NumericOps:
    """!
    Basic operations outputting numeric values that can be applied to categorical
    random variables
    """

    @staticmethod
    def max(r: NumCatRandomVariable, sampler: Callable = lambda x: x) -> NumericValue:
        """!
        \brief maximum marginal value

        We return the highest marginal/probability.

        \todo Update documentation evidence has its own type now.

        \todo test DONE

        \code{.py}
        >>> nid1 = "rvar1"
        >>> input_data = {
        >>>    "intelligence": {"outcome-values": [0.1, 0.9], "evidence": 0.9},
        >>>    "grade": {"outcome-values": [0.2, 0.4, 0.6], "evidence": 0.2},
        >>>    "dice": {"outcome-values": [i for i in range(1, 7)], "evidence": 1.0 / 6},
        >>> }

        >>> def intelligence_dist(intelligence_value: float) -> float:
        >>>    if intelligence_value == 0.1:
        >>>        return 0.7
        >>>    elif intelligence_value == 0.9:
        >>>        return 0.3
        >>>    else:
        >>>        raise ValueError(
        >>>            "intelligence_value does not belong to possible outcomes"
        >>>        )

        >>> # intelligence
        >>> intelligence = NumCatRVariable(
        >>>    node_id=nid1,
        >>>    input_data=input_data["intelligence"],
        >>>    distribution=intelligence_dist,
        >>> )
        >>> intelligence.max()
        >>> 0.7

        \endcode
        """
        mx, mxv = NumericOps.min_max_marginal_with_outcome(
            r, is_min=False, sampler=sampler
        )
        return mx

    @staticmethod
    def min(r: NumCatRandomVariable, sampler: Callable = lambda x: x) -> NumericValue:
        """!
        \brief minimum marginal value

        We return the lowest marginal/probability.

        \todo Update documentation evidence has its own type now.

        \todo test DONE

        \code{.py}
        >>> nid1 = "rvar1"
        >>> input_data = {
        >>>    "intelligence": {"outcome-values": [0.1, 0.9], "evidence": 0.9},
        >>>    "grade": {"outcome-values": [0.2, 0.4, 0.6], "evidence": 0.2},
        >>>    "dice": {"outcome-values": [i for i in range(1, 7)], "evidence": 1.0 / 6},
        >>> }

        >>> def intelligence_dist(intelligence_value: float) -> float:
        >>>    if intelligence_value == 0.1:
        >>>        return 0.7
        >>>    elif intelligence_value == 0.9:
        >>>        return 0.3
        >>>    else:
        >>>        raise ValueError("unknown intelligence event/possible outcome")

        >>> # intelligence
        >>> intelligence = NumCatRVariable(
        >>>    node_id=nid1,
        >>>    input_data=input_data["intelligence"],
        >>>    distribution=intelligence_dist,
        >>> )
        >>> intelligence.min()
        >>> 0.3

        \endcode

        """
        mx, mxv = NumericOps.min_max_marginal_with_outcome(
            r, is_min=True, sampler=sampler
        )
        return mx

    @staticmethod
    def min_max_marginal_with_outcome(
        r: NumCatRandomVariable, is_min: bool, sampler: Callable
    ) -> Tuple[float, NumericValue]:
        """!
        \brief returns highest/lowest probability with its outcome

        \param is_min flag for specifying whether to return lowest or highest
        probability-outcome pair

        \todo test
        """
        mx = float("inf") if is_min else float("-inf")
        mxv = None
        for v in r.image():
            marginal = r.marginal(v)
            cond = mx > marginal if is_min else mx < marginal
            if cond:
                mx = marginal
                mxv = v
        return mx, mxv

    @staticmethod
    def max_marginal_value(r: NumCatRandomVariable, sampler: Callable) -> NumericValue:
        """!
        \brief highest probability outcome

        Notice that this gives the outcome not the probability

        \todo Update documentation evidence has its own type now.

        \todo test DONE

        \code{.py}
        >>> nid1 = "rvar1"
        >>> input_data = {
        >>>    "intelligence": {"outcome-values": [0.1, 0.9], "evidence": 0.9},
        >>>    "grade": {"outcome-values": [0.2, 0.4, 0.6], "evidence": 0.2},
        >>>    "dice": {"outcome-values": [i for i in range(1, 7)], "evidence": 1.0 / 6},
        >>> }
        >>> def intelligence_dist(intelligence_value: float) -> float:
        >>>    if intelligence_value == 0.1:
        >>>        return 0.7
        >>>    elif intelligence_value == 0.9:
        >>>        return 0.3
        >>>    else:
        >>>        raise ValueError("unknown intelligence event/possible outcome")

        >>> # intelligence
        >>> intelligence = NumCatRVariable(
        >>>    node_id=nid1,
        >>>    input_data=input_data["intelligence"],
        >>>    distribution=intelligence_dist,
        >>> )
        >>> intelligence.max_marginal_value()
        >>> 0.1

        \endcode
        """
        mx, mxv = NumericOps.min_max_marginal_with_outcome(
            r, is_min=False, sampler=sampler
        )
        return mxv

    @staticmethod
    def min_marginal_value(r: NumCatRandomVariable, sampler) -> NumericValue:
        """!
        \brief highest probability outcome

        Notice that this gives the outcome not the probability

        \todo Update documentation evidence has its own type now.

        \todo test DONE

        \code{.py}
        >>> nid1 = "rvar1"
        >>> input_data = {
        >>>    "intelligence": {"outcome-values": [0.1, 0.9], "evidence": 0.9},
        >>>    "grade": {"outcome-values": [0.2, 0.4, 0.6], "evidence": 0.2},
        >>>    "dice": {"outcome-values": [i for i in range(1, 7)], "evidence": 1.0 / 6},
        >>> }

        >>> def intelligence_dist(intelligence_value: float) -> float:
        >>>    if intelligence_value == 0.1:
        >>>        return 0.7
        >>>    elif intelligence_value == 0.9:
        >>>        return 0.3
        >>>    else:
        >>>        raise ValueError("unknown intelligence event/possible outcome")

        >>> # intelligence
        >>> intelligence = NumCatRVariable(
        >>>    node_id=nid1,
        >>>    input_data=input_data["intelligence"],
        >>>    distribution=intelligence_dist,
        >>> )
        >>> intelligence.max_marginal_value()
        >>> 0.1

        \endcode
        """
        mx, mxv = NumericOps.min_max_marginal_with_outcome(
            r, is_min=True, sampler=sampler
        )
        return mxv

    @staticmethod
    def marginal_over(
        r: NumCatRandomVariable,
        evidence: AbstractEvidence,
        other: AbstractRandomVariable,
    ) -> NumericValue:
        """!
        \brief Compute marginal distribution over other random variable given
        evidence with respect to current random variable.

        Implements the following from Biagini and Campanino 2016, p. 35:
        \f$ \sum_{j=1}^n p(x_i) p(y_j) = p(x_i) \sum_{j=1}^n p(y_j) \f$

        \todo Update documentation evidence has its own type now.

        \todo test DONE

        \code{.py}
        >>> input_data = {
        >>>    "intelligence": {"outcome-values": [0.1, 0.9], "evidence": 0.9},
        >>>    "grade": {"outcome-values": [0.2, 0.4, 0.6], "evidence": 0.2},
        >>>    "dice": {"outcome-values": [i for i in range(1, 7)], "evidence": 1.0 / 6},
        >>> }

        >>> def grade_dist(grade_value: float):
        >>>     if grade_value == 0.2:
        >>>         return 0.25
        >>>     elif grade_value == 0.4:
        >>>         return 0.37
        >>>     elif grade_value == 0.6:
        >>>         return 0.38
        >>>     else:
        >>>         raise ValueError("unknown grade value")

        >>> def fair_dice_dist(dice_value: float):
        >>>     if dice_value in [i for i in range(1, 7)]:
        >>>         return 1.0 / 6.0
        >>>     else:
        >>>         raise ValueError("dice value")


        >>> nid2 = "rvar2"
        >>> grade = NumCatRVariable(
        >>>    node_id=nid2, input_data=input_data["grade"], distribution=grade_dist
        >>> )
        >>> nid3 = "rvar3"
        >>> dice = NumCatRVariable(
        >>>    node_id=nid3, input_data=input_data["dice"], distribution=fair_dice_dist
        >>> )
        >>> grade.marginal_over(0.2, dice)
        >>> 0.875

        \endcode
        """
        is_type(
            other,
            originType=CatRandomVariable,
            shouldRaiseError=True,
            val_name="other",
        )
        marginal = r.p(evidence.value())

        def phifn(x: CodomainValue):
            return x.value * marginal

        return CNumericOps.p_x_fn(other, phi=phifn)

    @staticmethod
    def marginal_over_evidence_key(
        r: NumCatRandomVariable, other: AbstractRandomVariable
    ) -> NumericValue:
        """!
        Compute marginal using evidence key.
        This means that the evidence is encoded to data associated to
        random variable

        \todo Update documentation evidence has its own type now.

        \todo test DONE

        \code{.py}
        >>> input_data = {
        >>>    "intelligence": {"outcome-values": [0.1, 0.9], "evidence": 0.9},
        >>>    "grade": {"outcome-values": [0.2, 0.4, 0.6], "evidence": 0.2},
        >>>    "dice": {"outcome-values": [i for i in range(1, 7)], "evidence": 1.0 / 6},
        >>> }

        >>> def grade_dist(grade_value: float):
        >>>     if grade_value == 0.2:
        >>>         return 0.25
        >>>     elif grade_value == 0.4:
        >>>         return 0.37
        >>>     elif grade_value == 0.6:
        >>>         return 0.38
        >>>     else:
        >>>         raise ValueError("unknown grade value")

        >>> def fair_dice_dist(dice_value: float):
        >>>     if dice_value in [i for i in range(1, 7)]:
        >>>         return 1.0 / 6.0
        >>>     else:
        >>>         raise ValueError("dice value")


        >>> nid2 = "rvar2"
        >>> grade = NumCatRVariable(
        >>>    node_id=nid2, input_data=input_data["grade"], distribution=grade_dist
        >>> )
        >>> nid3 = "rvar3"
        >>> dice = NumCatRVariable(
        >>>    node_id=nid3, input_data=input_data["dice"], distribution=fair_dice_dist
        >>> )
        >>> grade.marginal_over_evidence_key(dice)
        >>> 0.875

        \endcode
        """
        BoolOps.has_evidence(r, shouldRaiseError=True)
        data = r.data()
        evidence_value = data["evidence"]
        return NumericOps.marginal_over(r, evidence_value, other)

    @staticmethod
    def expected_value(
        r: NumCatRandomVariable, sampler: Callable = lambda x: x
    ) -> NumericValue:
        """!
        \brief Expected value of random variable
        from Koller, Friedman 2009, p. 31

        Implements the following formula:
        \f$ \sum_{i=1}^n x_i p(x_i) \f$


        \todo Update documentation evidence has its own type now.

        \todo test DONE

        \code{.py}
        >>> input_data = {
        >>>    "intelligence": {"outcome-values": [0.1, 0.9], "evidence": 0.9},
        >>>    "grade": {"outcome-values": [0.2, 0.4, 0.6], "evidence": 0.2},
        >>>    "dice": {"outcome-values": [i for i in range(1, 7)], "evidence": 1.0 / 6},
        >>> }

        >>> def fair_dice_dist(dice_value: float):
        >>>     if dice_value in [i for i in range(1, 7)]:
        >>>         return 1.0 / 6.0
        >>>     else:
        >>>         raise ValueError("dice value unknown")

        >>> nid3 = "rvar3"
        >>> dice = NumCatRVariable(
        >>>    node_id=nid3, input_data=input_data["dice"], distribution=fair_dice_dist
        >>> )
        >>> dice.expected_value()
        >>> 3.5

        \endcode
        """
        return sum(
            [
                codomain_member.value * r.p(codomain_member)
                for codomain_member in r.image(sampler=sampler)
            ]
        )

    @staticmethod
    def variance(r: NumCatRandomVariable):
        """!
        Koller, Friedman 2009, p. 33
        \f$ E[X^2] - (E[X])^2 \f$

        \todo test DONE
        """
        E_X2 = CNumericOps.expected_apply(r, phi=lambda x: x.value * x.value)
        return E_X2 - (NumericOps.expected_value(r) ** 2)

    @staticmethod
    def standard_deviation(r: NumCatRandomVariable):
        """!
        standard deviation Koller, Friedman 2009, p. 33

        \todo test DONE
        """
        return math.sqrt(NumericOps.variance(r))

    @staticmethod
    def P_X_e(r: NumCatRandomVariable, sampler: Callable = lambda x: x) -> NumericValue:
        """!
        \brief evaluate probability with given random variable's evidence if it is
        given.

        We output the expected value if there is no evidence associated to
        random variable

        \todo Update documentation evidence has its own type now.

        \todo test DONE

        \code{.py}
        >>> input_data = {
        >>>    "intelligence": {"outcome-values": [0.1, 0.9], "evidence": 0.9},
        >>>    "grade": {"outcome-values": [0.2, 0.4, 0.6], "evidence": 0.2},
        >>>    "dice": {"outcome-values": [i for i in range(1, 7)], "evidence": 1.0 / 6},
        >>> }

        >>> def grade_dist(grade_value: float):
        >>>     if grade_value == 0.2:
        >>>         return 0.25
        >>>     elif grade_value == 0.4:
        >>>         return 0.37
        >>>     elif grade_value == 0.6:
        >>>         return 0.38
        >>>     else:
        >>>         raise ValueError("unknown grade value")

        >>> nid2 = "rvar2"
        >>> grade = NumCatRVariable(
        >>>    node_id=nid2, input_data=input_data["grade"], distribution=grade_dist
        >>> )
        >>> grade.P_X_e()
        >>> 0.25

        \endcode
        """
        if "evidence" in r.data():
            evidence = r.data()["evidence"]
            return r.p(evidence.value())
        return NumericOps.expected_value(r, sampler=sampler)

    @staticmethod
    def max_marginal_e(r: NumCatRandomVariable) -> NumericValue:
        """!
        evaluate max probability with given random variable's evidence if it is
        present.

        \todo Update documentation evidence has its own type now.

        \todo test DONE

        \code{.py}
        >>> input_data = {
        >>>    "intelligence": {"outcome-values": [0.1, 0.9], "evidence": 0.9},
        >>>    "grade": {"outcome-values": [0.2, 0.4, 0.6], "evidence": 0.2},
        >>>    "dice": {"outcome-values": [i for i in range(1, 7)], "evidence": 1.0 / 6},
        >>> }

        >>> def grade_dist(grade_value: float):
        >>>     if grade_value == 0.2:
        >>>         return 0.25
        >>>     elif grade_value == 0.4:
        >>>         return 0.37
        >>>     elif grade_value == 0.6:
        >>>         return 0.38
        >>>     else:
        >>>         raise ValueError("unknown grade value")

        >>> nid2 = "rvar2"
        >>> grade = NumCatRVariable(
        >>>    node_id=nid2, input_data=input_data["grade"], distribution=grade_dist
        >>> )
        >>> grade.max_marginal_e()
        >>> 0.25

        \endcode
        """
        if "evidence" in r.data():
            evidence = r.data()["evidence"]
            return r.p(evidence.value())
        return NumericOps.max(r)

    @staticmethod
    def joint(r: CatRandomVariable, v: CatRandomVariable):
        """!
        Joint distribution of two random variables
        from Biagini and Campanino 2016 p. 35

        \todo test DONE
        """
        is_type(
            v, shouldRaiseError=True, originType=NumCatRandomVariable, val_name="v",
        )
        is_type(
            r, shouldRaiseError=True, originType=NumCatRandomVariable, val_name="r",
        )
        return NumericOps.P_X_e(r) * NumericOps.P_X_e(v)

    @staticmethod
    def min_marginal_e(r: NumCatRandomVariable) -> NumericValue:
        """!
        \brief evaluate min probability with given random variable's evidence
        if it is present.

        \todo Update documentation evidence has its own type now.

        \todo test DONE

        \code{.py}
        >>> input_data = {
        >>>    "intelligence": {"outcome-values": [0.1, 0.9], "evidence": 0.9},
        >>>    "grade": {"outcome-values": [0.2, 0.4, 0.6], "evidence": 0.2},
        >>>    "dice": {"outcome-values": [i for i in range(1, 7)], "evidence": 1.0 / 6},
        >>> }

        >>> def grade_dist(grade_value: float):
        >>>     if grade_value == 0.2:
        >>>         return 0.25
        >>>     elif grade_value == 0.4:
        >>>         return 0.37
        >>>     elif grade_value == 0.6:
        >>>         return 0.38
        >>>     else:
        >>>         raise ValueError("unknown grade value")

        >>> nid2 = "rvar2"
        >>> grade = NumCatRVariable(
        >>>    node_id=nid2, input_data=input_data["grade"], distribution=grade_dist
        >>> )
        >>> grade.min_marginal_e()
        >>> 0.25

        \endcode

        """
        if "evidence" in r.data():
            evidence = r.data()["evidence"]
            return r.p(evidence.value())
        return NumericOps.min(r)

    @staticmethod
    def max_joint(r: NumCatRandomVariable, v: NumCatRandomVariable):
        """!
        max joint probability

        \todo test DONE
        """
        type_check(
            val=r, other=v, shouldRaiseError=True, originType=NumCatRandomVariable,
        )
        rm = NumericOps.max_marginal_e(r)
        vm = NumericOps.max_marginal_e(v)
        return rm * vm

    @staticmethod
    def conditional(r: NumCatRandomVariable, other: NumCatRandomVariable):
        """!
        Conditional probability distribution (Bayes rule)
        from Koller and Friedman

        \todo test DONE
        """
        is_type(val=other, originType=NumCatRandomVariable, shouldRaiseError=True)
        is_type(val=r, originType=NumCatRandomVariable, shouldRaiseError=True)
        opxe = NumericOps.P_X_e(other)
        return NumericOps.joint(r, other) / opxe

    @staticmethod
    def max_conditional(r: NumCatRandomVariable, other: NumCatRandomVariable):
        """!
        \todo test
        """
        is_type(val=other, originType=NumCatRandomVariable, shouldRaiseError=True)
        is_type(val=r, originType=NumCatRandomVariable, shouldRaiseError=True)
        joint = NumericOps.max_joint(r, other)
        return max(
            [v for v in CNumericOps.apply_to_marginals(other, lambda x: joint / x)]
        )

    @staticmethod
    def joint_matrix(r: NumCatRandomVariable, other: NumCatRandomVariable):
        """!
        Obtain the joint distribution of numeric random variables in matrix
        form as stated by Biagini and Campanino 2016 p. 35

        \todo test
        """