"""!
\file factor.py

Defining a factor from Koller and Friedman 2009, p. 106-107
"""

class BaseFactor(AbstractFactor, GraphObject):
    """"""

    def __init__(
        self,
        gid: str,
        scope_vars: Set[NumCatRVariable],
        factor_fn: Optional[
            Callable[[Set[Tuple[str, NumCatRVariable]]], float]
        ] = None,
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

        ## scope variable hash table
        self.domain_table = {s.id(): s for s in self.scope_vars()}

        self.factor_fn = factor_fn

        ## cartesian product of factor domain
        self.scope_products: List[Set[Tuple[str, NumericValue]]] = list(
            product(*self.vars_domain())
        )

        ## constant normalization value
        self.Z = self.partition_value(self.vars_domain())

    def __str__(self):
        """"""
        msg = "Factor: " + self.id() + "\n"
        msg += "Scope variables: " + str(self.domain_table)
        msg += "Factor function: " + str(self.factor_fn)
        return msg

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, n: AbstractFactor):
        """!
        Check factor equality based on their domain and codomain values

        \warning this function works for categorical/discrete factors. For
        continuous domain factors, this won't work.

        \todo Adapt to continuous factors as well.
        """
        if not isinstance(n, AbstractFactor):
            return False
        #
        other_domain = n.vars_domain()
        this_domain = self.vars_domain()
        if other_domain != this_domain:
            return False
        #
        for dval in product(*other_domain):
            other_phi = n.phi(dval)
            this_phi = self.phi(dval)
            if this_phi != other_phi:
                return False
        return True

    def is_same(self, n: AbstractFactor):
        """!
        Check if two objects are same using identifier.
        """
        if not isinstance(n, AbstractFactor):
            return False
        return self.id() == n.id()

    def scope_vars(self, f=lambda x: x) -> Set[NumCatRVariable]:
        """!
        \brief get variables that are inside the scope of this factor

        \param f is a function that transforms the scope of this factor.

        \code{.py}

        >>> A = NumCatRVariable("A",
        >>>                     input_data={"outcome-values": [True, False]},
        >>>                     marginal_distribution=lambda x: 0.5)
        >>> fac = Factor(gid=str(uuid4()), scope_vars=set([A]))
        >>> fac.scope_vars(f=lambda x: set([(x,x)]))
        >>> set([(A, A)])

        \endcode
        """
        return f(self.svars)

    def vars_domain(
        self,
        rvar_filter: Callable[[NumCatRVariable], bool] = lambda x: True,
        value_filter: Callable[[NumericValue], bool] = lambda x: True,
        value_transform: Callable[[NumericValue], NumericValue] = lambda x: x,
    ) -> List[FrozenSet[Tuple[str, NumericValue]]]:
        """!
        \brief Get factor domain
        \see Factor.fdomain(D, rvar_filter, value_filter, value_transform)
        """
        return self.fdomain(
            D=self.scope_vars(),
            rvar_filter=rvar_filter,
            value_filter=value_filter,
            value_transform=value_transform,
        )

    @classmethod
    def from_abstract_factor(cls, f: AbstractFactor):
        """!
        \brief make BaseFactor from an AbstractFactor
        """
        svar = f.scope_vars()
        fn = f.phi
        return BaseFactor(
            gid=f.id(), data=f.data(), factor_fn=fn, scope_vars=svar
        )

    @classmethod
    def from_joint_vars(cls, svars: Set[NumCatRVariable]):
        """!
        \brief Make factor from joint variables

        \param svars set of random variables in the scope of the future factor

        We assume that the factor for the given set of random variables would
        be their marginal product.

        \code

        >>> A = NumCatRVariable("A",
        >>>                     input_data={"outcome-values": [True, False]},
        >>>                     marginal_distribution=lambda x: 0.5)
        >>> B = NumCatRVariable("B",
        >>>                     input_data={"outcome-values": [True, False]},
        >>>                     marginal_distribution=lambda x: 0.5)


        >>> fac = Factor.from_joint_vars(svars=set([A, B]))

        \endcode
        """
        return Factor(gid=str(uuid4()), scope_vars=svars)

    @classmethod
    def from_scope_variables_with_fn(
        cls,
        svars: Set[NumCatRVariable],
        fn: Callable[[Set[Tuple[str, NumericValue]]], float],
    ):
        """!
        \brief Make a factor from scope variables and a preference function
        """
        return BaseFactor(gid=str(uuid4()), scope_vars=svars, factor_fn=fn)

    @classmethod
    def fdomain(
        cls,
        D: Set[NumCatRVariable],
        rvar_filter: Callable[[NumCatRVariable], bool] = lambda x: True,
        value_filter: Callable[[NumericValue], bool] = lambda x: True,
        value_transform: Callable[[NumericValue], NumericValue] = lambda x: x,
    ) -> List[FrozenSet[Tuple[str, NumericValue]]]:
        """!
        \brief Get factor domain Val(D) D being a set of random variables

        \param D set of random variables
        \param rvar_filter filtering function for random variables
        \param value_filter filtering values from random variables' codomain
        \param value_transform apply a certain transformation to values from random variables' codomain.

        \return list of codomain of random variables

        \code

        >>> A = NumCatRVariable("A",
        >>>                     input_data={"outcome-values": [True, False]},
        >>>                     marginal_distribution=lambda x: 0.5)

        >>> B = NumCatRVariable("B",
        >>>                     input_data={"outcome-values": [True, False]},
        >>>                     marginal_distribution=lambda x: 0.5)

        >>> D = set([A,B])

        >>> fmatches = Factor.fdomain(D=D)
        >>> print(fmatches)

        >>> [frozenset(("A", True), ("A", True)),
        >>>  frozenset(("B", True), ("B", False)),
        >>> ]

        \endcode

        """
        return [
            s.value_set(
                value_filter=value_filter, value_transform=value_transform
            )
            for s in D
            if rvar_filter(s)
        ]

    def phi(self, scope_product: Set[Tuple[str, NumericValue]]) -> float:
        """!
        \brief obtain a factor value for given scope random variables

        Obtain factor value for given argument

        \param scope_product a row in conditional probability table of factor

        \code
        >>> A = NumCatRVariable("A",
        >>>                     input_data={"outcome-values": [True, False]},
        >>>                     marginal_distribution=lambda x: 0.5)

        >>> B = NumCatRVariable("B",
        >>>                     input_data={"outcome-values": [True, False]},
        >>>                     marginal_distribution=lambda x: 0.5)

        >>> fac = Factor.from_joint_vars(svars=set([A, B]))
        >>> fac.phi(scope_product=set([("A", True), ("B", True)]))
        >>> 0.25

        \endcode
        """
        return self.factor_fn(scope_product)

    def phi_normal(
        self, scope_product: Set[Tuple[str, NumericValue]]
    ) -> float:
        """!
        \brief normalize a given factor value

        \param scope_product a row in conditional probability table of factor

        \return normalized value preference value

        \see Factor.normalize(phi_result), Factor.phi(scope_product)

        """
        return self.phi(scope_product) / self.Z

    def partition_value(
        self, domains: List[FrozenSet[Tuple[str, NumericValue]]]
    ):
        """!
        \brief compute partition value aka normalizing value for the factor
        from Koller, Friedman 2009 p. 105
        For example given the following factors:

        \f[ P(a,b,c,d) = \frac{1}{Z} \phi_1(a,b) \cdot \phi_2(b,c) \cdot
        \phi_3(c, d) \cdot \phi_4(d, a) \f]

        The Z constant is the normalizing value also known as *partition
        function*. It is defined as the following:
        \f[Z = \sum_{a,b,c,d} \phi_1(a,b) \cdot \phi_2(b,c) \cdot
        \phi_3(c, d) \cdot \phi_4(d, a) \f]

        We basically sum every possible output for the joint distribution of
        given random variables.

        \param domains list of domain set of the involved random variables.

        \code{.py}

        >>> input_data = {
        >>>    "intelligence": {"outcome-values": [0.1, 0.9], "evidence": 0.9},
        >>>    "grade": {"outcome-values": [0.2, 0.4, 0.6], "evidence": 0.2},
        >>>    "dice": {"outcome-values": [i for i in range(1, 7)], "evidence": 1.0 / 6},
        >>>    "fdice": {"outcome-values": [i for i in range(1, 7)]},
        >>> }
        >>>
        >>> intelligence = NumCatRVariable(
        >>>     node_id="int",
        >>>     input_data=input_data["intelligence"],
        >>>     marginal_distribution=intelligence_dist,
        >>> )
        >>>
        >>> grade = NumCatRVariable(
        >>>     node_id=nid2, input_data=input_data["grade"],
        >>>     marginal_distribution=grade_dist
        >>> )
        >>>
        >>> dice = NumCatRVariable(
        >>>    node_id=nid3, input_data=input_data["dice"],
        >>>    marginal_distribution=fair_dice_dist
        >>> )
        >>>
        >>> f = Factor(
        >>>    gid="f", scope_vars=set([grade, dice, intelligence])
        >>> )
        >>>
        >>> pval = f.partition_value(f.vars_domain())
        >>> print(pval)
        >>> 1.0

        \endcode

        """
        scope_matches = list(product(*domains))
        return sum([self.factor_fn(scope_product=sv) for sv in scope_matches])

