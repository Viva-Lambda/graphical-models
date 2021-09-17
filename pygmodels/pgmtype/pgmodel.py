"""!
\file pgmodel.py

# Probabilistic Graphic Model with Factors

This file contains the main model that drives the inference algorithms. It
extends the graph definition by adding a new set called a set of factors.
The set of factors can be arbitrarily defined or can be deduced from edges
using independence structure assumed by the model. The #PGModel is the most
generic model, hence we do not assume a particular independence structure.

"""
import math
from typing import Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pygmodels.factorf.factoranalyzer import FactorAnalyzer
from pygmodels.factorf.factorops import FactorOps
from pygmodels.gmodel.graph import Graph
from pygmodels.graphf.bgraphops import BaseGraphOps
from pygmodels.graphf.bgraphops import BaseGraphNodeOps
from pygmodels.graphf.bgraphops import BaseGraphEdgeOps
from pygmodels.graphf.bgraphops import BaseGraphBoolOps
from pygmodels.graphf.graphanalyzer import BaseGraphAnalyzer
from pygmodels.graphf.graphanalyzer import BaseGraphBoolAnalyzer
from pygmodels.graphf.graphanalyzer import BaseGraphNumericAnalyzer
from pygmodels.graphf.graphanalyzer import BaseGraphNodeAnalyzer
from pygmodels.graphf.graphops import BaseGraphAlgOps
from pygmodels.gtype.edge import Edge
from pygmodels.gtype.node import Node
from pygmodels.pgmtype.factor import Factor
from pygmodels.pgmtype.randomvariable import NumCatRVariable, NumericValue


def min_unmarked_neighbours(g: Graph, nodes: Set[Node], marked: Dict[str, Node]):
    """!
    \brief find an unmarked node with minimum number of neighbours
    """
    ordered = [(n, BaseGraphNumericAnalyzer.nb_neighbours_of(g, n)) for n in nodes]
    ordered.sort(key=lambda x: x[1])
    for X, nb in sorted(ordered, key=lambda x: x[1]):
        if marked[X.id()] is False:
            return X
    return None


class PGModel(Graph):
    """"""

    def __init__(
        self,
        gid: str,
        nodes: Set[NumCatRVariable],
        edges: Set[Edge],
        factors: Optional[Set[Factor]] = None,
        data={},
    ):
        """!
        \brief constructor for a generic Probabilistic Graphical Model

        The generic model that extends the #Graph definition by adding a new
        set, called set of factors.
        Most of the parameters are documented in #Graph.
        """
        super().__init__(gid=gid, data=data, nodes=nodes, edges=edges)
        if factors is None:
            fs: Set[Factor] = set()
            for e in self.E:
                estart = e.start()
                eend = e.end()
                sdata = estart.data()
                edata = eend.data()
                evidences = set()
                if "evidence" in sdata:
                    evidences.add((estart.id(), sdata["evidence"]))
                if "evidence" in edata:
                    evidences.add((eend.id(), edata["evidence"]))
                f = Factor(gid=str(uuid4()), scope_vars=set([estart, eend]))
                if len(evidences) != 0:
                    f = f.reduced_by_value(evidences)
                fs.add(f)
            self.Fs = fs
        else:
            self.Fs = factors

    def markov_blanket(self, t: NumCatRVariable) -> Set[NumCatRVariable]:
        """!
        get markov blanket of a node from K. Murphy, 2012, p. 662
        """
        if BaseGraphBoolOps.is_in(self, t) is False:
            raise ValueError("Node not in graph: " + str(t))
        ns: Set[NumCatRVariable] = BaseGraphNodeOps.neighbours_of(self, t)
        return ns

    def factors(self, f=lambda x: x):
        """!
        Get factors of graph
        """
        return set([f(ff) for ff in self.Fs])

    def closure_of(self, t: NumCatRVariable) -> Set[NumCatRVariable]:
        """!
        get closure of node
        from K. Murphy, 2012, p. 662
        """
        return set([t]).union(self.markov_blanket(t))

    def is_conditionaly_independent_of(
        self, n1: NumCatRVariable, n2: NumCatRVariable
    ) -> bool:
        """!
        check if two nodes are conditionally independent
        from K. Murphy, 2012, p. 662
        """
        return BaseGraphBoolAnalyzer.is_node_independent_of(self, n1, n2)

    def scope_of(self, phi: Factor) -> Set[NumCatRVariable]:
        """!"""
        return phi.scope_vars()

    def is_scope_subset_of(self, phi: Factor, X: Set[NumCatRVariable]) -> bool:
        """!
        filter factors using Koller, Friedman 2009, p. 299 as criteria
        """
        s: Set[NumCatRVariable] = self.scope_of(phi)
        return s.intersection(X) == s

    def scope_subset_factors(self, X: Set[NumCatRVariable]) -> Set[Factor]:
        """!
        choose factors using Koller, Friedman 2009, p. 299 as criteria
        """
        return set([f for f in self.factors() if self.is_scope_subset_of(f, X) is True])

    def get_factor_product(self, fs: Set[Factor]):
        """!
        Multiply a set of factors.
        \f \prod_{i} \phi_i \f
        """
        factors = list(fs)
        if len(factors) == 0:
            raise ValueError("Must have a non empty list of factors")
        if len(factors) == 1:
            return factors[0], None
        prod = factors.pop(0)
        for i in range(0, len(factors)):
            prod, val = FactorOps.cls_product(
                f=prod,
                other=factors[i],
                product_fn=lambda x, y: x * y,
                accumulator=lambda x, y: x * y,
            )
        return prod, val

    def get_factor_product_var(
        self, fs: Set[Factor], Z: NumCatRVariable
    ) -> Tuple[Factor, Set[Factor], Set[Factor]]:
        """!
        Get products of factors whose scope involves variable Z.
        """
        factors = set([f for f in fs if Z in self.scope_of(f)])
        other_factors = set([f for f in fs if f not in factors])
        prod, v = self.get_factor_product(factors)
        return prod, set(factors), other_factors

    def eliminate_variable_by(
        self,
        factors: Set[Factor],
        Z: NumCatRVariable,
        elimination_strategy=lambda x, y: x.sumout_var(y),
    ):
        """!
        eliminate variables using given strategy. Unites max product and sum
        product
        """
        (prod, scope_factors, other_factors) = self.get_factor_product_var(factors, Z)
        sum_factor = elimination_strategy(prod, Z)
        other_factors = other_factors.union({sum_factor})
        return other_factors, sum_factor, prod

    def sum_prod_var_eliminate(self, factors: Set[Factor], Z: NumCatRVariable):
        """!
        Koller and Friedman 2009, p. 298
        multiply factors and sum out the given variable
        \param factors factors that we are going to multiply
        \param Z variable that we are going to sum out, i.e. marginalize
        """
        res = self.eliminate_variable_by(
            factors=factors,
            Z=Z,
            elimination_strategy=lambda x, y: FactorOps.cls_sumout_var(x, y),
        )
        return res[0]

    def sum_product_elimination(
        self, factors: Set[Factor], Zs: List[NumCatRVariable]
    ) -> Factor:
        """!
        sum product variable elimination
        Koller and Friedman 2009, p. 298

        \param factors factor representation of our graph, it corresponds
        mostly to edges if other factors are not provided.

        \param Zs elimination variables. They correspond to all variables that
        are not query variables.
        """
        for Z in Zs:
            factors = self.sum_prod_var_eliminate(factors, Z)

        prod, v = self.get_factor_product(factors)
        return prod

    def order_by_max_cardinality(self, nodes: Set[NumCatRVariable]):
        """!
        from Koller and Friedman 2009, p. 312
        """
        marked = {n.id(): False for n in nodes}
        cardinality = {n.id(): -1 for n in nodes}
        unmarked_node_with_largest_marked_neighbor = None
        nb_marked_neighbours = float("-inf")
        for i in range(len(nodes)):
            for n in nodes:
                if marked[n.id()] is True:
                    continue
                nb_marked_neighbours_counter = 0
                for n_ in BaseGraphNodeOps.neighbours_of(self, n):
                    if marked[n_.id()] is False:
                        nb_marked_neighbours_counter += 1
                #
                if nb_marked_neighbours_counter > nb_marked_neighbours:
                    nb_marked_neighbours = nb_marked_neighbours_counter
                    unmarked_node_with_largest_marked_neighbor = n
            #
            cardinality[n.id()] = i
            marked[n.id()] = True
        #
        return cardinality

    def order_by_greedy_metric(
        self,
        nodes: Set[NumCatRVariable],
        s: Callable[
            [Graph, Dict[Node, bool]], Optional[Node]
        ] = min_unmarked_neighbours,
    ) -> Dict[str, int]:
        """!
        From Koller and Friedman 2009, p. 314
        """
        marked = {n.id(): False for n in nodes}
        cardinality = {n.id(): -1 for n in nodes}
        for i in range(len(nodes)):
            X = s(g=self, nodes=nodes, marked=marked)
            if X is not None:
                cardinality[X.id()] = i
                TEMP = BaseGraphNodeOps.neighbours_of(self, X)
                while TEMP:
                    n_x = TEMP.pop()
                    for n in BaseGraphNodeOps.neighbours_of(self, X):
                        self = BaseGraphAlgOps.added_edge_between_if_none(
                            self, n_x, n, is_directed=False
                        )
                marked[X.id()] = True
        return cardinality

    def reduce_queries_with_evidence(
        self, queries: Set[NumCatRVariable], evidences: Set[Tuple[str, NumericValue]],
    ) -> Set[NumCatRVariable]:
        """"""
        reduced_queries = set()
        evs = {e[0]: e[1] for e in evidences}
        for q in queries:
            if q.id() in evs:
                ev = evs[q.id()]
                q.reduce_to_value(ev)
            reduced_queries.add(q)
        return reduced_queries

    def reduce_factors_with_evidence(self, evidences: Set[Tuple[str, NumericValue]]):
        """!
        reduce factors if there is evidence
        """
        if len(evidences) == 0:
            return self.factors(), set()
        if any(e[0] not in {v.id() for v in self.V} for e in evidences):
            raise ValueError("evidence set contains variables out of vertices of graph")
        elist = [e[0] for e in evidences]
        E = set([v for v in self.V if v.id() in elist])
        fs = self.factors()
        factors = set(
            [FactorOps.cls_reduced_by_value(f, assignments=evidences) for f in fs]
        )
        return factors, E

    def cond_prod_by_variable_elimination(
        self,
        queries: Set[NumCatRVariable],
        evidences: Set[Tuple[str, NumericValue]],
        ordering_fn=min_unmarked_neighbours,
    ):
        """!
        Compute conditional probabilities with variable elimination
        from Koller and Friedman 2009, p. 304
        """
        if queries.issubset(self.V) is False:
            raise ValueError("Query variables must be a subset of vertices of graph")
        queries = self.reduce_queries_with_evidence(queries, evidences)
        factors, E = self.reduce_factors_with_evidence(evidences)
        Zs = set()
        for z in self.V:
            if z not in E and z not in queries:
                Zs.add(z)
        return self.conditional_prod_by_variable_elimination(
            queries=queries, Zs=Zs, factors=factors, ordering_fn=ordering_fn
        )

    def conditional_prod_by_variable_elimination(
        self,
        queries: Set[NumCatRVariable],
        Zs: Set[NumCatRVariable],
        factors: Set[Factor],
        ordering_fn=min_unmarked_neighbours,
    ) -> Tuple[Factor, Factor]:
        """!
        Main conditional product by variable elimination function
        """
        cardinality = self.order_by_greedy_metric(nodes=Zs, s=ordering_fn)
        V = {v.id(): v for v in self.V}
        ordering = [
            V[n[0]] for n in sorted(list(cardinality.items()), key=lambda x: x[1])
        ]
        phi = self.sum_product_elimination(factors=factors, Zs=ordering)
        alpha = FactorOps.cls_sumout_vars(phi, queries)
        return phi, alpha

    def max_product_eliminate_var(
        self, factors: Set[Edge], Z: NumCatRVariable
    ) -> Tuple[Set[Factor], Factor]:
        """!
        from Koller and Friedman 2009, p. 557
        """
        return self.eliminate_variable_by(
            factors=factors,
            Z=Z,
            elimination_strategy=lambda x, y: FactorOps.cls_maxout_var(x, y),
        )

    def max_product_eliminate_vars(self, factors: Set[Edge], Zs: List[NumCatRVariable]):
        """!
        from Koller and Friedman 2009, p. 557
        """
        Z_potential: List[Tuple[Factor, int]] = []
        for i in range(len(Zs)):
            Z = Zs[i]
            factors, maxed_out, z_phi = self.max_product_eliminate_var(factors, Z=Z)
            Z_potential.append(z_phi)
        #
        values = self.traceback_map(potentials=Z_potential, X_is=Zs)
        return values, factors, z_phi

    def max_product_ve(self, evidences: Set[Tuple[str, NumericValue]]):
        """!
        Compute most probable assignments given evidences
        """
        factors, E = self.reduce_factors_with_evidence(evidences)
        Zs = set()
        for z in self.V:
            if z not in E:
                Zs.add(z)
        cardinality = self.order_by_greedy_metric(nodes=Zs, s=min_unmarked_neighbours)
        V = {v.id(): v for v in self.V}
        ordering = [
            V[n[0]] for n in sorted(list(cardinality.items()), key=lambda x: x[1])
        ]
        assignments, factors, z_phi = self.max_product_eliminate_vars(
            factors=factors, Zs=ordering
        )
        return assignments, factors, z_phi

    def mpe_prob(self, evidences: Set[Tuple[str, NumericValue]]) -> float:
        """!
        obtain the probability of the most probable instantiation of
        the model
        """
        assignments, factors, z_phi = self.max_product_ve(evidences=evidences)
        probs = set()
        for f in z_phi.factor_domain():
            probs.add(z_phi.phi(f))
        return max(probs)

    def traceback_map(
        self, potentials: List[Factor], X_is: List[NumCatRVariable]
    ) -> List[Tuple[str, NumericValue]]:
        """!
        from Koller and Friedman 2009, p. 557
        The idea here is the following:
        For the last variable eliminated, Z, the factor for the value x
        contains the probability of the most likely assignment that contains
        Z=x.
        For example:
        let's say g* = argmax(psi(G))
        2. l* = argmax(psi[g*](L))
        3. d* = argmax(psi[l*](D))
        """
        max_assignments = {}
        for i in range(len(potentials) - 1, -1, -1):
            pmax = FactorAnalyzer.cls_max_value(potentials[i])
            diff = set([p for p in pmax if p[0] not in max_assignments])
            max_assign = diff.pop()
            max_assignments[max_assign[0]] = max_assign[1]
        return max_assignments
