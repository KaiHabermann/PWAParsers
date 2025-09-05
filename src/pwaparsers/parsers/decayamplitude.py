from pydantic import BaseModel, field_validator
from typing import Any
from pwaparsers.utils import flat_sorted_tuple, to_tuple
from decayangle.decay_topology import TopologyCollection

def sanitize(name: str) -> str:
    """
    Sanitize a name for use in python code
    """
    replacements = [
        ("*", "star"),
        ("(", ""),
        (")", ""),
        ("[", ""),
        ("]", ""),
        ("{", ""),
        ("}", ""),
    ] + [
        (a, "_") for a in " -+.,/\\^'\"~!?=>|&$#@%;:`´§°"
    ]
    for replacement in replacements:
        name = name.replace(*replacement)
    
    return name

def tuple_recurse(tpl):
    if isinstance(tpl, tuple):
        yield tpl
        for item in tpl:
            yield from tuple_recurse(item)

class FinalStateParticle(BaseModel):
    name: str | int
    spin: int
    parity: int
    parityConserved: bool | None = None

    @property
    def quoted_name(self):
        return "\"" + self.name + "\""

class FinalState(BaseModel):
    finalStateData: dict[int, FinalStateParticle]
    nodes: list[int]

    def code(self):
        finalStateData = self.finalStateData.copy()
        finalStateData.pop(0, None)
        return f"""
final_state_qn = {{
    {('\n' + 4*' ').join([f'{i}: Particle(spin={p.spin}, parity={p.parity}, name={p.quoted_name}), ' for i, p in finalStateData.items()])}
}}
setup = DecaySetup(final_state_particles=final_state_qn)
topologies = [
    setup.symmetrize(topology) for topology in topologies
]
"""
    def mother_code(self):
        mother = self.finalStateData[0]
        mother_resonance = Resonance(
            spin=mother.spin,
            parity=mother.parity,
            tuple=0,
            name=mother.name,
            parityConserved=mother.parityConserved,
            width=None,
            mass=None
        )
        return mother_resonance.code(lineshape=ConstantLineshape)

class BWLineshape:
    def __init__(self, tuple:tuple, base_name: str):
        self.tuple = tuple
        self.base_name = base_name
    
    def code(self):
        return f"""lineshape=BW_lineshape(mass_from_node(Node({self.tuple}), momenta)), argnames=['{sanitize(self.base_name)}_mass', '{sanitize(self.base_name)}_width']"""

class ConstantLineshape:
    def __init__(self, tuple:tuple, base_name: str):
        self.tuple = tuple
        self.base_name = base_name
    
    def code(self):
        return f"""lineshape=constant_lineshape, argnames=[]"""

class Resonance(BaseModel):
    spin: int
    parity: int
    tuple: tuple[Any, ...] | int
    name: str
    parityConserved: bool = True
    width: float | None = None
    mass: float | None = None

    @field_validator('tuple', mode='before')
    @classmethod
    def validate_tuple(cls, v):
        if isinstance(v, list):
            return to_tuple(v)
        return v
    
    @field_validator('parityConserved', mode='before')
    @classmethod
    def validate_parity_conserved(cls, v):
        if v is None:
            return True
        return bool(v)

    def code(self, lineshape=BWLineshape):
        ls = lineshape(self.tuple, self.name)
        parity = {True: "True", False: "False"}[self.parityConserved]
        return f"Resonance(Node({self.tuple}), quantum_numbers=QN({self.spin}, {self.parity}), {ls.code()}, preserve_partity={parity}, name='{self.name}')"

class Check(BaseModel):
    resonances: dict[str, Resonance]
    topology: str
    check: bool
    failed: list[str]

class FullCheck(BaseModel):
    trees: list[Check]
    partial: bool = True
    full: bool = True

class Isobar(BaseModel):
    label: str
    tuple: tuple[Any, ...]
    resonances: dict[str, Resonance]

    @field_validator('tuple', mode='before')
    @classmethod
    def validate_tuple(cls, v):
        if isinstance(v, list):
            return to_tuple(v)
        return v

    def code(self):
        if len(self.resonances) == 0:
            return ""
        return f"""
        {tuple(self.tuple)}: [
        {(',\n' + 12*' ').join([resonance.code() for resonance in self.resonances.values()])}
        ]"""

class IntermdeiateState(BaseModel):
    isobars: dict[str, Isobar]

    def resonance_list_code(self, mother_code):
        isobars = ",\n".join(isobar.code() for isobar in self.isobars.values() if isobar.code())
        return f"""
    resonances = {{
        0:    [
            {mother_code} 
            # Mother particle, set the QN to what they need to be for your decay. Usually a weak initial decay is assumed, so parity is not preserved. If this is not the case set it to True here or in the frontend.
                    # Replace the lineshape with a function of your liking, the call signatrue has to be f(L, S, *args), where the args will be exposed under the names you give in argnames
                    # The mass points at which to evaluate the lineshape are not given at runtime. Look at the bw_lineshpe for details
            ],
        {isobars}
    }}
"""
    @property
    def filled_nodes(self):
        filled_nodes = set()
        for tpl, isobar in self.as_dict.items():
            if len(isobar.resonances) > 0:
                filled_nodes.add(flat_sorted_tuple(tpl))
        return filled_nodes

    @property
    def as_dict(self):
        return {isobar.tuple: isobar for isobar in self.isobars.values()}

class DecaySetup(BaseModel):
    decay: int
    finalState: FinalState
    intermediateState: IntermdeiateState

    def import_header(self):
        return """
# This file was generated by decaytreeedit-backend
# Copyright © 2025 Kai Habermann

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# The Author(s) shall be credited in any publication or presentation that uses this software, including but not limited to conference presentations, journal articles, and theses. The Authors shall be credited by reference to the decayangle and decayamplitude repositories, as well as the citation of the original paper: https://doi.org/10.1103/PhysRevD.111.056015

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 

from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN
from decayamplitude.chain import MultiChain
from decayamplitude.combiner import ChainCombiner
from decayamplitude.particle import Particle, DecaySetup
from decayamplitude.backend import numpy as np
from decayangle.decay_topology import Topology, Node
from decayangle.config import config as decayangle_config
from decayamplitude.kinematics_helpers import mass_from_node
decayangle_config.backend = "jax"
decayangle_config.sorting = "value"  # this makes sure, that the order of entries in a node is handeled internaly and we do not have to worry about it

from jax import jit, grad

def constant_lineshape(L, S, *args):
    #This is a dummy lineshape function, it does nothing, but is needed for the code to work
    # It is suited to model a mother resonance for example
    return 1.0

def BW_lineshape(mass):
    # usually the function is called for a fixed set of points in the phasespace. 
    # Thus we can give the mass values as fixed
    # This allows the compiler to optimize the function better
    # decayamplitude will provide l and s values per default for the resonances, but not the mass values
    def BW(l,s, m0, gamma):
        return 1 / (mass**2 - m0**2 + 1j * mass * gamma) 
    return BW

"""

    def needed_topologies(self):
        topologies = TopologyCollection(self.decay, self.finalState.nodes)
        needed = []

        for topolgy in topologies.topologies:
            contained_daughters = [item for item in tuple_recurse(topolgy.tuple)][1:]
            if all(flat_sorted_tuple(item) in self.intermediateState.filled_nodes for item in contained_daughters):
                needed.append(topolgy)
        return needed
    
    def topology_header(self):
        needed_topologies = self.needed_topologies()
        def topo_code(tpl):
            return f"Topology(0, decay_topology={tpl})"
        header = (",\n" + 8*' ').join([topo_code(topo.tuple) for topo in needed_topologies])
        return f"""
topologies = [
        {header}
    ]
        """
    
    def chain_combiner_header(self):
        return """
def amplitude(momenta):
    # momenta has the form of 
    # {
    #  1: np.array(shape=(...,4)), with indices 0,1,2,3 for p_x,p_y,p_z,E
    #  2: np.array(shape=(...,4)), with indices 0,1,2,3 for p_x,p_y,p_z,E
    #  ...
    #}
"""

    def chain_combiner_code(self):
        return """

    chains = [
        MultiChain(
            topology=topology,
            resonances = resonances,
            momenta = momenta,
            final_state_qn = final_state_qn
        ) for topology in topologies
    ]
    full = ChainCombiner(chains)

    # The unpolarized amplitude is the simplest one, and the default case in LHCb
    unpolarized, argnames = full.unpolarized_amplitude(
        full.generate_couplings() # This is a helper function to generate the couplings for the hadronic system, if you want to restrict them, you will have to do it manually.
                                    # Alternatively you can also restrict the couplings in the fitter later.      
        )
    # argnames are the names of the arguments of the function, which are the masses and widths of the resonances and the couplings
    # The order of argnames is the order of the arguments in the function
    # The function can be called with positional arguments, or with keyword arguments
    # so unpolarized(*[1, 2, 3, 4, 5, 6]) is the same as unpolarized(mass_resonance_1=1, width_resonance_1=2, mass_resonance_2=3, width_resonance_2=4, mass_resonance_3=5, width_resonance_3=6), omitting the couplings
    print(argnames)

    # an issue with jax, where the internal caching structure needs to be prewarmed, so that in the compilation step the correct types are inferred
    print(unpolarized(*([1.0] * len(argnames))))

    # we can now jit the function, to make it faster after the compile
    unpolarized = jit(unpolarized) 
    print(unpolarized(*([1.0] * len(argnames))))


    # for the gradient calculation we need to define a log likelihood function or something, that produces a single value
    def LL(*args):
        return np.sum(
            np.log(unpolarized(*args))
                )
    # we can calc the gradient of the log likelihood function
    unpolarized_grad = jit(grad(LL, argnums=[i for i in range(len(argnames))]))

    # and a test call (may take quite some time)
    # print(unpolarized_grad(*([1.0] * len(argnames))))

    # Further calls will be much faster, since we dont need to compile again
    # print(unpolarized_grad(*([1.0] * len(argnames))))
    # print(unpolarized_grad(*([2.0] * len(argnames))))


    # Other options for amplitudes, one might be interested in
    # polarized, lambdas ,polarized_argnames = full.polarized_amplitude(full.generate_couplings())
    # print(lambdas)
    # print(polarized(*lambda_values,*([1] * len(polarized_argnames))) )

    # matrix_function, matrix_argnames = full.matrix_function(full.generate_couplings())
    # print(matrix_argnames)
    # print(matrix_function(0, *([1] * len(argnames))) )

    return full
"""
    def run_with_phsp(self):
        return """
if __name__ == "__main__":
    mother_mass = 1
    daughter_masses = [0.1] * len(final_state_qn)
    from decayangle.kinematics import from_mass_and_momentum
    import random 

    random_angles = [
        (random.uniform(0, 2 * np.pi) , random.uniform(0, np.pi)) 
         for _ in range(len(final_state_qn))
         ]

    three_momenta = [
        np.array([0.1,0,0]) * np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])
        for theta, phi in random_angles
    ]

    momenta = {
        i: from_mass_and_momentum(m, p)
            for i, (m, p) in enumerate(zip(daughter_masses, three_momenta), start=1)
    }
    momenta = topologies[0].to_rest_frame(momenta)
    full = amplitude(momenta)
"""
    def code(self):
        decay_code = self.import_header()
        decay_code += self.topology_header()
        decay_code += self.finalState.code()
        decay_code += self.chain_combiner_header()
        decay_code += self.intermediateState.resonance_list_code(self.finalState.mother_code())
        decay_code += self.chain_combiner_code()
        decay_code += self.run_with_phsp()
        return decay_code
