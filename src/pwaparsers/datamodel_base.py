"""
This file contains the data model, which will be used to parse the JSON input.
It is meant to be copied and adapted to produce a method code, which will produce the desired output in a specific framework.
DecaySetup is the top-level, which should have the method code. 
There is no immediate restrictions to also alter the other classes. 
"""

from pydantic import BaseModel, field_validator
from utils import to_tuple


class FinalStateParticle(BaseModel):
    name: str | int
    spin: int
    parity: int
    parityConserved: bool | None = None

class FinalState(BaseModel):
    finalStateData: dict[int, FinalStateParticle]
    nodes: list[int]

class Resonance(BaseModel):
    spin: int
    parity: int
    tuple: tuple[int, ...] | int
    name: str
    parityConserved: bool = True
    width: float | None = None
    mass: float | None = None

class Isobar(BaseModel):
    label: str
    tuple: tuple[int, ...]
    resonances: dict[str, Resonance]

    @field_validator('tuple', mode='before')
    @classmethod
    def validate_tuple(cls, v):
        if isinstance(v, list):
            return to_tuple(v)
        return v

class IntermdeiateState(BaseModel):
    isobars: dict[str, Isobar]

class DecaySetup(BaseModel):
    decay: int
    finalState: FinalState
    intermediateState: IntermdeiateState
