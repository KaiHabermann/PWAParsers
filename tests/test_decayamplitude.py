import pytest
import json
from pwaparsers.parsers.decayamplitude import DecaySetup as DecayAmplitudeParser


@pytest.fixture
def load_test_data():
    with open('tests/test_data/jpsikplpipi.json') as f:
        data = json.load(f)
    return data


def test_decayamplitude(load_test_data):
    data = load_test_data
    decay_setup = DecayAmplitudeParser(**data)
    code = decay_setup.code()
