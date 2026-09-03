from abc import ABC

class OptimizerBase(ABC):
    def __init__(self, params) -> None:
        pass

    def step(self):
        pass