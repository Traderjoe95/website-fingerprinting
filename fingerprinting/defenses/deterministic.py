from abc import ABCMeta
from typing import List, Type

import numpy as np
import pandas as pd

from ..api import StatelessDefense, Defense
from ..api.typing import Traces


def plugin_path() -> str:
    return 'core.defenses.deterministic'


def defenses() -> List[Type[Defense]]:
    return [LinearPadding, ExponentialPadding, MiceElephantsPadding, PadToMTU]


class DigitizingPadding(StatelessDefense, metaclass=ABCMeta):
    def __init__(self, steps: np.ndarray = np.array([])):
        self._steps = steps

    def defend(self, traces: Traces) -> Traces:
        direction = np.clip(traces["size"].values, -1, 1)
        idxs = np.digitize(np.abs(traces["size"]), self.__steps, right=True)

        traces["size"] = self.__steps[idxs] * direction

        return traces

    @property
    def _steps(self) -> np.ndarray:
        return self.__steps

    @_steps.setter
    def _steps(self, steps: np.ndarray):
        self.__steps = steps


class LinearPadding(DigitizingPadding):
    def __init__(self, increment: int = 128):
        super(LinearPadding, self).__init__()
        self.increment = increment

    @staticmethod
    def name() -> str:
        return "linear-padding"

    @property
    def increment(self) -> int:
        return self.__increment

    @increment.setter
    def increment(self, increment: int):
        if increment < 1 or increment > 1500:
            raise ValueError("The step size must be between 1 and 1500")

        self.__increment = increment
        self._steps = np.append(np.arange(0, 1500, increment), 1500)

    def get_params(self):
        return {'increment': self.increment}

    def set_params(self, **params):
        if 'increment' in params:
            self.increment = int(params['increment'])


class ExponentialPadding(DigitizingPadding):
    def __init__(self):
        super(ExponentialPadding, self).__init__(
            np.append(np.append(0, 2 ** np.arange(0, int(np.log2(1500)) + 1)), 1500))

    @staticmethod
    def name() -> str:
        return "exponential-padding"

    def get_params(self):
        return {}

    def set_params(self, **params):
        pass


class MiceElephantsPadding(DigitizingPadding):
    def __init__(self):
        super(MiceElephantsPadding, self).__init__(np.array([0, 128, 1500]))

    @staticmethod
    def name() -> str:
        return "mice-elephants"

    def get_params(self):
        return {}

    def set_params(self, **params):
        pass


class PadToMTU(DigitizingPadding):
    def __init__(self):
        super(PadToMTU, self).__init__(np.array([0, 1500]))

    @staticmethod
    def name() -> str:
        return "pad-to-mtu"

    def get_params(self):
        return {}

    def set_params(self, **params):
        pass
