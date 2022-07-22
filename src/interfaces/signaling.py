from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy import signal


class SignalModel:

    class VariableConfig:
        pass


@dataclass
class SignalVariable:
    interval: list
    step_size: Union[int, float]


class StepSignal(SignalModel):

    class VariableConfig:
        amplitude = SignalVariable([0, 500], 0.1)
        step_time = SignalVariable([0, 100000], 1)

    def __init__(self, t_start, amplitude, phase):

        self.amplitude = amplitude
        self.step_time = t_start + phase

    def __call__(self, t, t_mod):
        if t < self.step_time:
            return 0
        else:
            return self.amplitude


class PulseSignal(SignalModel):

    class VariableConfig:
        amplitude = SignalVariable([0, 500], 0.1)
        period = SignalVariable([1, 1000], 1)
        duty = SignalVariable([0, 1], 0.01)

    def __init__(self, amplitude, phase, frequency, pulse_length):
        self.amplitude = amplitude
        self.period = int(1000 / frequency)
        self.duty = round(pulse_length/self.period, 2)
        self.phase = phase

    def __call__(self, t, t_mod):
        return (signal.square(((t + self.phase) * 2 * np.pi)/self.period, self.duty) * self.amplitude / 2
                + self.amplitude / 2)
