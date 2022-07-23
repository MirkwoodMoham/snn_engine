from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from scipy import signal


class SignalModel:

    class VariableConfig:
        pass


@dataclass
class SignalVariable:
    interval: list
    step_size: Union[int, float]
    unit: Optional[str] = None


@dataclass
class AmplitudeVariable(SignalVariable):
    interval: list = field(default_factory=lambda: [0, 500])
    step_size: float = 0.1
    unit: str = 'pA'


@dataclass
class StepTimeVariable(SignalVariable):
    interval: list = field(default_factory=lambda: [0, 100000])
    step_size: int = 1
    unit: str = 'ms'


class StepSignal(SignalModel):

    class VariableConfig:
        amplitude = AmplitudeVariable()
        step_time = StepTimeVariable()

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
        amplitude = AmplitudeVariable()
        period = SignalVariable([1, 1000], 1, 'ms')
        duty = SignalVariable([0, 1], 0.01, 'ms')

    def __init__(self, amplitude, phase, frequency, pulse_length):
        self.amplitude = amplitude
        self.period = int(1000 / frequency)
        self.duty = round(pulse_length/self.period, 2)
        self.phase = phase

    def __call__(self, t, t_mod):
        return (signal.square(((t + self.phase) * 2 * np.pi)/self.period, self.duty) * self.amplitude / 2
                + self.amplitude / 2)


class DiscretePulseSignal(SignalModel):

    class VariableConfig:
        amplitude = AmplitudeVariable()
        period = SignalVariable([1, 1000], 1, 'ms')
        duty_period = SignalVariable([0, 1000], 1, 'ms')
        phase = SignalVariable([0, 1000], 0.01, 'ms')

    def __init__(self, amplitude, phase, frequency, pulse_length):
        self.amplitude = amplitude
        self.period = int(1000 / frequency)
        self.duty_period = pulse_length
        self.phase = phase

    def __call__(self, t, t_mod):
        return (((t + self.phase) % self.period) < self.duty_period) * self.amplitude
