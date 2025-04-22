import logging
from typing import List, Optional

import simpy

from .visitor import CoreVisitor

logger = logging.getLogger(__name__)


class Stage:
    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        input_buffer: simpy.Store,
        output_buffer: Optional[simpy.Store],
        visitor: CoreVisitor,
    ):
        self.env = env
        self.name = name
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.visitor = visitor
        self.completion_event = self.env.event()

    def run(self):
        """
        This method should be called as a SimPy process.
        """
        while True:
            op = yield self.input_buffer.get()

            if op is None or op.type == "hlt":  # Check for termination signal
                if self.output_buffer:
                    # Propagate the termination signal
                    yield self.output_buffer.put(None)
                logger.debug(f"Stage {self.name} received termination signal")

                # Signal completion for this stage
                if not self.completion_event.triggered:
                    self.completion_event.succeed()

                break  # Exit the process loop

            result = op.accept(self.visitor)
            if isinstance(result, simpy.Event):
                # If the result is a simpy event, it marks completion
                yield result
            elif isinstance(result, int):
                # If the result is an integer, it indicates the latency for this stage
                time = result
                yield self.env.timeout(time)
            else:
                print(f"Stage {self.name} processing operation: {op}")
                raise ValueError(f"Unexpected result type: {type(result)}")

            if self.output_buffer:
                yield self.output_buffer.put(op)


class StageConfig:
    """Configuration for a single pipeline stage."""

    def __init__(self, name: str, visitor: CoreVisitor):
        self.name = name
        self.visitor = visitor


class Pipeline:
    """Encapsulates the setup and execution of a multi-stage pipeline."""

    def __init__(self, env: simpy.Environment, config: List[StageConfig]):
        self.env = env
        self.config = config
        self.stages: List[Stage] = []
        self.first_stage_buffer = simpy.Store(self.env, capacity=1)
        self.done_event = self.env.event()
        self._build()

    def _build(self):
        """Creates Stage instances based on config."""
        current_input_buffer = self.first_stage_buffer

        for i, stage_config in enumerate(self.config):
            name = stage_config.name
            visitor = stage_config.visitor

            output_buffer = None if i == len(self.config) - 1 else simpy.Store(self.env, capacity=1)

            stage = Stage(
                env=self.env, name=name, input_buffer=current_input_buffer, output_buffer=output_buffer, visitor=visitor
            )
            self.stages.append(stage)

            current_input_buffer = output_buffer

    def run(self):
        for stage in self.stages:
            self.env.process(stage.run())

        self._monitor_for_completion()

    def put(self, op):
        """Feed an operation into the pipeline."""
        self.first_stage_buffer.put(op)

    def _monitor_for_completion(self):
        """Wait for all stages to complete."""
        for stage in self.stages:
            yield stage.completion_event

        # Signal that the pipeline has completed
        if not self.done_event.triggered:
            self.done_event.succeed()

    def complete(self):
        """Get the event that signals when the pipeline has completed."""
        return self.done_event
