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

    def run(self):
        """
        This method should be called as a SimPy process.
        """
        while True:
            op = yield self.input_buffer.get()

            if op is None:  # Check for termination signal
                if self.output_buffer:
                    # Propagate the termination signal
                    yield self.output_buffer.put(None)
                logger.debug(f"Stage {self.name} received termination signal")
                break  # Exit the process loop

            time = op.accept(self.visitor)
            yield self.env.timeout(time)

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

    def put(self, op):
        """Feed an operation into the pipeline."""
        self.first_stage_buffer.put(op)
