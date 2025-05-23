import math
from typing import ClassVar
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .data_config import BitConfig, DataConfig
from .hardware.noc_config import NOCConfig
from .archetecture.tile_config import TileConfig
from .archetecture.core_config import CoreConfig
from .archetecture.mvmu_config import MVMUConfig


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)
    """Configuration for the RAMwich Simulator"""
    num_nodes: int = Field(default=1, description="Number of nodes in the system")
    num_tiles_per_node: int = Field(default=4, description="Number of tiles per node")
    num_cores_per_tile: int = Field(default=8, description="Number of cores per tile")
    num_mvmus_per_core: int = Field(default=6, description="Number of MVMUs per core")

    # Add configuration for components with default factories
    data_config: DataConfig = Field(default_factory=DataConfig)
    noc_config: NOCConfig = Field(default_factory=NOCConfig)
    tile_config: TileConfig = Field(default_factory=TileConfig)
    core_config: CoreConfig = Field(default_factory=CoreConfig)
    mvmu_config: MVMUConfig = Field(default_factory=MVMUConfig)

    @model_validator(mode="after")
    def validate_and_calculate(self):
        self.mvmu_config.stored_bit = []
        self.mvmu_config.bits_per_cell = []
        self.mvmu_config.is_xbar_rram = []
        self.mvmu_config.rram_to_output_map = []
        self.mvmu_config.sram_to_output_map = []

        bits = 0  # total bits number in the operand
        self.mvmu_config.num_rram_xbar_per_mvmu = 0  # number of RRAM xbars
        self.mvmu_config.num_sram_xbar_per_mvmu = 0  # number of SRAM xbars
        for i in self.data_config.storage_config:
            self.mvmu_config.stored_bit.append(bits)
            if i == BitConfig.SRAM:
                self.mvmu_config.num_sram_xbar_per_mvmu += 1
                self.mvmu_config.bits_per_cell.append(1)
                self.mvmu_config.is_xbar_rram.append(False)
                bits += 1
                self.mvmu_config.have_sram_xbar = True
            else:
                self.mvmu_config.num_rram_xbar_per_mvmu += 1
                self.mvmu_config.bits_per_cell.append(int(i))
                self.mvmu_config.is_xbar_rram.append(True)
                bits += int(i)
                self.mvmu_config.have_rram_xbar = True
        self.mvmu_config.stored_bit.append(bits)
        assert bits == self.data_config.weight_width, (
            "storage config invalid: check if total bits in storage config = weight width"
        )

        self.mvmu_config.num_xbar_per_mvmu = (
            self.mvmu_config.num_sram_xbar_per_mvmu + self.mvmu_config.num_rram_xbar_per_mvmu
        )

        # Assign output maps based on xbar type
        for i in range(self.mvmu_config.num_xbar_per_mvmu):
            if self.mvmu_config.is_xbar_rram[i]:
                self.mvmu_config.rram_to_output_map.append(i)
            else:
                self.mvmu_config.sram_to_output_map.append(i)

        self.tile_config.edram_size = self.tile_config.edram_size_in_KB * 1024 * 8 // self.data_config.activation_width

        return self
