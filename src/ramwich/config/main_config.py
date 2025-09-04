from pydantic import BaseModel, ConfigDict, Field, model_validator

from .archetecture.core_config import CoreConfig
from .archetecture.mvmu_config import MVMUConfig
from .archetecture.tile_config import TileConfig
from .data_config import DataConfig
from .hardware.noc_config import NOCConfig


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

    mvmu_configs: dict[int, MVMUConfig] = Field(
        default_factory=lambda: {
            0: MVMUConfig(mvmu_type=0),
            1: MVMUConfig(mvmu_type=1),
            2: MVMUConfig(mvmu_type=2),
            3: MVMUConfig(mvmu_type=3),
            4: MVMUConfig(mvmu_type=4),
            5: MVMUConfig(mvmu_type=5),
            6: MVMUConfig(mvmu_type=6),
            7: MVMUConfig(mvmu_type=7),
        }
    )

    @model_validator(mode="after")
    def validate_and_calculate(self):
        self.tile_config.edram_size = self.tile_config.edram_size_in_KB * 1024 * 8 // self.data_config.activation_width

        return self
