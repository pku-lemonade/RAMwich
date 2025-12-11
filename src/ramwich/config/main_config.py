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

    addr_width: int = Field(default=32, description="Address width")
    instrn_width: int = Field(default=48, description="Instruction width")

    # Add configuration for components with default factories
    data_config_list: dict[int, DataConfig] = Field(default_factory=dict)
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
        self.tile_config.edram_size = self.tile_config.edram_size_in_KB * 1024 * 8 // 16  # 16 for 16-bit activation

        # Normalize the data-config list to an int-keyed dictionary
        data_cfg_map: dict[int, DataConfig]
        if isinstance(self.data_config_list, list):  # Allow list-style config inputs
            data_cfg_map = dict(enumerate(self.data_config_list))
        else:
            data_cfg_map = {int(k): v for k, v in self.data_config_list.items()}
        object.__setattr__(self, "data_config_list", data_cfg_map)

        # Normalize mvmu_configs to an int-keyed dictionary and wire the proper data config per type
        mvmu_cfgs_raw = self.mvmu_configs
        if isinstance(mvmu_cfgs_raw, list):  # Allow list-style config inputs
            mvmu_cfg_map: dict[int, MVMUConfig] = {}
            for idx, cfg in enumerate(mvmu_cfgs_raw):
                key = cfg.mvmu_type if cfg.mvmu_type is not None else idx
                mvmu_cfg_map[int(key)] = cfg
        else:
            mvmu_cfg_map = {int(k): v for k, v in mvmu_cfgs_raw.items()}

        for mvmu_type, mvmu_cfg in mvmu_cfg_map.items():
            data_cfg = data_cfg_map.get(mvmu_type)
            if data_cfg is not None:
                mvmu_cfg.data_config = data_cfg.model_copy(deep=True)
                mvmu_cfg.calculate_derived_values()
        object.__setattr__(self, "mvmu_configs", mvmu_cfg_map)

        # Ensure the default mvmu_config matches the specialized config if available
        default_mvmu_cfg = self.mvmu_config
        if default_mvmu_cfg is not None:
            default_type = default_mvmu_cfg.mvmu_type
            specialized_cfg = mvmu_cfg_map.get(default_type)
            if specialized_cfg is not None:
                object.__setattr__(self, "mvmu_config", specialized_cfg)
            else:
                data_cfg = data_cfg_map.get(default_type)
                if data_cfg is not None:
                    default_mvmu_cfg.data_config = data_cfg.model_copy(deep=True)
                    default_mvmu_cfg.calculate_derived_values()
                    object.__setattr__(self, "mvmu_config", default_mvmu_cfg)

        return self
