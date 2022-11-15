"""
=================================================
@path   : pytorch_lighting_example -> hydra_inherit
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/15 16:47
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from omegaconf import OmegaConf


@dataclass
class DBConfig:
    host: str = "localhost"
    port: int = MISSING
    driver: str = MISSING


@dataclass
class MySQLConfig(DBConfig):
    driver: str = "mysql"
    port: int = 3306


@dataclass
class PostGreSQLConfig(DBConfig):
    driver: str = "postgresql"
    port: int = 5432
    timeout: int = 10


@dataclass
class Config:
    # We can now annotate db as DBConfig which
    # improves both static and dynamic type safety.
    tmp_value: int = MISSING
    db: DBConfig = MISSING


@dataclass
class AConfig(Config):
    db = PostGreSQLConfig()
    tmp_value = 1


@dataclass
class BConfig:  # 继承不能在最终类上使用，否则无法初始化。
    a_config: Config = AConfig()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
# cs.store(name="config_inherit", node=AConfig)
cs.store(name="config_inherit", node=BConfig)
cs.store(group="db", name="mysql", node=MySQLConfig)
cs.store(group="db", name="postgresql", node=PostGreSQLConfig)


@hydra.main(version_base=None, config_name="config_inherit")
def my_app(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
