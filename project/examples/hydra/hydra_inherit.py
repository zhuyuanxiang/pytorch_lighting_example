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
    user: str = MISSING
    password: str = MISSING


@dataclass
class PostGreSQLConfig(DBConfig):
    driver: str = "postgresql"
    user: str = MISSING
    port: int = 5432
    password: str = MISSING
    timeout: int = 10


@dataclass
class Config:
    # We can now annotate db as DBConfig which
    # improves both static and dynamic type safety.
    db: DBConfig = MISSING
    debug: bool = False


@dataclass
class AConfig(Config):
    db = PostGreSQLConfig()
    tmp_value = 1


@dataclass
class BConfig:  # 继承不能在最终类上使用，否则无法初始化。
    a_config: Config = AConfig()


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
# cs.store(name="config_inherit", node=AConfig)
cs.store(name="config_inherit", node=BConfig)
cs.store(group="db", name="base_mysql", node=MySQLConfig)
cs.store(group="db", name="base_postgresql", node=PostGreSQLConfig)


@hydra.main(version_base=None, config_path='conf', config_name="config")
def my_app(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
