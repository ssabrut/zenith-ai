from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from core.config import Settings
from core.config import get_settings


@lru_cache
def load_settings() -> Settings:
    return get_settings()


SettingDependencies = Annotated[Settings, Depends(load_settings)]
