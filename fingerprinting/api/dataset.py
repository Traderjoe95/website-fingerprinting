from typing import Optional


class DatasetConfig:
    def __init__(self,
                 default_name: str,
                 name: Optional[str] = None,
                 module: Optional[str] = None,
                 path: Optional[str] = None):
        if name is None:
            name = default_name

        self.name: str = name
        self.module: str = module or f".datasets.{self.name}"
        self.path: str = path or f"data/{self.name}"
