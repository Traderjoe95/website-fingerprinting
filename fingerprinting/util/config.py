from datetime import datetime, date, time
from typing import Union, List, Dict, Any, Optional, Type, TypeVar, Callable

import pytomlpp

ConfigPath = Union[str, int, List[int], List[str], List[Union[str, int]]]

ComplexItem = Union['ConfigList', 'ConfigObject']
ConfigValue = Union[int, str, float, bool, datetime, date, time, ComplexItem]

StdValue = Union[int, str, float, bool, datetime, List[Any], Dict[str, Any]]


class ConfigBase:
    def get(self, path: ConfigPath, default: Optional[ConfigValue] = None) -> ConfigValue:
        val = self[path]

        if val is not None:
            return val
        elif default is not None:
            return default
        else:
            raise ValueError(f"No configuration setting with key {path} found")

    def get_str(self, path: ConfigPath, default: Optional[str] = None) -> str:
        return str(self.get(path, default))

    def get_int(self, path: ConfigPath, default: Optional[int] = None) -> int:
        return int(self.get(path, default))

    def get_float(self, path: ConfigPath, default: Optional[float] = None) -> float:
        return float(self.get(path, default))

    def get_bool(self, path: ConfigPath, default: Optional[bool]) -> bool:
        return bool(self.get(path, default))

    def get_list(self, path: ConfigPath, default: Optional['ConfigList'] = None) -> 'ConfigList':
        if default is None:
            default = ConfigList([])

        val = self.get(path, default)

        if isinstance(val, ConfigList):
            return val
        else:
            raise ValueError(f"Not a list: {val}")

    def get_obj(self, path: ConfigPath, default: Optional['ConfigObject'] = None) -> 'ConfigObject':
        if default is None:
            default = ConfigObject({})

        val = self.get(path, default)

        if isinstance(val, ConfigObject):
            return val
        else:
            raise ValueError(f"Not an object: {val}")

    def __contains__(self, path: ConfigPath) -> bool:
        return self[path] is not None

    def __getitem__(self, path: ConfigPath) -> Optional[ConfigValue]:
        if isinstance(path, int):
            return self._get_int_idx(path)
        elif isinstance(path, str) and '.' not in path:
            return self._get_str_idx(path)
        elif isinstance(path, str):
            return self[path.split('.')]
        elif len(path) == 0:
            return self._as_config_value()
        else:
            if path[0] not in self:
                return None

            val = self[path[0]]

            if len(path) == 1:
                return val
            elif isinstance(val, ConfigList) or isinstance(val, ConfigObject):
                return val[path[1:]]
            else:
                return None

    def _get_int_idx(self, idx: int) -> Optional[ConfigValue]:
        ...

    def _get_str_idx(self, idx: str) -> Optional[ConfigValue]:
        ...

    def _as_config_value(self) -> Optional[ConfigValue]:
        ...


T = TypeVar('T', covariant=True)


class ConfigList(ConfigBase):
    def __init__(self, __list: List[ConfigValue]):
        self.__list = __list

    def __iter__(self):
        return iter(self.__list)

    def __len__(self):
        return len(self.__list)

    def __repr__(self):
        return repr(self.__list)

    def __str__(self):
        return str(self.__list)

    def as_list(self) -> List[StdValue]:
        return [_to_stdlib(value) for value in self.__list]

    def find(self, predicate: Callable[[ConfigValue], bool]) -> Optional[ConfigValue]:
        for obj in self:
            if predicate(obj):
                return obj

        return None

    def find_obj(self, predicate: Callable[['ConfigObject'], bool]) -> Optional['ConfigObject']:
        # noinspection Mypy
        return self.find(lambda obj: isinstance(obj, ConfigObject) and predicate(obj))

    def _get_int_idx(self, idx: int) -> Optional[ConfigValue]:
        return self.__list[idx] if len(self) > idx else None

    def _get_str_idx(self, idx: str) -> Optional[ConfigValue]:
        try:
            return self[int(idx)]
        except ValueError:
            return None

    def _as_config_value(self) -> Optional[ConfigValue]:
        return self


class ConfigObject(ConfigBase):
    def __init__(self, __dict: Dict[str, ConfigValue]):
        self.__dict = __dict

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)

    def __repr__(self):
        return repr(self.__dict)

    def __str__(self):
        return str(self.__dict)

    def as_dict(self):
        return self.__dict

    def as_obj(self, cls: Type[T], **kwargs) -> T:
        # noinspection Mypy
        return cls(**self.as_dict(), **kwargs)

    def _get_int_idx(self, idx: int) -> Optional[ConfigValue]:
        return self[str(idx)]

    def _get_str_idx(self, idx: str) -> Optional[ConfigValue]:
        if idx in self.__dict:
            return self.__dict[idx]
        else:
            return None

    def _as_config_value(self) -> Optional[ConfigValue]:
        return super()._as_config_value()


def _to_stdlib(value: ConfigValue) -> StdValue:
    if isinstance(value, str) or isinstance(value, int) or isinstance(value, float) or isinstance(value, bool):
        return value
    elif isinstance(value, datetime) or isinstance(value, date) or isinstance(value, time):
        return value
    elif isinstance(value, ConfigList):
        return value.as_list()
    elif isinstance(value, ConfigObject):
        return value.as_dict()

    raise ValueError("Unrecognized input type " + str(type(value)))


def _to_config_value(value: StdValue) -> ConfigValue:
    if isinstance(value, str) or isinstance(value, int) or isinstance(value, float) or isinstance(value, bool):
        return value
    elif isinstance(value, datetime) or isinstance(value, date) or isinstance(value, time):
        return value
    elif isinstance(value, list):
        return ConfigList([_to_config_value(val) for val in value])
    elif isinstance(value, dict):
        return ConfigObject(dict((key, _to_config_value(value[key])) for key in value))

    raise ValueError("Unrecognized input type " + str(type(value)))


def load() -> 'ConfigObject':
    with open('config.toml', 'r') as f:
        # noinspection PyTypeChecker
        _dict = pytomlpp.load(f)

        return ConfigObject(dict((key, _to_config_value(_dict[key])) for key in _dict))


CONFIG = load()
