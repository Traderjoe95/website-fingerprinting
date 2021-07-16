from datetime import datetime, date, time
from typing import Union, List, Dict, Any, Optional, Type, TypeVar, Callable
import copy

import pytomlpp

ConfigPath = Union[str, int, List[int], List[str], List[Union[str, int]]]

ComplexItem = Union['ConfigList', 'ConfigObject']
ConfigValue = Union[int, str, float, bool, datetime, date, time, ComplexItem]

StdValue = Union[int, str, float, bool, datetime, date, time, List[Any], Dict[str, Any]]


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
        return {key: _to_stdlib(value) for key, value in self.__dict.items()}

    def as_obj(self, cls: Type[T], **kwargs) -> T:
        # noinspection Mypy
        return cls(**self.as_dict(), **kwargs)

    def merge_into(self, other: 'ConfigObject', *, other_is_override: bool = False) -> 'ConfigObject':
        m = _merge(self.as_dict(), other.as_dict()) if other_is_override else _merge(other.as_dict(), self.as_dict())
        return ConfigObject({key: _to_config_value(value) for key, value in m.items()})

    def _get_int_idx(self, idx: int) -> Optional[ConfigValue]:
        return self[str(idx)]

    def _get_str_idx(self, idx: str) -> Optional[ConfigValue]:
        if idx in self.__dict:
            return self.__dict[idx]
        else:
            return None

    def _as_config_value(self) -> Optional[ConfigValue]:
        return self


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
        return ConfigObject({key: _to_config_value(val) for key, val in value.items()})

    raise ValueError("Unrecognized input type " + str(type(value)))


def load() -> 'ConfigObject':
    with open('config.toml', 'r') as f:
        # noinspection PyTypeChecker
        _dict = __resolve(pytomlpp.load(f), ['config.toml'])

        return ConfigObject({key: _to_config_value(value) for key, value in _dict.items()})


def __resolve(_dict, resolving):
    if resolving is None:
        resolving = []

    included = None

    for key in _dict:
        if key == '__include__':
            val = _dict['__include__']

            if str(val) in resolving:
                raise RuntimeError('Encountered __include__ cycle:\n'
                                   f' {" -> ".join(resolving)} -> {val}')

            with open(val, 'r') as f:
                # noinspection PyTypeChecker
                included = __resolve(pytomlpp.load(f), resolving + [val])
        else:
            val = _dict[key]

            if isinstance(val, dict):
                _dict[key] = __resolve(val, resolving)
            elif isinstance(val, list) and any(isinstance(it, dict) for it in val):
                _dict[key] = [it if not isinstance(it, dict) else __resolve(it, resolving) for it in val]

    if included is not None:
        _dict = _merge(included, _dict)

    return _dict


def _merge(defaults, overrides):
    merged = copy.deepcopy(defaults)

    for key in overrides:
        if key not in merged:
            merged[key] = overrides[key]
        else:
            val = merged[key]
            override = overrides[key]

            if isinstance(val, dict) and isinstance(override, dict):
                merged[key] = _merge(val, override)
            else:
                merged[key] = override

    return merged


CONFIG = load()
