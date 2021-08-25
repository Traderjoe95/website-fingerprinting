import re
from abc import ABCMeta, abstractmethod
from importlib import import_module
from logging import getLogger
from typing import Iterable, Optional, Any, Dict, Type, List, Set, Union

from click import Group
from typing_extensions import Protocol, runtime_checkable

from .pipeline import Dataset, Defense, AttackDefinition

_logger = getLogger(__name__)


@runtime_checkable
class Plugin(Protocol, metaclass=ABCMeta):
    def plugin_path(self) -> str:
        ...

    def __str__(self) -> str:
        return self.plugin_path()

    def __repr__(self) -> str:
        return self.plugin_path()


class CliModule:
    def __init__(self,
                 *,
                 name: str,
                 top_level_commands: Optional[Iterable[Any]] = None,
                 dataset_commands: Optional[Iterable[Any]] = None):
        if top_level_commands is None:
            top_level_commands = []
        if dataset_commands is None:
            dataset_commands = []

        self.__name = name

        self.__main = list(top_level_commands)
        self.__dataset = list(dataset_commands)

    def name(self) -> str:
        return self.__name

    def install(self, main_group, dataset_group):
        for cmd in self.__main:
            main_group.add_command(cmd)

        for cmd in self.__dataset:
            dataset_group.add_command(cmd)

    @staticmethod
    def load(module_name: str) -> Optional['CliModule']:
        if module_name.startswith("."):
            mod = import_module(module_name, "fingerprinting")
        else:
            mod = import_module(module_name)

        if hasattr(mod, "cli_plugin"):
            # noinspection PyUnresolvedReferences,Mypy
            return getattr(mod, "cli_plugin")()
        else:
            return None


@runtime_checkable
class CliPlugin(Plugin, Protocol):
    def cli_module(self) -> CliModule:
        ...


@runtime_checkable
class DatasetPlugin(Plugin, Protocol):
    def datasets(self) -> List[Type[Dataset]]:
        ...


@runtime_checkable
class DefensePlugin(Plugin, Protocol):
    def defenses(self) -> List[Type[Defense]]:
        ...


@runtime_checkable
class AttackPlugin(Plugin, Protocol):
    def attacks(self) -> List[Type[AttackDefinition]]:
        ...


class PluginError(metaclass=ABCMeta):
    @property
    @abstractmethod
    def message(self) -> str:
        ...


class RetryableError(PluginError, metaclass=ABCMeta):
    pass


class InvalidPluginName(PluginError):
    def __init__(self, plugin_name: str, reason: str):
        self.__name = plugin_name
        self.__reason = reason

    @property
    def message(self) -> str:
        return f"Invalid plugin name '{self.__name}': {self.__reason}"


class PluginNotFound(RetryableError):
    def __init__(self, plugin_name: str, item: str, path: str, subtree: Set[str]):
        self.__plugin = plugin_name
        self.__item = item
        self.__path = path
        self.__subtree = subtree

    @property
    def message(self) -> str:
        available = '<none>' if len(self.__subtree) == 0 else ', '.join(sorted(f"'{it}'" for it in self.__subtree))
        return f"Plugin '{self.__plugin}' not found: Path item '{self.__item}' not present in " \
               f"plugin path '{self.__path}'.\n" \
               f"Available subtrees: {available}"


class WrongPluginType(PluginError):
    def __init__(self, plugin_name: str, expected_type: str):
        self.__name = plugin_name
        self.__expected = expected_type

    @property
    def message(self) -> str:
        return f"Error looking up {self.__expected}s: '{self.__name}' is no {self.__expected} plugin"


class AlgorithmNotFound(RetryableError):
    def __init__(self, plugin_name: str, lookup_name: str, alg_name: str, alg_type: str, available: Set[str]):
        self.__plugin = plugin_name
        self.__looked_up = lookup_name
        self.__alg = alg_name
        self.__type = alg_type
        self.__available = available

    @property
    def message(self) -> str:
        article = "an" if re.match(r"^[aeiou].*", self.__type) is not None else "a"
        available = '<none>' if len(self.__available) == 0 else ', '.join(sorted(f"'{it}'" for it in self.__available))
        return f"Plugin '{self.__plugin}' (looked up as '{self.__looked_up}') does not have {article} {self.__type} " \
               f"named '{self.__alg}'.\n" \
               f"Available {self.__type}s: {available}"


class NoAlgorithms(PluginError):
    def __init__(self, plugin_name: str, lookup_name: str, alg_type: str):
        self.__plugin = plugin_name
        self.__looked_up = lookup_name
        self.__type = alg_type

    @property
    def message(self) -> str:
        return f"'Plugin '{self.__plugin}' (looked up as '{self.__looked_up}') has no {self.__type}s"


class AmbiguousAlgorithms(PluginError):
    def __init__(self, plugin_name: str, lookup_name: str, alg_type: str, count: int, available: Set[str]):
        self.__plugin = plugin_name
        self.__looked_up = lookup_name
        self.__type = alg_type
        self.__count = count
        self.__available = available

    @property
    def message(self) -> str:
        available = '<none>' if len(self.__available) == 0 else ', '.join(sorted(f"'{it}'" for it in self.__available))
        return f"Ambiguous {self.__type} name '{self.__looked_up}': Plugin '{self.__plugin}' " \
               f"has {self.__count} attacks: {available}\n" \
               f"Did you mean to load all of them? Consider using '{self.__looked_up}/*' instead."


class ApplicationContext:
    def __init__(self):
        self.__click = None

        self.__installed_plugins: Set[str] = set()

        self.__datasets: Dict[str, Type[Dataset]] = {}
        self.__defenses: Dict[str, Type[Defense]] = {}
        self.__attacks: Dict[str, Type[AttackDefinition]] = {}

        self.__tree: Dict[str, Any] = {}

    def set_root(self, click_root: Group):
        self.__click = click_root

    def __contains__(self, plugin: Plugin):
        return plugin.plugin_path() in self.__installed_plugins

    def lookup_dataset(self, name: str) -> List[Type[Dataset]]:
        if name in self.__datasets:
            return self.__datasets[name]

        result = self.__do_lookup_dataset(name)

        if isinstance(result, PluginError):
            raise RuntimeError(result.message)
        else:
            self.__datasets[name] = result
            return result

    def lookup_defense(self, name: str) -> List[Type[Defense]]:
        if name in self.__defenses:
            return self.__defenses[name]

        result = self.__do_lookup_defense(name)

        if isinstance(result, PluginError):
            raise RuntimeError(result.message)
        else:
            self.__defenses[name] = result
            return result

    def lookup_attack(self, name: str) -> List[Type[AttackDefinition]]:
        if name in self.__attacks:
            return self.__attacks[name]

        result = self.__do_lookup_attack(name)

        if isinstance(result, PluginError):
            raise RuntimeError(result.message)
        else:
            self.__attacks[name] = result
            return result

    def __main_cli_group(self) -> Group:
        return self.__click

    def __dataset_cli_group(self) -> Group:
        # noinspection Mypy,PyTypeChecker
        return self.__click.commands['dataset']

    def _install_cli_module(self, plugin: CliPlugin) -> None:
        plugin.cli_module().install(self.__main_cli_group(), self.__dataset_cli_group())
        self.__installed_plugins.add(plugin.plugin_path())

    def _install_datasets(self, plugin: DatasetPlugin):
        self.__installed_plugins.add(plugin.plugin_path())
        datasets = self.__create_alg_subtree(plugin.plugin_path(), 'dataset')

        for dataset in plugin.datasets():
            datasets[dataset.name()] = dataset

    def _install_defenses(self, plugin: DefensePlugin):
        self.__installed_plugins.add(plugin.plugin_path())
        defenses = self.__create_alg_subtree(plugin.plugin_path(), 'defense')

        for defense in plugin.defenses():
            defenses[defense.name()] = defense

    def _install_attacks(self, plugin: AttackPlugin):
        self.__installed_plugins.add(plugin.plugin_path())
        attacks = self.__create_alg_subtree(plugin.plugin_path(), 'attack')

        for attack in plugin.attacks():
            attacks[attack.name()] = attack

    def __do_lookup_dataset(self, _name: str,
                            looked_up: Optional[str] = None) -> Union[PluginError, List[Type[Dataset]]]:
        if _name.startswith('.'):
            name = f'core.datasets{_name}'
            _logger.debug("'%s' is a relative name, looking up '%s'", _name, name)
        else:
            name = _name

        if '/' not in name:
            result = self.__lookup(looked_up or _name, name, 'dataset')
        elif name.count('/') == 1:
            [plugin, dataset] = name.split('/')
            result = self.__lookup(looked_up or _name, plugin, 'dataset', dataset)
        else:
            raise ValueError("Invalid dataset name '{0}': Only one '/' is allowed.".format(_name))

        if isinstance(result, RetryableError) and not name.startswith('core.datasets.'):
            # Try relative lookup in core modules
            _logger.debug("Loading dataset '%s' through direct lookup failed, trying relative lookup in "
                          "core modules", _name)

            core_result = self.__do_lookup_dataset(f'core.datasets.{name}', _name)

            if isinstance(core_result, PluginError):
                _logger.debug("Loading dataset '%s' through relative lookup in core modules failed", _name)

                if isinstance(core_result, RetryableError):
                    return result
                else:
                    return core_result
            else:
                return core_result
        else:
            return result

    def __do_lookup_defense(self, _name: str,
                            looked_up: Optional[str] = None) -> Union[PluginError, List[Type[Defense]]]:
        if _name.startswith('.'):
            name = f'core.defenses{_name}'
            _logger.debug("'%s' is a relative name, looking up '%s'", _name, name)
        else:
            name = _name

        if '/' not in name:
            result = self.__lookup(looked_up or _name, name, 'defense')
        elif name.count('/') == 1:
            [plugin, defense] = name.split('/')
            result = self.__lookup(looked_up or _name, plugin, 'defense', defense)
        else:
            raise ValueError("Invalid defense name '{0}': Only one '/' is allowed.".format(_name))

        if isinstance(result, RetryableError) and not name.startswith('core.defenses.'):
            # Try relative lookup in core modules
            _logger.debug("Loading defense '%s' through direct lookup failed, trying relative lookup in "
                          "core modules", _name)

            core_result = self.__do_lookup_defense(f'core.defenses.{name}', _name)

            if isinstance(core_result, PluginError):
                _logger.debug("Loading defense '%s' through relative lookup in core modules failed", _name)

                if isinstance(core_result, RetryableError):
                    return result
                else:
                    return core_result
            else:
                return core_result
        else:
            return result

    def __do_lookup_attack(self, _name: str,
                           looked_up: Optional[str] = None) -> Union[PluginError, List[Type[AttackDefinition]]]:
        if _name.startswith('.'):
            name = f'core.attacks{_name}'
            _logger.debug("'%s' is a relative name, looking up '%s'", _name, name)
        else:
            name = _name

        if '/' not in name:
            result = self.__lookup(looked_up or _name, name, 'attack')
        elif name.count('/') == 1:
            [plugin, attack] = name.split('/')
            result = self.__lookup(looked_up or _name, plugin, 'attack', attack)
        else:
            raise ValueError("Invalid attack name '{0}': Only one '/' is allowed.".format(_name))

        if isinstance(result, RetryableError) and not name.startswith('core.attacks.'):
            # Try relative lookup in core modules
            _logger.debug("Loading attack '%s' through direct lookup failed, trying relative lookup in "
                          "core modules", _name)

            core_result = self.__do_lookup_attack(f'core.attacks.{name}', _name)

            if isinstance(core_result, PluginError):
                _logger.debug("Loading attack '%s' through relative lookup in core modules failed", _name)

                if isinstance(core_result, RetryableError):
                    return result
                else:
                    return core_result
            else:
                return core_result
        else:
            return result

    def __lookup(self, lookup_path: str, real_path: str, alg_type: str,
                 name: Optional[str] = None) -> Union[PluginError, Any]:
        subtree = self.__alg_subtree(real_path, alg_type)
        if name is None:
            return subtree if isinstance(subtree, PluginError) else self.__extract_without_name(lookup_path, real_path,
                                                                                                alg_type)(subtree)
        else:
            return subtree if isinstance(subtree, PluginError) else self.__extract_with_name(lookup_path, real_path,
                                                                                             alg_type, name)(subtree)

    @staticmethod
    def __extract_without_name(lookup_path: str, real_path: str, alg_type: str):
        def extract(subtree) -> Union[PluginError, List[Any]]:
            if len(subtree) == 1:
                return list(subtree.values())
            elif len(subtree) == 0:
                return NoAlgorithms(real_path, lookup_path, alg_type)
            else:
                return AmbiguousAlgorithms(real_path, lookup_path, alg_type, len(subtree), set(subtree.keys()))

        return extract

    @staticmethod
    def __extract_with_name(lookup_path: str, real_path: str, alg_type: str, alg_name: str):
        def extract(subtree) -> Union[PluginError, List[Any]]:
            if alg_name == '*':
                if len(subtree) == 0:
                    return NoAlgorithms(real_path, lookup_path, alg_type)
                else:
                    return list(subtree.values())
            elif alg_name in subtree:
                return [subtree[alg_name]]
            else:
                return AlgorithmNotFound(real_path, lookup_path, alg_name, alg_type, set(subtree.keys()))

        return extract

    def __alg_subtree(self, path: str, alg_type: str):
        subtree = self.__plugin_subtree(path)

        def extract(d: Dict[str, Any]) -> Union[PluginError, Dict[str, Any]]:
            if f"__{alg_type}s__" not in d:
                return WrongPluginType(path, alg_type)

            return d[f"__{alg_type}s__"]

        return subtree if isinstance(subtree, PluginError) else extract(subtree)

    __not_allowed = {'__datasets__', '__attacks__', '__defenses__', ''}

    def __plugin_subtree(self, path: str) -> Union[PluginError, Dict[str, Any]]:
        if '/' in path:
            return InvalidPluginName(path, "'/' are not allowed")

        subtree = self.__tree
        seen = ""

        for item in path.split("."):
            if item in self.__not_allowed:
                return InvalidPluginName(path, f"Invalid path component '{item}'")

            if item not in subtree:
                return PluginNotFound(path, item, seen, set(subtree.keys()).difference(self.__not_allowed))

            seen += f'.{item}'
            subtree = subtree[item]

        return subtree

    def __create_alg_subtree(self, path: str, alg_type: str):
        subtree = self.__create_plugin_subtree(path)

        if f'__{alg_type}s__' not in subtree:
            subtree[f'__{alg_type}s__'] = {}

        return subtree[f'__{alg_type}s__']

    def __create_plugin_subtree(self, path: str) -> Dict[str, Any]:
        if '/' in path:
            raise ValueError("Invalid plugin name %s: '/' are not allowed", path)

        subtree = self.__tree

        for item in path.split("."):
            if item in {'__datasets__', '__attacks__', '__defenses__', ''}:
                raise KeyError("Invalid plugin name component '{0}' in plugin '{1}'".format(item, path))

            if item not in subtree:
                subtree[item] = {}

            subtree = subtree[item]

        return subtree


def load_core_modules(ctx: ApplicationContext) -> None:
    for mod in [
        ".datasets.dummy", ".datasets.liberatore",
        ".defenses.deterministic", ".defenses.randomized",
        ".attacks.dyer", ".attacks.herrmann", ".attacks.liberatore", ".attacks.panchenko", ".attacks.cumul",
        ".cli.evaluation", ".cli.error_bound", ".cli.outlier_detection"
    ]:
        install(mod, ctx)


# noinspection PyProtectedMember
def install(module_name: str, application_ctx: ApplicationContext):
    _logger.debug("Loading plugin module %s", module_name)

    try:
        if module_name.startswith("."):
            module = import_module(module_name, "fingerprinting")
        else:
            module = import_module(module_name)
    except ImportError:
        _logger.fatal("Plugin module %s not found", module_name)
        raise

    if not isinstance(module, Plugin):
        _logger.warning("Trying to import module %s, which is no plugin (has no plugin_path() method)", module_name)
    elif isinstance(module, Plugin) and module in application_ctx:
        _logger.error("Trying to install plugin %s (module %s), but a plugin with the same name was already installed",
                      module.plugin_path(), module_name)
    else:
        used = False

        if isinstance(module, CliPlugin):
            used = True
            _logger.debug("Installing plugin %s (module %s) as a CLI module", module.plugin_path(), module_name)
            application_ctx._install_cli_module(module)
        if isinstance(module, DatasetPlugin):
            used = True
            _logger.debug("Installing plugin %s (module %s) as a dataset plugin", module.plugin_path(), module_name)
            application_ctx._install_datasets(module)
        if isinstance(module, DefensePlugin):
            used = True
            _logger.debug("Installing plugin %s (module %s) as a defense plugin", module.plugin_path(), module_name)
            application_ctx._install_defenses(module)
        if isinstance(module, AttackPlugin):
            used = True
            _logger.debug("Installing plugin %s (module %s) as an attack plugin", module.plugin_path(), module_name)
            application_ctx._install_attacks(module)

        if not used:
            _logger.warning("Plugin module %s does not have the structure of a known plugin type", module_name)
