from importlib import import_module
from logging import getLogger
from typing import Iterable, Optional, Any, Dict

from click import Group
from typing_extensions import Protocol, runtime_checkable

from .pipeline import Dataset

_logger = getLogger(__name__)


class Plugin:
    def name(self) -> str:
        ...

    def __str__(self) -> str:
        return self.name()

    def __repr__(self) -> str:
        return self.name()


class CliPlugin(Plugin):
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
    def load(module_name: str) -> Optional['CliPlugin']:
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
class CliPluginProvider(Protocol):
    def cli_plugin(self) -> CliPlugin:
        ...


@runtime_checkable
class DatasetProvider(Protocol):
    def dataset(self) -> Dataset:
        ...


class ApplicationContext:
    def __init__(self, click_root: Group):
        self.__click = click_root

        self.__datasets: Dict[str, Dataset] = {}

    @staticmethod
    def of(*, click_root) -> 'ApplicationContext':
        return ApplicationContext(click_root)

    def main_cli_group(self) -> Group:
        return self.__click

    def dataset_cli_group(self) -> Group:
        # noinspection Mypy
        return self.__click.commands['dataset']

    def install_cli_plugin(self, plugin: CliPlugin) -> None:
        _logger.debug("Installing CLI plugin %s", plugin)
        plugin.install(self.main_cli_group(), self.dataset_cli_group())

    def install_dataset(self, dataset: Dataset):
        self.__datasets[dataset.name] = dataset


def install(module_name: str, application_ctx: ApplicationContext):
    _logger.debug("Loading plugin module %s", module_name)

    try:
        if module_name.startswith("."):
            module = import_module(module_name, "fingerprinting")
        else:
            module = import_module(module_name)
    except ImportError:
        _logger.fatal(f"Plugin module %s not found", module_name)
        raise

    if isinstance(module, CliPluginProvider):
        application_ctx.install_cli_plugin(module.cli_plugin())
    if isinstance(module, DatasetProvider):
        application_ctx.install_dataset(module.dataset())
