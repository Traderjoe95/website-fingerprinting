from datetime import timedelta
from logging import getLogger
from os.path import join, isdir, abspath, isfile
from typing import Optional, List, Type

import click
import pandas as pd
import pendulum
from pendulum import duration, Duration

from ..api import Dataset
from ..api.dataset import DatasetConfig
from ..api.plugin import CliModule
from ..api.typing import OffsetOrDelta, SiteSelection, TracesStream
from ..util import mkdirs, TraceGenerator
from ..util.config import CONFIG
from ..util.dt import to_duration

_PAGES = [{
    "trace_length_mean": 50,
    "trace_length_variance": 12.5,
    "download_prob": 0.7,
    "inter_packet_time_mean": 100,
    "inter_packet_time_variance": 50.
}, {
    "trace_length_mean": 20,
    "trace_length_variance": 3.,
    "download_prob": 0.9,
    "inter_packet_time_mean": 200,
    "inter_packet_time_variance": 20.
}, {
    "trace_length_mean": 70,
    "trace_length_variance": 20.,
    "download_prob": 0.5,
    "inter_packet_time_mean": 50,
    "inter_packet_time_variance": 10.
}, {
    "trace_length_mean": 40,
    "trace_length_variance": 7.5,
    "download_prob": 0.3,
    "inter_packet_time_mean": 400,
    "inter_packet_time_variance": 50.
}, {
    "trace_length_mean": 35,
    "trace_length_variance": 5,
    "download_prob": 0.75,
    "inter_packet_time_mean": 150,
    "inter_packet_time_variance": 45.
}]

_logger = getLogger(__name__)


class DummyDatasetConfig(DatasetConfig):
    def __init__(self, trace_delta: Optional[str] = None, **kwargs):
        super(DummyDatasetConfig, self).__init__(default_name='dummy', **kwargs)

        self.trace_delta: Duration = pendulum.parse(trace_delta or "P1D")


_CONFIG = CONFIG.get_obj("datasets.dummy").as_obj(DummyDatasetConfig)


def cli_module() -> CliModule:
    return CliModule(name="dummy_dataset", dataset_commands=[__dummy])


def plugin_path() -> str:
    return 'core.datasets.dummy'


def datasets() -> List[Type[Dataset]]:
    return [DummyDataset]


class DummyDataset(Dataset):
    def __init__(self, path: str = _CONFIG.path):
        super(DummyDataset, self).__init__(sites=5, traces_per_site=50, config=_CONFIG)

        if not isdir(path) and not isfile(path):
            mkdirs(path)
        elif isfile(path):
            raise ValueError(f"{path} exists and is not a directory")

        self.__path = abspath(path)

    @staticmethod
    def name() -> str:
        return "dummy"

    def _do_load(self,
                 offset: OffsetOrDelta = 0,
                 examples_per_site: Optional[int] = None,
                 sites: SiteSelection = None) -> TracesStream:

        _offset, _count = self._check_offset_and_count(offset, examples_per_site)
        _sites = self._check_sites(sites)

        _requested_sites = self._intersect(_sites, range(5))

        if _requested_sites:
            if isinstance(_requested_sites, bool):
                _requested_sites = range(5)

            for s in _requested_sites:
                yield DummyDataset.__load_file(join(self.__path, f"site_{s}.traces.csv"), _offset, _count)

    def traces_per_site(self) -> int:
        return 50

    @staticmethod
    def __load_file(name: str, offset: int, count: int) -> pd.DataFrame:
        df = pd.read_csv(name)

        return df[(df["trace_id"] >= offset) & (df["trace_id"] < offset + count)]

    @staticmethod
    def generate(*,
                 dataset_base: str = _CONFIG.path,
                 trace_delta: timedelta = _CONFIG.trace_delta,
                 force: bool = False):
        _logger.info("Generating dummy dataset in %s", dataset_base)

        if trace_delta < duration():
            raise ValueError("The trace time delta must not be negative")

        mkdirs(dataset_base)
        DummyDataset.__generate_traces(dataset_base, trace_delta, force)

        _logger.info("Dummy dataset generated")

    @staticmethod
    def __generate_traces(dataset_base, trace_delta: timedelta, force: bool):
        for site_id in range(5):
            _logger.info("Generating traces for site %d", site_id)

            file = join(dataset_base, f"site_{site_id}.traces.csv")

            if not isfile(file) or force:
                gen = TraceGenerator(**_PAGES[site_id])

                df = pd.DataFrame(columns=["site_id", "trace_id", "collection_time", "time", "size"])

                for instance in gen.generate(webpage_id=site_id, count=50, trace_delta=trace_delta):
                    df = df.append(instance)

                df.to_csv(file, index=False)
            else:
                _logger.info("Traces already found and force=False, skipping")

    def _delta_to_offset(self, delta: timedelta, base: int = 0) -> int:
        if delta < duration():
            raise ValueError("Negative time deltas are not allowed")

        _d = to_duration(delta)
        trace_delta = _CONFIG.trace_delta
        last_trace_offset = 50 * trace_delta

        if _d > last_trace_offset:
            raise IndexError(f"The dataset only contains data for {last_trace_offset.in_words()}, "
                             f"but the requested offset was {_d.in_words()}")

        if delta % trace_delta == 0:
            return delta // trace_delta
        else:
            return delta // trace_delta + 1


@click.group("dummy", help="Manage the dummy dataset")
def __dummy():
    pass


@__dummy.command("generate", help="Generate new traces for the dummy dataset")
@click.option("-b", "--base", help="The dataset's base directory", type=str, default=_CONFIG.path, show_default=True)
@click.option("-f", "--force", help="Ignore already existing files and recreate them", is_flag=True)
def __generate_dummy(base: str, force: bool):
    DummyDataset.generate(dataset_base=base, force=force)
