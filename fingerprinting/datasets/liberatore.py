import os
import warnings
from datetime import datetime, timedelta
from functools import partial
from glob import glob
from logging import getLogger
from multiprocessing import Queue
from os import remove, getcwd, chdir, listdir
from os.path import join, isfile, isdir, sep, split
from time import perf_counter
from typing import Iterable, Optional, List, Type, Dict

import curio
import numpy as np
import pandas as pd
from click import option, group
from curio import subprocess, run_in_process, TaskGroup
from dpkt import pcap, ethernet
from pendulum import duration

from ..api import Dataset
from ..api.dataset import DatasetConfig
from ..api.plugin import CliModule
from ..api.typing import OffsetOrDelta, SiteSelection, TracesStream
from ..util.config import CONFIG, ConfigObject, ConfigValue
from ..util.dt import to_duration
from ..util.fs import extract_tarball, mkdirs, remove_recursive
from ..util.http import download_to
from ..util.logging import configure_worker, get_queue

_logger = getLogger(__name__)


class ExtractionConfig:
    def __init__(self, max_cpu_cores: int = 4, max_sites_per_store: int = 500):
        if max_cpu_cores < 1:
            raise ValueError(f"datasets.liberatore.extract.max_cpu_cores must be positive, but was {max_cpu_cores}")

        if max_sites_per_store < 1:
            raise ValueError("datasets.liberatore.extract.max_sites_per_store must be positive, "
                             f"but was {max_sites_per_store}")

        self.__max_cpu_cores = max_cpu_cores
        self.__max_sites_per_store = max_sites_per_store

    @property
    def max_cpu_cores(self) -> int:
        return self.__max_cpu_cores

    @property
    def max_sites_per_store(self) -> int:
        return self.__max_sites_per_store

    @property
    def chunks(self) -> int:
        return 2000 // self.__max_sites_per_store + 0 if 2000 % self.__max_sites_per_store == 0 else 1


class LiberatoreDatasetConfig(DatasetConfig):
    def __init__(self,
                 url: str = "http://skulddata.cs.umass.edu/traces/network",
                 files: Optional[List[str]] = None,
                 extraction: Optional[Dict[str, ConfigValue]] = None,
                 **kwargs):
        super(LiberatoreDatasetConfig, self).__init__(default_name='liberatore', **kwargs)

        self.__url = url
        _files = files or [
            "pcap-logs-0.tar.bz2", "pcap-logs-1.tar.bz2", "pcap-logs-2.tar.bz2", "pcap-logs-3.tar.bz2"
        ]

        if any(map(lambda f: not isinstance(f, str), _files)):
            raise ValueError(f"datasets.liberatore.files must be a list of strings, but was {_files}")

        self.__files: Iterable[str] = list(map(str, _files))

        self.__extract = ConfigObject(extraction or {}).as_obj(ExtractionConfig)

    @property
    def url(self) -> str:
        return self.__url

    @property
    def files(self) -> Iterable[str]:
        return self.__files

    @property
    def extract(self) -> ExtractionConfig:
        return self.__extract


_CONFIG = CONFIG.get_obj("datasets.liberatore").as_obj(LiberatoreDatasetConfig)


def cli_module() -> CliModule:
    return CliModule(name="liberatore_dataset", dataset_commands=[__liberatore])


def plugin_path() -> str:
    return 'core.datasets.liberatore'


def datasets() -> List[Type[Dataset]]:
    return [LiberatoreDataset]


class LiberatoreDataset(Dataset):
    def __init__(self, path: str = _CONFIG.path):
        super(LiberatoreDataset, self).__init__(sites=2000, traces_per_site=205, config=_CONFIG)
        self.__path = path

    @staticmethod
    def name() -> str:
        return "liberatore"

    def _do_load(self,
                 offset: OffsetOrDelta = 0,
                 examples_per_site: Optional[int] = None,
                 sites: SiteSelection = None) -> TracesStream:
        _offset, _count = self._check_offset_and_count(offset, examples_per_site)
        _sites = self._check_sites(sites)

        for store_id in range(_CONFIG.extract.chunks):
            store_range = range(_CONFIG.extract.max_sites_per_store * store_id,
                                _CONFIG.extract.max_sites_per_store * (store_id + 1))
            intersection = Dataset._intersect(_sites, store_range)

            if intersection:
                with pd.HDFStore(join(self.__path, "traces", f"traces-{store_id}.h5"), mode='r') as store:
                    cond = [f"(trace_id >= {_offset}) & (trace_id < {_offset + _count})"]

                    if isinstance(intersection, bool) or len(intersection) == _CONFIG.extract.max_sites_per_store:
                        _logger.debug("Loading all sites from store %d", store_id)

                        if _offset == 0 and _count == self.traces_per_site:
                            _logger.debug("Loading all traces from store %d", store_id)
                            yield store["traces"]
                            continue

                    elif isinstance(intersection, range):
                        _logger.debug("Using range optimization for loading %s from store %d", intersection, store_id)
                        if intersection.stop < store_range.stop:
                            cond.append(f"site_id < {intersection.stop}")
                        if intersection.start > store_range.start:
                            cond.append(f"site_id >= {intersection.start}")
                        if intersection.step != 1:
                            # TODO: Change this when pandas issue with arithmetic operators is fixed
                            # TODO: https://github.com/pandas-dev/pandas/issues/41100
                            step = intersection.step
                            mod = intersection.start % step
                            # cond.append(f"(site_id % {intersection.step}) == {mod}")

                            # noinspection PyTypeChecker
                            df: pd.DataFrame = store.select("traces", where=cond)
                            yield df[(df["site_id"] % step) == mod].copy()
                            continue

                    elif len(intersection) == 1:
                        _logger.debug("Loading site using site id from store %d", store_id)
                        cond.append(f"site_id == {intersection.pop()}")
                    else:
                        _logger.debug("Loading sites using set member check from store %d", store_id)
                        ids = sorted(intersection)

                        if len(ids) in {31, 32}:
                            # Workaround for pandas issue with selecting the appropriate input size
                            ids += ids[-3:]

                        values = "[" + ", ".join(str(s) for s in ids) + "]"
                        cond.append(f"site_id == {values}")

                    _logger.debug("Selecting sites from store %d where %s", store_id, cond)

                    yield store.select("traces", where=cond)

    def _delta_to_offset(self, delta: timedelta, base: int = 0) -> int:
        if delta < duration():
            raise ValueError("Negative time deltas are not allowed")

        _d = to_duration(delta)

        with pd.HDFStore(join(self.__path, 'traces', "traces-0.h5"), mode='r') as store:
            # noinspection PyTypeChecker
            df: pd.DataFrame = store.select("traces", where=['site_id == 0'])

            times = np.unique(df["collection_time"].dt.to_pydatetime())[base:]

        _deltas = [to_duration(t - times[0]) for t in times]

        if _d > _deltas[-1]:
            raise IndexError(f"The dataset only contains data for {_deltas[-1].in_words()}, "
                             f"but the requested offset was {_d.in_words()}")

        for i in range(base, self.traces_per_site - 2):
            if delta <= _deltas[i]:
                return i

        return self.traces_per_site - 1

    @staticmethod
    async def download(*,
                       dataset_base: str = _CONFIG.path,
                       base_url: str = _CONFIG.url,
                       files: Iterable[str] = _CONFIG.files,
                       force: bool = False):
        _logger.info("Downloading Liberatore Dataset...")
        downloads = ((f"{base_url}/{file}", file) for file in files)
        tasks = (download_to(url, join(dataset_base, file)) for url, file in downloads
                 if not isfile(join(dataset_base, file)) or force)

        if not LiberatoreDataset.__check_store_presence(dataset_base) or force:
            for task in tasks:
                await task
            _logger.info("Download complete.")
        else:
            _logger.info("Skipping download as there are already extracted traces.")

    @staticmethod
    async def extract(*, dataset_base: str = _CONFIG.path, files: Iterable[str] = _CONFIG.files, force: bool = False):
        _logger.info("Extracting Liberatore Dataset...")

        if not LiberatoreDataset.__check_store_presence(dataset_base) or force:
            pcap_dir = join(dataset_base, "pcap-logs")

            if not isdir(pcap_dir) or len(listdir(pcap_dir)) != 205 or force:
                for file in files:
                    path = join(dataset_base, file)

                    if not isfile(path):
                        _logger.warning("Archive file not found: %s", path)
                        continue

                    await extract_tarball(path, dataset_base)
                    remove(path)

                _logger.info("Extraction complete.")
        else:
            _logger.info("Skipping extraction as there are already extracted traces.")

    @staticmethod
    async def process_traces(*, dataset_base: str = _CONFIG.path, force: bool = False):
        mkdirs(join(dataset_base, 'traces'))

        _logger.info("Reading packet traces...")

        chunks = _CONFIG.extract.chunks
        sites_per_chunk = _CONFIG.extract.max_sites_per_store

        if not LiberatoreDataset.__check_store_presence(dataset_base) or force:
            max_worker_proc_old = curio.workers.MAX_WORKER_PROCESSES
            curio.workers.MAX_WORKER_PROCESSES = _CONFIG.extract.max_cpu_cores

            before = perf_counter()

            if _CONFIG.extract.max_cpu_cores > 1 and chunks > 1:
                async with TaskGroup(wait=all) as g:
                    for chunk_id in range(chunks):
                        await g.spawn(
                            run_in_process,
                            partial(LiberatoreDataset._translate_site_trace_chunk,
                                    worker=True,
                                    queue=get_queue(),
                                    log_level=getLogger().level), chunk_id * sites_per_chunk,
                                                                  (chunk_id + 1) * sites_per_chunk, chunk_id,
                            dataset_base)
            elif _CONFIG.extract.max_cpu_cores == 1:
                for chunk_id in range(chunks):
                    LiberatoreDataset._translate_site_trace_chunk(chunk_id * sites_per_chunk,
                                                                  (chunk_id + 1) * sites_per_chunk, chunk_id,
                                                                  dataset_base)
            else:
                LiberatoreDataset._translate_site_trace_chunk(0, 2000, 0, dataset_base)

            curio.workers.MAX_WORKER_PROCESSES = max_worker_proc_old

            if LiberatoreDataset.__check_store_presence(dataset_base):
                _logger.info(f"Trace processing completed in %s. Deleting PCAP logs...",
                             duration(seconds=perf_counter() - before).in_words())
                remove_recursive(join(dataset_base, 'pcap-logs'))

                _logger.info("Packet traces processing complete.")
            else:
                _logger.error(f"Trace processing failed after %s", duration(seconds=perf_counter() - before))
        else:
            _logger.info("Skipping trace processing as there are already processed traces.")

    @staticmethod
    def __check_store_presence(dataset_base: str) -> bool:
        for chunk_id in range(_CONFIG.extract.chunks):
            store_path = join(dataset_base, LiberatoreDataset.__compressed_store(chunk_id))

            if not isfile(store_path):
                return False

        return True

    @staticmethod
    def _translate_site_trace_chunk(start_site_id: int,
                                    end_site_id: int,
                                    chunk_id: int,
                                    dataset_base: str,
                                    worker: bool = False,
                                    queue: Optional[Queue] = None,
                                    log_level: Optional[int] = None) -> int:
        global _logger

        if worker and queue is not None:
            configure_worker(queue, log_level)
            _logger = getLogger("fingerprinting.datasets.liberatore")
        elif worker:
            warnings.warn("No queue passed to trace translation job, can't use logging", RuntimeWarning)

        counter = 0
        cwd = getcwd()

        try:
            chdir(dataset_base)

            store_file = LiberatoreDataset.__uncompressed_store(chunk_id)
            _logger.info("Opening HDF5 store %s...", store_file)

            with pd.HDFStore(store_file, mode="w") as store:
                for site_id in range(start_site_id, end_site_id):
                    counter += LiberatoreDataset.__translate_site_traces(site_id, store)

            _logger.info("Store %s is complete. Repacking and compressing...", store_file)
            compressed_store_file = LiberatoreDataset.__compressed_store(chunk_id)
            curio.run(LiberatoreDataset.__ptrepack, store_file, compressed_store_file)
            _logger.info("Store %s was compressed successfully.", compressed_store_file)

            remove(store_file)

            return counter
        finally:
            chdir(cwd)

    @staticmethod
    def __translate_site_traces(site_id: int, store: pd.HDFStore) -> int:
        expression = f'pcap-logs{sep}*{sep}*-site-{site_id}'

        _logger.debug("Collecting traces for site %d using '%s'" % (site_id, expression))

        traces = glob(expression)

        if len(traces) == 0:
            _logger.warning("Found no pcap logs for '%s'", expression)
            return 0

        trace_id_offset = 0

        base = perf_counter()
        previous = perf_counter()

        avg_duration = 0.

        dfs = []
        for pcap_file in traces:
            collection_path, _ = split(pcap_file)
            _, collection_id = split(collection_path)

            df = LiberatoreDataset.__pcap_to_df(pcap_file)

            df['site_id'] = site_id
            df['trace_id'] = trace_id_offset
            df['collection_time'] = pd.to_datetime(datetime.strptime(collection_id, '%Y-%m-%dT%H_%M_%S.%f'))

            trace_id_offset += 1

            dfs.append(df)

            now = perf_counter()
            time = now - previous
            previous = now
            avg_duration += time / 205

        df = pd.concat(dfs, ignore_index=True)
        df["site_id"] = df["site_id"].astype(np.uint16)
        df["trace_id"] = df["trace_id"].astype(np.uint8)
        df["size"] = df["size"].astype(np.int16)
        df["time"] = df["time"].astype(np.float32)
        df["collection_time"] = pd.to_datetime(df["collection_time"])
        df = df[['site_id', 'trace_id', 'collection_time', 'time', 'size']]

        now = perf_counter()
        concat_time = now - previous
        previous = now

        store.put(f'traces', df, append=True, format='t', data_columns=["site_id", "trace_id"])
        now = perf_counter()
        write_time = now - previous

        _logger.debug("Site %d processed in %f seconds, avg %f seconds/trace, concat %f seconds, write %f seconds" %
                      (site_id, now - base, avg_duration, concat_time, write_time))

        return trace_id_offset

    @staticmethod
    def __pcap_to_df(pcap_file: str) -> pd.DataFrame:
        with open(pcap_file, "rb") as f:
            _pcap = pcap.Reader(f)

            ref_time = None

            data = []

            for ts, buf in _pcap:
                eth = ethernet.Ethernet(buf)
                ip = eth.data
                tcp = ip.data

                if ref_time is None:
                    ref_time = ts

                direction = 1 if tcp.sport == 22 else -1

                data.append({"time": (ts - ref_time) * 1000, "size": direction * ip.len})

            if len(data) == 0:
                data.append({"time": 0, "size": 0})

            return pd.DataFrame(data, columns=["time", "size"])

    @staticmethod
    def __uncompressed_store(chunk_id: int) -> str:
        return join('traces', f'traces-{chunk_id}.uncompressed.h5')

    @staticmethod
    def __compressed_store(chunk_id: int) -> str:
        return join('traces', f'traces-{chunk_id}.h5')

    @staticmethod
    async def __ptrepack(in_store: str, out_store: str):
        try:
            os.remove(out_store)
        except FileNotFoundError:
            pass

        before = perf_counter()

        proc = subprocess.Popen(
            ['ptrepack', '--chunkshape=auto', '--propindexes', '--complib=blosc:zstd', '--complevel=9',
             in_store, out_store],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)

        _, stderr = await proc.communicate()
        return_code = await proc.wait()

        _logger.debug("Repacking of %s took %f seconds", in_store, perf_counter() - before)

        if stderr:
            raise RuntimeError(stderr.decode())
        elif return_code != 0:
            raise RuntimeError(f"ptrepack exited with code {return_code}")


@group(name="liberatore",
       chain=True,
       short_help="Manage the liberatore dataset",
       help="""Manage the liberatore dataset. 

When using this command, multiple subcommands may be chained, e.g.

python main.py dataset liberatore download extract process""")
def __liberatore():
    pass


@__liberatore.command(name="download", help="Download the liberatore dataset")
@option("-f", "--force", help="Force the download even if the data is already there", is_flag=True)
def __download_liberatore(force: bool):
    curio.run(partial(LiberatoreDataset.download, force=force))


@__liberatore.command(name="extract", help="Extract the liberatore dataset")
@option("-f", "--force", help="Force the extraction even if the data is already there", is_flag=True)
def __extract_liberatore(force: bool):
    curio.run(partial(LiberatoreDataset.extract, force=force))


@__liberatore.command(name="process", help="Read and process the PCAP traces")
@option("-f", "--force", help="Force the process even if the data is already there", is_flag=True)
def __process_liberatore(force: bool):
    curio.run(partial(LiberatoreDataset.process_traces, force=force))
