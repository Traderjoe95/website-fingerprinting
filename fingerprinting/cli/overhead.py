from typing import List
from random import randint, sample
from itertools import tee

import numpy as np
import pandas as pd
from click import command, pass_context, option, IntRange

from ..api.plugin import CliModule, ApplicationContext
from logging import getLogger
from ..datasets import Liberatore
from ..defenses import SessionRandomGaussian

_logger = getLogger(__name__)


def plugin_path() -> str:
    return 'core.cli.overhead'


def cli_module() -> CliModule:
    return CliModule(name='overhead_cli', top_level_commands=[__overhead])


@command("overhead")
@option("-ds", "--dataset", "datasets", metavar="DATASET", type=str, multiple=True, required=True,
        help="The names of the data sets to use in the overhead evaluation. May be used multiple times.")
@option("-def", "--defense", "defenses", metavar="DATASET", type=str, multiple=True, required=True,
        help="The names of the defenses to evaluate overheads for")
@option("-o", "--out-prefix", type=str, default="overhead", metavar="FILE", show_default=True)
@option("--size/--no-size", default=True, help="Whether to measure the bandwidth overhead")
@option("--time/--no-time", default=True, help="Whether to measure the transmission time overhead")
@option("-r", "--runs", type=IntRange(min=1), default=10, show_default=True,
        help="Number of random runs to do per dataset and defense")
@option("-s", "--sites", type=IntRange(min=1), default=100, show_default=True,
        help="Number of sites to use for overhead measurement")
@option("-t", "--traces", type=IntRange(min=1), default=25, show_default=True,
        help="Number of traces to use per site for overhead measurement")
@pass_context
def __overhead(ctx, datasets: List[str], defenses: List[str], out_prefix: str, size: bool, time: bool,
               runs: int, sites: int, traces: int):
    if not size and not time:
        _logger.info("Both, --no-size and --no-time are set. Exiting.")
        return

    app: ApplicationContext = ctx.obj['application_ctx']

    _datasets = [_t_ds() for name in datasets for _t_ds in app.lookup_dataset(name)]
    _defenses = [_t_def for name in defenses for _t_def in app.lookup_defense(name)]

    first = True

    for ds in _datasets:
        _logger.info("Running overhead estimation on dataset %s", ds.name())

        for r in range(runs):
            max_offset = ds.traces_per_site - traces
            offset = randint(0, max_offset)

            selected_sites = sample(ds.sites, k=sites)

            data = tee(ds.load(offset=offset, examples_per_site=traces, sites=selected_sites), len(_defenses) + 1)

            undefended = pd.concat(d.copy(deep=True) for d in data[-1])
            del undefended['collection_time']
            undefended = np.abs(undefended).groupby(['site_id', 'trace_id']).agg({'time': 'max', 'size': 'sum'})

            overhead_raw = []
            overhead_agg = []

            for i, _d in enumerate(_defenses):
                _logger.info("Measuring overhead for defense %s", _d.name())
                d = _d()

                defended = pd.concat(d.fit_defend(d.copy(deep=True) for d in data[i]))
                del defended['collection_time']

                overhead = __calculate_overhead(undefended, defended, size, time)
                overhead_mean = overhead.mean()
                overhead_mean['dataset'] = ds.name()
                overhead_mean['defense'] = _d.name()

                overhead_raw.append(overhead.assign(defense=_d.name(), dataset=ds.name()))
                overhead_agg.append(overhead_mean)

            sort_order = ['dataset', 'defense']

            if size:
                sort_order += ['size_diff', 'size_ratio', 'size_overhead']
            if time:
                sort_order += ['time_diff', 'time_ratio', 'time_overhead']

            oh = pd.concat(overhead_raw).reset_index(drop=True).reindex(sort_order, axis=1)
            oh_mean = pd.DataFrame(overhead_agg).reindex(sort_order, axis=1)

            if first:
                oh.to_csv(f'{out_prefix}.raw.csv', mode='w', index=False)
                oh_mean.to_csv(f'{out_prefix}.mean.csv', mode='w', index=False)
                first = False
            else:
                oh.to_csv(f'{out_prefix}.raw.csv', mode='a', index=False, header=False)
                oh_mean.to_csv(f'{out_prefix}.mean.csv', mode='a', index=False, header=False)

            _logger.info("Run %d/%d finished", r + 1, runs)


def __calculate_overhead(undefended: pd.DataFrame, defended: pd.DataFrame,
                         size: bool, time: bool):
    agg = {}
    if not size:
        del defended['size']
        undefended = undefended[['time']]
    else:
        agg['size'] = 'sum'
    if not time:
        del defended['time']
        undefended = undefended[['size']]
    else:
        agg['time'] = 'max'

    defended = np.abs(defended).groupby(['site_id', 'trace_id']).agg(agg)
    joined = undefended.join(defended, how="inner", lsuffix='', rsuffix="_def")

    result = {}
    if size:
        result['size_diff'] = joined['size_def'] - joined['size']
        result['size_ratio'] = joined['size_def'] / joined['size']
        result['size_overhead'] = result['size_diff'] / joined['size']
    if time:
        result['time_diff'] = joined['time_def'] - joined['time']
        result['time_ratio'] = joined['time_def'] / joined['time']
        result['time_overhead'] = result['time_diff'] / joined['time']

    return pd.DataFrame(result)
