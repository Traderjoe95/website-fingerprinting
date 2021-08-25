from logging import getLogger

import numpy as np
import pandas as pd
from click import pass_context, command, argument, option
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from ..api.plugin import ApplicationContext, CliModule

_LOGGER = getLogger(__name__)


def plugin_path() -> str:
    return 'core.cli.outlier_detection'


def cli_module() -> CliModule:
    return CliModule(name='run_cli', dataset_commands=[__outliers])


@command("analyze-outliers", help="Run an outlier analysis on the specified dataset")
@argument("dataset", required=True, type=str)
@option('-s', '--out-suffix', type=str, help='Suffix for the output files', default='')
@option('-a', '--all-sites', help='Use all sites instead of the ones defined in config.toml', is_flag=True)
@option('-c', '--chunk-size', type=int, help='The chunk size to process sites with', default=100)
@pass_context
def __outliers(ctx, dataset: str, out_suffix: str, all_sites: bool, chunk_size: int):
    if chunk_size < 1:
        raise ValueError(f"The chunk size must be at least 1, but got {chunk_size}")

    app: ApplicationContext = ctx.obj['application_ctx']
    _ds = app.lookup_dataset(dataset)

    if len(_ds) != 1:
        raise RuntimeError(f"Expected 1 dataset, but got {len(_ds)}")

    _LOGGER.info("Starting outlier analysis for data set '%s'", dataset)
    if all_sites:
        _LOGGER.info("Using all sites from the data set")
    else:
        _LOGGER.info("Using sites defined in config.toml")

    ds = _ds[0]()
    __outlier_analysis(ds, out_suffix=out_suffix, sites=range(ds.site_count) if all_sites else ds.sites,
                       chunk_size=chunk_size)


def __outlier_analysis(ds, sites, chunk_size, out_suffix=''):
    site_count = len(sites)
    chunk_count = site_count // chunk_size if site_count % chunk_size == 0 else site_count // chunk_size + 1

    trace_outliers = []
    site_stats = []

    for i in range(chunk_count):
        chunk_start = i * chunk_size
        chunk_end = min((i + 1) * chunk_size, site_count)

        chunk = sites[chunk_start:chunk_end] if sites is not None else range(chunk_start, chunk_end)

        traces = pd.concat([t for t in ds.load(offset=0, examples_per_site=205, sites=chunk)])
        traces = traces[["site_id", "trace_id", "size"]].set_index(['site_id', 'trace_id'])

        traces_out = traces[traces['size'] <= 0].groupby(['site_id', 'trace_id']).agg([np.count_nonzero, 'sum'])
        traces_out.columns = traces_out.columns.levels[1]
        traces_out = traces_out.rename({'count_nonzero': 'count_out', 'sum': 'bw_out'}, axis=1)

        traces_in = traces[traces['size'] >= 0].groupby(['site_id', 'trace_id'], as_index=False).agg([np.count_nonzero,
                                                                                                      'sum'])
        traces_in.columns = traces_in.columns.levels[1]
        traces_in = traces_in.rename({'count_nonzero': 'count_in', 'sum': 'bw_in'}, axis=1)

        traces = traces_in.join(traces_out, how='outer').fillna(0).astype(np.float64)

        stats = traces.groupby('site_id').describe().drop('count', axis=1, level=1)
        stats.columns = pd.Index(['_'.join(col) for col in stats.columns.values])

        def outliers(df):
            lof = LocalOutlierFactor().fit_predict(df)
            ilf = IsolationForest().fit_predict(df)

            try:
                env = EllipticEnvelope(support_fraction=1.).fit_predict(df)

                return df.assign(outlier_lof=lof,
                                 outlier_ilf=ilf,
                                 outlier_env=env)
            except:
                return df.assign(outlier_lof=lof,
                                 outlier_ilf=ilf,
                                 outlier_env=np.ones_like(lof))

        traces = traces.groupby('site_id').apply(outliers)
        traces['outlier'] = traces['outlier_lof'] + traces['outlier_ilf'] + traces['outlier_env']

        outlier_counts = traces['outlier'].groupby('site_id').apply(lambda s: s.groupby(s).size())
        outlier_counts.index = outlier_counts.index.rename('score', level=1)
        outlier_counts = outlier_counts.reset_index()
        outlier_counts = outlier_counts.pivot(index='site_id', values='outlier', columns='score').rename(
            {-3: 'outlier', -1: 'prob_outlier', 1: 'prob_inlier', 3: 'inlier'}, axis=1
        ).fillna(0.)

        stats = pd.concat([stats, outlier_counts], axis=1)

        for col in ('count_out', 'bw_out', 'count_in', 'bw_in'):
            stats[f'{col}_var'] = stats[f'{col}_std'] ** 2
            stats[f'{col}_iqr'] = stats[f'{col}_75%'] - stats[f'{col}_25%']
            stats[f'{col}_range'] = stats[f'{col}_max'] - stats[f'{col}_min']

        trace_outliers.append(traces)
        site_stats.append(stats)

        _LOGGER.info(f"Processed chunk %d/%d (sites #%d - #%d)", i + 1, chunk_count, chunk_start, chunk_end - 1)

    site_stats = pd.concat(site_stats)
    site_stats = site_stats.reindex(site_stats.columns.sort_values(), axis=1)

    trace_outliers = pd.concat(trace_outliers)
    trace_outliers = trace_outliers.reindex(trace_outliers.columns.sort_values(), axis=1)

    trace_outliers.to_csv(f'trace_outliers{out_suffix}.csv')
    site_stats.to_csv(f'site_stats{out_suffix}.csv')
