from typing import Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.graph_objs.box as bx
import plotly.graph_objs.layout as gl
import plotly.graph_objs.scatter as sc
import plotly.graph_objs.violin as vl
import plotly.subplots as sp
import scipy.stats as st
from plotly.basedatatypes import BaseTraceType

COLORS = [
    'rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)',
    'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)'
]

FADED_COLORS = [
    'rgba(31, 119, 180, 0.2)', 'rgba(255, 127, 14, 0.2)', 'rgba(44, 160, 44, 0.2)', 'rgba(214, 39, 40, 0.2)',
    'rgba(148, 103, 189, 0.2)', 'rgba(140, 86, 75, 0.2)', 'rgba(227, 119, 194, 0.2)', 'rgba(127, 127, 127, 0.2)',
    'rgba(188, 189, 34, 0.2)', 'rgba(23, 190, 207, 0.2)'
]

MARKERS = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'pentagon', 'hexagon', 'star']

TRANSPARENT = 'rgba(255, 255, 255, 0)'

SUB_PLOTS = [(1, 1), (1, 2), (1, 3), (2, 2), (1, 5), (2, 3), (1, 7), (2, 4), (3, 3), (2, 5), (1, 11), (3, 4), (1, 13),
             (2, 7), (3, 5), (4, 4)]


def __preprocess_aggregated(results: pd.DataFrame,
                            x: str,
                            y: str = 'score',
                            series: Optional[str] = None,
                            agg: str = 'mean') -> pd.DataFrame:
    return pd.pivot_table(results, index=x, values=y, columns=series, aggfunc=agg)


def __preprocess_distribution(results: pd.DataFrame,
                              x: str,
                              y: str = 'score',
                              series: Optional[str] = None) -> pd.DataFrame:
    grouper = [x] if series is None else [x, series]

    tmp = results.groupby(grouper, as_index=False).apply(lambda df: df.reset_index(drop=True)).reset_index(level=0,
                                                                                                           drop=True)
    return pd.pivot_table(tmp.reset_index(), index=[x, "index"], values=y, columns=series).reset_index(level='index',
                                                                                                       drop=True)


def __single_layout(title: Optional[str], series: Optional[str], x_title: Optional[str], x: str,
                    x_range: Optional[Tuple[float, float]], y_title: Optional[str], y: str,
                    y_range: Optional[Tuple[float, float]], **layout_kwargs) -> go.Layout:
    return go.Layout(title=None if title is None else gl.Title(text=title),
                     xaxis=gl.XAxis(range=x_range, title=gl.xaxis.Title(text=x_title or x.capitalize())),
                     yaxis=gl.YAxis(range=y_range, title=gl.yaxis.Title(text=y_title or y.capitalize())),
                     legend=gl.Legend(title=None if series is None else gl.legend.Title(text=series.capitalize())),
                     **layout_kwargs)


def __create_figure(traces: Union[List[BaseTraceType], List[List[BaseTraceType]]],
                    title: Optional[str],
                    series: Optional[str],
                    x_title: Optional[str],
                    x: str,
                    x_range: Optional[Tuple[float, float]],
                    y_title: Optional[str],
                    y: str,
                    y_range: Optional[Tuple[float, float]],
                    subplots: Optional[List[str]] = None,
                    subplot_arrangement: Optional[Tuple[int, int]] = None,
                    **layout_kwargs) -> go.Figure:
    if subplots is None:
        layout = __single_layout(title, series, x_title, x, x_range, y_title, y, y_range, **layout_kwargs)
        _traces: List[BaseTraceType] = []
        for t in traces:
            if isinstance(t, list):
                _traces += t
            else:
                _traces.append(t)

        return go.Figure(data=_traces, layout=layout)
    else:
        arrangement = subplot_arrangement or SUB_PLOTS[len(subplots) - 1]
        fig = sp.make_subplots(rows=arrangement[0],
                               cols=arrangement[1],
                               subplot_titles=[s.capitalize() for s in subplots],
                               vertical_spacing=0.15,
                               horizontal_spacing=0.075)
        fig.update_layout(title=None if title is None else gl.Title(text=title.capitalize()),
                          legend=gl.Legend(title=None if series is None else gl.legend.Title(text=series.capitalize())),
                          **layout_kwargs)
        fig.update_xaxes(range=x_range, title=gl.xaxis.Title(text=x_title or x.capitalize()), matches='x')
        fig.update_yaxes(range=y_range,
                         title=gl.yaxis.Title(text=y_title or y.capitalize()),
                         rangemode='tozero',
                         matches='y')

        for i, trace in enumerate(traces):
            col = (i % arrangement[1]) + 1
            row = (i // arrangement[1]) + 1

            if isinstance(trace, list):
                for t in trace:
                    fig.add_trace(t, row=row, col=col)
            else:
                fig.add_trace(trace, row=row, col=col)

        return fig


def __line_traces(results: pd.DataFrame,
                  x: str,
                  y: str,
                  series: Optional[str],
                  agg: str,
                  confidence: Optional[float],
                  confidence_style: str,
                  show_legend: bool = True) -> List[go.Scatter]:
    data = __preprocess_aggregated(results, x, y, series, agg)

    traces: List[go.Scatter] = []
    confidence_style = confidence_style.lower()

    if confidence is not None:
        if confidence <= 0 or confidence > 1:
            raise ValueError('Must select a confidence interval between 0 and 1')

        if series is None:
            grouper = [x]
            selection = [x, y]
        else:
            grouper = [series, x]
            selection = [x, y, series]

        agg = results[selection].groupby(grouper).aggregate(['size', 'mean', st.sem]) + 1.0E-10
        confidence_int = np.array(
            st.t.interval(confidence, agg[(y, 'size')], loc=agg[(y, 'mean')], scale=agg[(y, 'sem')]))
        confidence_df = pd.DataFrame(confidence_int.T, columns=['conf_int_lower', 'conf_int_upper'],
                                     index=agg.index).reset_index()
        confidence_lower = pd.pivot_table(confidence_df, columns=series, values='conf_int_lower',
                                          index=x).rename({'conf_int_lower': 'score'}, axis=1)
        confidence_upper = pd.pivot_table(confidence_df, columns=series, values='conf_int_upper',
                                          index=x).rename({'conf_int_upper': 'score'}, axis=1)
    else:
        confidence_lower = pd.DataFrame()
        confidence_upper = pd.DataFrame()

    for i, c in enumerate(__sort_none_first(list(data.columns))):
        base_trace = go.Scatter(x=data.index,
                                y=data[c],
                                mode='lines',
                                line=sc.Line(shape='spline', smoothing=0.4, color=COLORS[i % len(COLORS)]),
                                marker=sc.Marker(symbol=MARKERS[i % len(MARKERS)]),
                                name=c.capitalize(),
                                legendgroup=c,
                                showlegend=show_legend)

        if confidence is None:
            traces.append(base_trace)
        elif confidence_style == 'error':
            base_trace.error_y = sc.ErrorY(array=confidence_upper[c] - data[c],
                                           arrayminus=data[c] - confidence_lower[c],
                                           type='data')
            traces.append(base_trace)
        elif confidence_style in {'line', 'area'}:
            traces.append(
                go.Scatter(
                    x=data.index,
                    y=confidence_upper[c],
                    mode='lines',
                    line=sc.Line(shape='spline',
                                 smoothing=0.3,
                                 dash='dash',
                                 width=1,
                                 color=COLORS[i % len(COLORS)] if confidence_style == 'line' else TRANSPARENT),
                    showlegend=False,
                    legendgroup=c,
                    name="95% Upper",
                ))
            traces.append(
                go.Scatter(
                    x=data.index,
                    y=confidence_lower[c],
                    mode='lines',
                    line=sc.Line(shape='spline',
                                 smoothing=0.3,
                                 dash='dash',
                                 width=1,
                                 color=COLORS[i % len(COLORS)] if confidence_style == 'line' else TRANSPARENT),
                    fill=None if confidence_style == 'line' else 'tonexty',
                    fillcolor=None if confidence_style == 'line' else FADED_COLORS[i % len(FADED_COLORS)],
                    showlegend=False,
                    legendgroup=c,
                    name="95% Lower",
                ))
            traces.append(base_trace)

    return traces


def __box_traces(results: pd.DataFrame,
                 x: str,
                 y: str,
                 series: Optional[str],
                 show_sd: bool,
                 show_legend: bool = True) -> List[go.Box]:
    data = __preprocess_distribution(results, x, y, series)

    return [
        go.Box(x=data.index,
               y=data[c],
               marker=bx.Marker(color=COLORS[i % len(COLORS)]),
               boxmean='sd' if show_sd else True,
               name=c.capitalize(),
               offsetgroup=c,
               legendgroup=c,
               showlegend=show_legend) for i, c in enumerate(__sort_none_first(list(data.columns)))
    ]


def __violin_traces(results: pd.DataFrame,
                    x: str,
                    y: str,
                    series: Optional[str],
                    bandwidth: float,
                    show_legend: bool = True) -> List[go.Violin]:
    data = __preprocess_distribution(results, x, y, series)

    return [
        go.Violin(x=data.index,
                  y=data[c],
                  box=vl.Box(visible=True, width=0.5),
                  marker=vl.Marker(color=COLORS[i % len(COLORS)]),
                  meanline=vl.Meanline(visible=True),
                  name=c.capitalize(),
                  bandwidth=bandwidth,
                  offsetgroup=c,
                  legendgroup=c,
                  showlegend=show_legend) for i, c in enumerate(__sort_none_first(list(data.columns)))
    ]


def line(results: pd.DataFrame,
         *,
         x: str,
         y: str = 'score',
         series: Optional[str] = None,
         agg='mean',
         confidence: Optional[float] = None,
         confidence_style: str = 'error',
         title: Optional[str] = None,
         x_title: Optional[str] = None,
         x_range: Optional[Tuple[float, float]] = None,
         y_title: Optional[str] = None,
         y_range: Optional[Tuple[float, float]] = None) -> go.Figure:
    traces: List[go.Scatter] = __line_traces(results, x, y, series, agg, confidence, confidence_style)

    return __create_figure(traces, title, series, x_title, x, x_range, y_title, y, y_range, hovermode='x unified')


def box(results: pd.DataFrame,
        *,
        x: str,
        y: str = 'score',
        series: Optional[str] = None,
        show_sd: bool = False,
        title: Optional[str] = None,
        x_title: Optional[str] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_title: Optional[str] = None,
        y_range: Optional[Tuple[float, float]] = None) -> go.Figure:
    traces = __box_traces(results, x, y, series, show_sd)
    return __create_figure(traces, title, series, x_title, x, x_range, y_title, y, y_range, boxmode='group')


def violin(results: pd.DataFrame,
           *,
           x: str,
           y: str = 'score',
           series: Optional[str] = None,
           bandwidth: float = 0.025,
           x_title: Optional[str] = None,
           y_title: Optional[str] = None,
           x_range: Optional[Tuple[float, float]] = None,
           y_range: Optional[Tuple[float, float]] = None,
           title: Optional[str] = None) -> go.Figure:
    traces = __violin_traces(results, x, y, series, bandwidth)
    return __create_figure(traces, title, series, x_title, x, x_range, y_title, y, y_range, violinmode='group')


def violin_compare(results: pd.DataFrame,
                   *,
                   x: str,
                   series: str,
                   y: str = 'score',
                   bandwidth: float = 0.025,
                   x_title: Optional[str] = None,
                   y_title: Optional[str] = None,
                   x_range: Optional[Tuple[float, float]] = None,
                   y_range: Optional[Tuple[float, float]] = None,
                   title: Optional[str] = None) -> go.Figure:
    data = __preprocess_distribution(results, x, y, series)

    if len(data.columns) != 2:
        raise ValueError(f"The input data is not suitable for violin_compare: The series dimension '{series}'"
                         f" should have 2 distinct values, but has {len(data.columns)}")

    c1 = data.columns[0]
    c2 = data.columns[1]

    traces = [
        go.Violin(x=data.index,
                  y=data[c1],
                  box=vl.Box(visible=False),
                  marker=vl.Marker(color=COLORS[0]),
                  meanline=vl.Meanline(visible=True),
                  bandwidth=bandwidth,
                  name=c1.capitalize(),
                  side='negative'),
        go.Violin(x=data.index,
                  y=data[c2],
                  box=vl.Box(visible=False),
                  marker=vl.Marker(color=COLORS[1]),
                  meanline=vl.Meanline(visible=True),
                  bandwidth=bandwidth,
                  name=c2.capitalize(),
                  side='positive')
    ]

    return __create_figure(traces,
                           title,
                           series,
                           x_title,
                           x,
                           x_range,
                           y_title,
                           y,
                           y_range,
                           violingap=0,
                           violinmode='overlay')


def __sort_none_first(values: List[str]) -> List[str]:
    if "none" in values:
        return ["none"] + [v for v in values if v != "none"]
    else:
        return values


def line_multi(results: pd.DataFrame,
               *,
               x: str,
               y: str = 'score',
               subplot: str,
               series: Optional[str] = None,
               agg='mean',
               confidence: Optional[float] = None,
               confidence_style: str = 'error',
               title: Optional[str] = None,
               x_title: Optional[str] = None,
               x_range: Optional[Tuple[float, float]] = None,
               y_title: Optional[str] = None,
               y_range: Optional[Tuple[float, float]] = None) -> go.Figure:
    subplots = __sort_none_first(list(np.unique(results[subplot])))
    traces: List[List[go.Scatter]] = []

    first = True

    for s in subplots:
        traces.append(
            __line_traces(results[results[subplot] == s],
                          x,
                          y,
                          series,
                          agg,
                          confidence,
                          confidence_style,
                          show_legend=first))
        first = False

    return __create_figure(traces,
                           title,
                           series,
                           x_title,
                           x,
                           x_range,
                           y_title,
                           y,
                           y_range,
                           list(subplots),
                           hovermode='x unified',
                           showlegend=series is not None)


def box_multi(results: pd.DataFrame,
              *,
              x: str,
              y: str = 'score',
              subplot: str,
              series: Optional[str] = None,
              show_sd: bool = False,
              title: Optional[str] = None,
              x_title: Optional[str] = None,
              x_range: Optional[Tuple[float, float]] = None,
              y_title: Optional[str] = None,
              y_range: Optional[Tuple[float, float]] = None) -> go.Figure:
    subplots = __sort_none_first(list(np.unique(results[subplot])))
    traces: List[List[go.Box]] = []

    first = True

    for s in subplots:
        traces.append(__box_traces(results[results[subplot] == s], x, y, series, show_sd, show_legend=first))
        first = False

    return __create_figure(traces,
                           title,
                           series,
                           x_title,
                           x,
                           x_range,
                           y_title,
                           y,
                           y_range,
                           list(subplots),
                           boxmode='group',
                           showlegend=series is not None)


def violin_multi(results: pd.DataFrame,
                 *,
                 x: str,
                 y: str = 'score',
                 subplot: str,
                 series: Optional[str] = None,
                 bandwidth: float = 0.025,
                 title: Optional[str] = None,
                 x_title: Optional[str] = None,
                 x_range: Optional[Tuple[float, float]] = None,
                 y_title: Optional[str] = None,
                 y_range: Optional[Tuple[float, float]] = None) -> go.Figure:
    subplots = __sort_none_first(list(np.unique(results[subplot])))
    traces: List[List[go.Violin]] = []

    first = True

    for s in subplots:
        traces.append(__violin_traces(results[results[subplot] == s], x, y, series, bandwidth, show_legend=first))
        first = False

    return __create_figure(traces,
                           title,
                           series,
                           x_title,
                           x,
                           x_range,
                           y_title,
                           y,
                           y_range,
                           list(subplots),
                           violinmode='group',
                           showlegend=series is not None)


def __grid_figure(traces: Union[List[BaseTraceType], List[List[BaseTraceType]]], title: Optional[str],
                  series: Optional[str], x_title: Optional[str], x: str, x_range: Optional[Tuple[float, float]],
                  y_title: Optional[str], y: str, y_range: Optional[Tuple[float, float]], cols: List[str],
                  rows: List[str], **layout_kwargs) -> go.Figure:
    subplots = [f'{c.capitalize()} / {r.capitalize()}' for r in rows for c in cols]

    return __create_figure(traces,
                           title,
                           series,
                           x_title,
                           x,
                           x_range,
                           y_title,
                           y,
                           y_range,
                           list(subplots),
                           showlegend=series is not None,
                           subplot_arrangement=(len(rows), len(cols)),
                           **layout_kwargs)


def line_grid(results: pd.DataFrame,
              *,
              x: str,
              y: str = 'score',
              row: str,
              col: str,
              series: Optional[str] = None,
              agg='mean',
              confidence: Optional[float] = None,
              confidence_style: str = 'error',
              title: Optional[str] = None,
              x_title: Optional[str] = None,
              x_range: Optional[Tuple[float, float]] = None,
              y_title: Optional[str] = None,
              y_range: Optional[Tuple[float, float]] = None) -> go.Figure:
    rows = __sort_none_first(list(np.unique(results[row])))
    cols = __sort_none_first(list(np.unique(results[col])))
    traces: List[List[go.Scatter]] = []

    first = True

    for r in rows:
        for c in cols:
            traces.append(
                __line_traces(results[(results[row] == r) & (results[col] == c)],
                              x,
                              y,
                              series,
                              agg,
                              confidence,
                              confidence_style,
                              show_legend=first))
            first = False

    return __grid_figure(traces,
                         title,
                         series,
                         x_title,
                         x,
                         x_range,
                         y_title,
                         y,
                         y_range,
                         list(cols),
                         list(rows),
                         hovermode='x unified')


def box_grid(results: pd.DataFrame,
             *,
             x: str,
             y: str = 'score',
             row: str,
             col: str,
             series: Optional[str] = None,
             show_sd: bool = False,
             title: Optional[str] = None,
             x_title: Optional[str] = None,
             x_range: Optional[Tuple[float, float]] = None,
             y_title: Optional[str] = None,
             y_range: Optional[Tuple[float, float]] = None) -> go.Figure:
    rows = __sort_none_first(list(np.unique(results[row])))
    cols = __sort_none_first(list(np.unique(results[col])))
    traces: List[List[go.Box]] = []

    first = True

    for r in rows:
        for c in cols:
            traces.append(
                __box_traces(results[(results[row] == r) & (results[col] == c)],
                             x,
                             y,
                             series,
                             show_sd,
                             show_legend=first))
            first = False

    return __grid_figure(traces,
                         title,
                         series,
                         x_title,
                         x,
                         x_range,
                         y_title,
                         y,
                         y_range,
                         list(cols),
                         list(rows),
                         boxmode='group')


def violin_grid(results: pd.DataFrame,
                *,
                x: str,
                y: str = 'score',
                row: str,
                col: str,
                series: Optional[str] = None,
                bandwidth: float = 0.025,
                title: Optional[str] = None,
                x_title: Optional[str] = None,
                x_range: Optional[Tuple[float, float]] = None,
                y_title: Optional[str] = None,
                y_range: Optional[Tuple[float, float]] = None) -> go.Figure:
    rows = __sort_none_first(list(np.unique(results[row])))
    cols = __sort_none_first(list(np.unique(results[col])))
    traces: List[List[go.Violin]] = []

    first = True

    for r in rows:
        for c in cols:
            traces.append(
                __violin_traces(results[(results[row] == r) & (results[col] == c)],
                                x,
                                y,
                                series,
                                bandwidth,
                                show_legend=first))
            first = False

    return __grid_figure(traces,
                         title,
                         series,
                         x_title,
                         x,
                         x_range,
                         y_title,
                         y,
                         y_range,
                         cols,
                         rows,
                         violinmode='group')
