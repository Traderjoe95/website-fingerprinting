from datetime import datetime as py_datetime, timedelta

from pendulum import duration, Duration, datetime, DateTime


def to_duration(delta: timedelta) -> Duration:
    if isinstance(delta, Duration):
        return delta

    return duration(days=delta.days, seconds=delta.seconds, microseconds=delta.microseconds)


def to_datetime(dt: py_datetime) -> DateTime:
    if isinstance(dt, DateTime):
        return dt

    return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond, tz=dt.tzname())
