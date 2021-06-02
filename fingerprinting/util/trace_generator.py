from datetime import timedelta
from random import gauss, random, randint
from typing import Iterator

import pandas as pd
import pendulum
from pendulum import datetime, duration


class TraceGenerator:
    __trace_length_mean: int
    __trace_length_variance: float

    __inter_packet_time_mean: int
    __inter_packet_time_variance: float

    __download_prob: float

    def __init__(self,
                 trace_length_mean: int = 50,
                 trace_length_variance: float = 12.5,
                 download_prob: float = 0.7,
                 inter_packet_time_mean: int = 100,
                 inter_packet_time_variance: float = 50.):
        if trace_length_mean <= 0:
            raise ValueError("The trace length mean must be positive")
        if trace_length_variance < 0:
            raise ValueError("The trace length variance must not be negative")
        if inter_packet_time_mean <= 0:
            raise ValueError("The inter packet time mean must be positive")
        if inter_packet_time_variance < 0:
            raise ValueError("The inter packet time variance must not be negative")
        if download_prob < 0 or download_prob > 1:
            raise ValueError("The download probability must be between 0 and 1")

        self.__trace_length_mean = trace_length_mean
        self.__trace_length_variance = trace_length_variance

        self.__inter_packet_time_mean = inter_packet_time_mean
        self.__inter_packet_time_variance = inter_packet_time_variance

        self.__download_prob = download_prob

    def generate(self,
                 *,
                 webpage_id: int,
                 count: int = 25,
                 trace_id_offset: int = 0,
                 trace_delta: timedelta = duration(days=1)) -> Iterator[pd.DataFrame]:
        now = pendulum.now(tz="UTC")

        collection_times = [now - trace_delta * (count - i - 1) for i in range(count)]

        for trace_id in range(trace_id_offset, trace_id_offset + count):
            yield self.__random_instance(webpage_id, trace_id, collection_times[trace_id - trace_id_offset])
            trace_id += 1

    def __random_instance(self, webpage_id: int, trace_id: int, collection_time: datetime) -> pd.DataFrame:
        length = max(1, round(gauss(self.__trace_length_mean, self.__trace_length_variance)))

        packets = []

        time = 0.
        for idx in range(length):
            size = randint(52, 1500)

            if idx == 0:
                direction = -1
            elif idx == 1:
                direction = 1
            else:
                direction = 1 if random() <= self.__download_prob else -1

            packets.append({
                "site_id": webpage_id,
                "trace_id": trace_id,
                "collection_time": pd.to_datetime(collection_time),
                "time": time,
                "size": direction * size
            })

            time += max(0., gauss(self.__inter_packet_time_mean, self.__inter_packet_time_variance))

        return pd.DataFrame(packets, columns=["site_id", "trace_id", "collection_time", "time", "size"])

    def __random_direction(self) -> int:
        return 1 if random() <= self.__download_prob else -1
