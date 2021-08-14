import multiprocessing.pool
from multiprocessing import Pipe, Pool
from multiprocessing.connection import Connection
from os import cpu_count
from typing import AsyncIterable, TypeVar, Generic, AsyncIterator, Iterator, Callable, Iterable

import curio
from asyncstdlib.itertools import aiter, anext, tee

from .config import CONFIG

T = TypeVar("T")


class MultiprocessingConfig:
    def __init__(self, worker_processes: int = (cpu_count() or 1) * 2):
        if worker_processes < 1:
            raise ValueError("multiprocessing.worker_processes must be >= 1")
        self.__workers = worker_processes

    @property
    def workers(self):
        return self.__workers


_CONFIG = CONFIG.get_obj("multiprocessing").as_obj(MultiprocessingConfig)


def create_pool(processes: int = _CONFIG.workers) -> multiprocessing.pool.Pool:
    return Pool(processes=processes if processes >= 1 else _CONFIG.workers)


class WorkerPool:
    def __init__(self, processes: int = _CONFIG.workers):
        self.__pool = create_pool(processes) if processes > 1 else None

    async def submit(self, task: Callable[[], T]) -> T:
        if self.__pool is not None:
            e = curio.UniversalEvent()
            fut = self.__pool.apply_async(task, callback=lambda _: e.set(), error_callback=self.__on_err)
            await e.wait()

            return fut.get()
        else:
            return task()

    def __on_err(self, err):
        raise err

    def __enter__(self) -> 'WorkerPool':
        if self.__pool is not None:
            self.__pool.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__pool is not None:
            self.__pool.__exit__(exc_type, exc_val, exc_tb)


class ManagerAdapter(Generic[T]):
    def __init__(self, conn: Connection, objects: AsyncIterable[T]):
        self.__conn = conn
        self.__iter = aiter(objects)

    async def poll(self) -> bool:
        if self.__conn.poll():
            request = self.__conn.recv()

            for i in range(request):
                try:
                    self.__conn.send(await anext(self.__iter))
                except StopAsyncIteration as stop:
                    self.__conn.send(stop)
                    return False

        return True


class WorkerAdapter(Generic[T], AsyncIterator[T], Iterator[T]):
    def __init__(self, conn: Connection):
        self.__conn = conn
        self.__done = False
        self.__requested = 0

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        if self.__done:
            raise StopAsyncIteration()

        if self.__requested == 0:
            self.__request(1)

        while not self.__conn.poll():
            await curio.sleep(0.5)

        received = self.__conn.recv()
        self.__requested -= 1

        if isinstance(received, StopAsyncIteration):
            self.__done = True
            self.__requested = 0
            raise received
        else:
            return received

    def __iter__(self):
        return self

    def __next__(self) -> T:
        if self.__done:
            raise StopIteration()

        if self.__requested == 0:
            self.__request(1)

        self.__conn.poll(None)

        received = self.__conn.recv()
        self.__requested -= 1

        if isinstance(received, StopAsyncIteration):
            self.__done = True
            self.__requested = 0
            raise StopIteration()
        else:
            return received

    def __request(self, n: int):
        # print(self.__conn.fileno(), self.__conn.closed, self.__conn.readable, self.__conn.writable)
        self.__conn.send(n)
        self.__requested += n


class Manager(Generic[T]):
    def __init__(self, objects: Iterable[T], n: int):
        self.__cloned_iterable = tee(objects, n)
        self.n = n

        pipes = [Pipe() for _ in range(n)]
        self.__manager_conns = [pipe[0] for pipe in pipes]
        self.worker_conns = [pipe[1] for pipe in pipes]

        self.__adapters = [ManagerAdapter(self.__manager_conns[i], self.__cloned_iterable[i]) for i in range(n)]

    async def run(self):
        running = [True for _ in range(self.n)]

        while any(running):
            for i in range(self.n):
                if running[i]:
                    running[i] = await self.__adapters[i].poll()

            await curio.sleep(0.4)
