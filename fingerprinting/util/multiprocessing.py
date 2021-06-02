from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import AsyncIterable, TypeVar, Generic, AsyncIterator, Iterator

import curio
from asyncstdlib.itertools import aiter, anext, tee

T = TypeVar("T")


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
        self.__conn.send(n)
        self.__requested += n


class Manager(Generic[T]):
    def __init__(self, objects: AsyncIterable[T], n: int):
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
