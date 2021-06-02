import datetime
import os
import re
import tarfile
from logging import getLogger
from os import mkdir
from os.path import isdir, isfile, abspath, isabs, dirname, join
from typing import Awaitable, Optional, AsyncIterator

import progressbar
from curio.file import aopen

_logger = getLogger(__name__)


def mkdirs(path: str):
    if not isabs(path):
        path = abspath(path)

    if isdir(path):
        return

    if isfile(path):
        raise ValueError(f"'{path}' is a regular file, but was supposed to be a directory")

    mkdirs(dirname(path))
    mkdir(path)


def remove_recursive(path: str):
    if not isabs(path):
        path = abspath(path)

    if isfile(path):
        os.remove(path)
        return

    if isdir(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for f in files:
                os.remove(join(root, f))

            for d in dirs:
                os.removedirs(join(root, d))


async def write_binary(data: bytes, target: str):
    mkdirs(dirname(target))

    async with aopen(target, "wb") as f:
        await f.write(data)


async def write_async_binary(data: Awaitable[bytes], target: str):
    await write_binary(await data, target)


async def write_streaming_binary(data: AsyncIterator[bytes],
                                 target: str,
                                 chunk_size: int = 8192,
                                 full_size: Optional[int] = None):
    if chunk_size <= 0 or not isinstance(chunk_size, int):
        raise ValueError("The chunk size must be a positive integer")

    mkdirs(dirname(target))

    speed = progressbar.AdaptiveTransferSpeed(samples=datetime.timedelta(seconds=5))
    speed.INTERVAL = datetime.timedelta(milliseconds=500)

    if full_size is not None and full_size > 0 and isinstance(chunk_size, int):
        eta = progressbar.AdaptiveETA(samples=datetime.timedelta(seconds=5))
        eta.INTERVAL = datetime.timedelta(milliseconds=500)

        widgets = [
            ' ',
            progressbar.Timer(format='%(elapsed)s'),
            ' ',
            progressbar.Bar(left='[', right=']'),
            ' ',
            progressbar.Percentage(),
            ' - ',
            progressbar.DataSize(),
            '/',
            progressbar.DataSize('max_value'),
            ' @ ',
            speed,
            ' (',
            eta,
            ') ',
        ]
        bar = progressbar.ProgressBar(max_value=full_size, widgets=widgets, redirect_stdout=True)
    else:
        widgets = [
            ' ',
            progressbar.Timer(format='%(elapsed)s'), ' - ',
            progressbar.DataSize(), ' @ ', speed, ' - ',
            progressbar.AnimatedMarker()
        ]

        bar = progressbar.ProgressBar(widgets=widgets, redirect_stdout=True, redirect_stderr=True)

    async with aopen(target, "wb") as f:
        with bar:
            async for chunk in data:
                await f.write(chunk)
                bar += len(chunk)


async def write_text(data: str, target: str):
    mkdirs(dirname(target))

    async with aopen(target, "w") as f:
        await f.write(data)


async def write_async_text(data: Awaitable[str], target: str):
    await write_text(await data, target)


async def extract_tarball(file_path: str, target_path: str, compression: str = None):
    mode = "r|"

    if compression is not None and compression in {'bz2', 'gz', 'xz'}:
        mode += compression
    else:
        mode += "*"

    _logger.info("Extracting %s to %s", file_path, target_path)

    mkdirs(target_path)

    speed = progressbar.AdaptiveTransferSpeed(samples=datetime.timedelta(seconds=5))
    speed.INTERVAL = datetime.timedelta(milliseconds=500)

    eta = progressbar.AdaptiveETA(samples=datetime.timedelta(seconds=5))
    eta.INTERVAL = datetime.timedelta(milliseconds=500)

    bar = progressbar.ProgressBar(widgets=[
        ' ',
        progressbar.Timer(format='%(elapsed)s'), f' - Reading {file_path} @ ', speed, ' - ',
        progressbar.AnimatedMarker()
    ],
                                  redirect_stdout=True)

    with bar:
        bar.start()

        with __open_tarball(file_path, mode) as tar:
            for member in tar:
                member.name = re.sub(r'[:]', '_', member.name)

                if not member.isfile():
                    continue

                fileobj = tar.extractfile(member)

                if fileobj is None:
                    continue

                rel_path = member.name.split("/")
                target_file = join(target_path, *rel_path)

                mkdirs(dirname(target_file))

                async with aopen(target_file, "wb") as f:
                    while True:
                        chunk = fileobj.read()

                        if not chunk:
                            break

                        await f.write(chunk)

                bar += member.size


def __open_tarball(file_path: str, mode: str) -> tarfile.TarFile:
    return tarfile.open(file_path, mode)


def fmt_byte_size(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)
