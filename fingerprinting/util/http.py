from logging import getLogger

import asks

from .fs import write_streaming_binary

_logger = getLogger(__name__)


async def download_to(url: str, target: str):
    _logger.info("Downloading %s to %s", url, target)
    response = await asks.get(url, stream=True)

    async with response.body as body:
        await write_streaming_binary(body,
                                     target,
                                     chunk_size=16384,
                                     full_size=int(response.headers.get("Content-Length")))
