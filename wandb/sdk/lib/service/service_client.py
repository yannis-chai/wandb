from __future__ import annotations

import asyncio
import logging
import struct

from wandb.proto import wandb_server_pb2 as spb
from wandb.sdk.lib import asyncio_manager
from wandb.sdk.mailbox.mailbox import Mailbox
from wandb.sdk.mailbox.mailbox_handle import MailboxHandle

_logger = logging.getLogger(__name__)

_HEADER_BYTE_INT_LEN = 5
_HEADER_BYTE_INT_FMT = "<BI"


class BrokenClientError(Exception):
    """The socket broke and cannot be used.

    After a socket operation raises an exception, calls to ServiceClient methods
    raise this exception immediately. This is necessary because asyncio's
    StreamReader and StreamWriter store and re-raise an exception, which
    results in huge, repetitive tracebacks. See
    https://bugs.python.org/issue45924.
    """


class ServiceClient:
    """Implements socket communication with the internal service."""

    def __init__(
        self,
        asyncer: asyncio_manager.AsyncioManager,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._broken = False

        self._reader = reader
        self._writer = writer
        self._mailbox = Mailbox(asyncer, self._cancel_request)
        asyncer.run_soon(
            self._forward_responses,
            daemon=True,
            name="ServiceClient._forward_responses",
        )

    async def publish(self, request: spb.ServerRequest) -> None:
        """Send a request without waiting for a response."""
        await self._send_server_request(request)

    async def deliver(
        self,
        request: spb.ServerRequest,
    ) -> MailboxHandle[spb.ServerResponse]:
        """Send a request and return a handle to wait for a response.

        NOTE: This may mutate the request. The request should not be used
        after.

        Raises:
            MailboxClosedError: If used after the client is closed or has
                stopped due to an error.
        """
        handle = self._mailbox.require_response(request)
        await self._send_server_request(request)
        return handle

    async def _send_server_request(self, request: spb.ServerRequest) -> None:
        if self._broken:
            raise BrokenClientError

        header = struct.pack(_HEADER_BYTE_INT_FMT, ord("W"), request.ByteSize())
        self._writer.write(header)

        data = request.SerializeToString()
        self._writer.write(data)

        try:
            await self._writer.drain()
        except:
            self._broken = True
            raise

    async def _cancel_request(self, id: str, /) -> None:
        """Cancel a request by ID.

        Args:
            id: The request_id of a previously-sent ServerRequest.
        """
        await self.publish(
            spb.ServerRequest(
                cancel=spb.ServerCancelRequest(
                    request_id=id,
                )
            )
        )

    async def close(self) -> None:
        """Flush and close the socket."""
        self._writer.close()
        await self._writer.wait_closed()

    async def _forward_responses(self) -> None:
        try:
            while response := await self._read_server_response():
                await self._mailbox.deliver(response)

        except Exception:
            _logger.exception("Error reading server response.")

        else:
            _logger.info("Reached EOF.")

        finally:
            self._mailbox.close()

    async def _read_server_response(self) -> spb.ServerResponse | None:
        try:
            header = await self._reader.readexactly(_HEADER_BYTE_INT_LEN)
        except asyncio.IncompleteReadError as e:
            if e.partial:
                raise
            else:
                return None

        magic, length = struct.unpack(_HEADER_BYTE_INT_FMT, header)

        if magic != ord("W"):
            raise ValueError(f"Bad header: {header.hex()}")

        data = await self._reader.readexactly(length)
        response = spb.ServerResponse()
        response.ParseFromString(data)
        return response
