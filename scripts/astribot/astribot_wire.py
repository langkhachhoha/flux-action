"""Socket framing shared by the Astribot policy server and its ROS 2 client.

The ROS 2 client runs under the robot's Python 3.10 (rclpy of ROS 2 Humble) while FLUX Action needs
Python 3.12, so the two talk over a local TCP socket. The format needs only the standard library and
NumPy, and never unpickles anything:

    8-byte big-endian header length | UTF-8 JSON header | raw array bytes, in header order

    header = {"meta": {...}, "arrays": [{"name", "dtype", "shape", "nbytes"}, ...]}

policy_server.py and robot_client.py both import this file. The older client of
vmo_diffusion_policy_vla keeps a copy (diffusion_policy/real_world/flux_wire.py); the format is unchanged.
"""

from __future__ import annotations

import json
import socket
import struct

import numpy as np

_LENGTH = struct.Struct(">Q")
MAX_HEADER_BYTES = 1 << 20
MAX_ARRAY_BYTES = 1 << 30
ALLOWED_DTYPES = ("|u1", "<f4", "<f8", "<i4", "<i8")


def send_message(sock: socket.socket, meta: dict, arrays: dict[str, np.ndarray] | None = None) -> None:
    arrays = arrays or {}
    specs, payloads = [], []
    for name, value in arrays.items():
        array = np.ascontiguousarray(value)
        if array.dtype.str not in ALLOWED_DTYPES:
            raise ValueError(f"{name}: unsupported dtype {array.dtype.str}")
        specs.append(
            {"name": name, "dtype": array.dtype.str, "shape": list(array.shape), "nbytes": array.nbytes}
        )
        payloads.append(memoryview(array).cast("B"))
    header = json.dumps({"meta": meta, "arrays": specs}).encode()
    sock.sendall(_LENGTH.pack(len(header)) + header)
    for payload in payloads:
        sock.sendall(payload)


def _recv_exactly(sock: socket.socket, size: int) -> bytearray:
    buffer = bytearray(size)
    view = memoryview(buffer)
    received = 0
    while received < size:
        count = sock.recv_into(view[received:], size - received)
        if count == 0:
            raise ConnectionError("peer closed the connection")
        received += count
    return buffer


def recv_message(sock: socket.socket) -> tuple[dict, dict[str, np.ndarray]]:
    (length,) = _LENGTH.unpack(_recv_exactly(sock, _LENGTH.size))
    if length > MAX_HEADER_BYTES:
        raise ValueError(f"header of {length} bytes exceeds {MAX_HEADER_BYTES}")
    header = json.loads(_recv_exactly(sock, length).decode())
    arrays = {}
    for spec in header["arrays"]:
        dtype = np.dtype(spec["dtype"])
        if dtype.str not in ALLOWED_DTYPES:
            raise ValueError(f"{spec['name']}: unsupported dtype {dtype.str}")
        shape = tuple(int(n) for n in spec["shape"])
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if nbytes != spec["nbytes"] or nbytes > MAX_ARRAY_BYTES:
            raise ValueError(f"{spec['name']}: invalid payload size {spec['nbytes']}")
        arrays[spec["name"]] = np.frombuffer(_recv_exactly(sock, nbytes), dtype=dtype).reshape(shape)
    return header["meta"], arrays
