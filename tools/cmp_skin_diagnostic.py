from __future__ import annotations

import argparse
import struct
import sys
from collections import Counter
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from cmg_probe import bundle_strings
from cmp_probe import find_cmp_packets, parse_cmp_skeleton
from parser_core import PipeworksParser


def le32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmp", type=Path)
    ap.add_argument("--hex", action="store_true")
    args = ap.parse_args()

    parser = PipeworksParser(str(args.cmp))
    entries = parser.parse()
    data = parser.file_data or b""
    strings = bundle_strings(parser)
    main_entry = next(e for e in entries if e["file_type"] == 17 and not e["is_resource"])
    res_entry = next(e for e in entries if e["file_type"] == 17 and e["is_resource"])
    main = data[main_entry["offset"] : main_entry["offset"] + main_entry["size"]]
    resource = data[res_entry["offset"] : res_entry["offset"] + res_entry["size"]]
    packets = find_cmp_packets(main, resource)
    bones, _globals = parse_cmp_skeleton(parser, entries, 1.0)

    table_count = le32(main, 0) >> 16 if len(main) >= 4 else 0
    print(f"main_size=0x{len(main):x} resource_size=0x{len(resource):x} bones={len(bones)} table_count={table_count}")
    print("skeleton:")
    for bone in bones:
        print(f"  {bone['idx']:3}: parent={bone['parent']:3} {bone['name']}")
    print("main_table:")
    for i in range(table_count):
        off = 0x10 + i * 4
        if off + 4 > len(main):
            break
        string_idx = le32(main, off)
        name = strings[string_idx] if 0 <= string_idx < len(strings) else "<invalid>"
        skeleton_idx = next((bone["idx"] for bone in bones if bone["name"] == name), None)
        print(f"  {i:3}: off=0x{off:04x} string={string_idx:4} skeleton={skeleton_idx!s:>4} {name}")

    for pi, packet in enumerate(packets):
        controls = []
        for i in range(packet["count"]):
            off = packet["rel"] + i * 16 + 12
            controls.append(struct.unpack_from("<HBB", resource, off))
        pairs = Counter((a, b) for _blend, a, b in controls)
        ids = Counter(v for _blend, a, b in controls for v in (a, b))
        print(
            f"packet{pi}: desc=0x{packet['desc']:x} rel=0x{packet['rel']:x} "
            f"count={packet['count']} ids={dict(sorted(ids.items()))}"
        )
        print(f"  pairs={dict(sorted(pairs.items()))}")
        desc_start = max(0, packet["desc"] - 0x20)
        desc_end = min(len(main), packet["desc"] + 0x100)
        print(f"  descriptor_words=0x{desc_start:x}..0x{desc_end:x}")
        for off in range(desc_start, desc_end, 4):
            print(f"    0x{off:04x}: 0x{le32(main, off):08x}")

    if args.hex:
        print("main_hex:")
        for off in range(0, len(main), 16):
            print(f"  {off:04x}: {main[off:off + 16].hex(' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
