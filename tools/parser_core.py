"""Pipeworks BDG parser (slim).

Originally developed by Digitzaki. Trimmed down to only what the viewer needs:
parse a Pipeworks bundle, list its files, extract file bytes, and overwrite a
file's bytes back into the bundle in place.
"""
from __future__ import annotations

import os
import struct


class PipeworksParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file_data: bytes | None = None
        self.is_big_endian = False
        self.string_offset = 0
        self.file_count = 0
        self.metadata_offset = 0
        self.main_data_offset = 0
        self.resource_data_offset = 0

    # ---- low level readers ----
    def read_bytes(self, offset, size):
        return self.file_data[offset:offset + size]

    def read_long(self, offset):
        endian = '>' if self.is_big_endian else '<'
        return struct.unpack(f'{endian}I', self.read_bytes(offset, 4))[0]

    def read_short(self, offset):
        endian = '>' if self.is_big_endian else '<'
        return struct.unpack(f'{endian}H', self.read_bytes(offset, 2))[0]

    def read_byte(self, offset):
        return self.file_data[offset]

    def read_long_little(self, offset):
        return struct.unpack('<I', self.read_bytes(offset, 4))[0]

    def read_string(self, offset):
        end = offset
        while end < len(self.file_data) and self.file_data[end] != 0:
            end += 1
        raw = self.file_data[offset:end].decode('ascii', errors='ignore')
        return ''.join(c for c in raw if 32 <= ord(c) <= 126).strip()

    # ---- parsing ----
    def get_file_info(self, file_num):
        try:
            metadata_start = self.metadata_offset + (file_num * 0x10)
            entry_offset = metadata_start + 0x2
            file_type = self.read_byte(entry_offset)
            str_id = self.read_long(entry_offset + 2)
            str_offset_pos = self.string_offset + (str_id * 0x4) + 0x4
            str_offset = self.read_long_little(str_offset_pos)
            string_pos = self.string_offset + str_offset
            name = self.read_string(string_pos) or f'file_{file_num}'
            return f'{file_type}/{name.replace("|", "_")}', file_type
        except Exception:
            return f'file_{file_num}', 0

    def parse(self):
        with open(self.filepath, 'rb') as f:
            self.file_data = f.read()
        if self.file_data[0:9].decode('ascii', errors='ignore') != 'Pipeworks':
            raise ValueError("Not a Pipeworks bundle (header missing)")

        endian_check = struct.unpack('<H', self.file_data[0x2C:0x2E])[0]
        if endian_check == 0:
            self.is_big_endian = True

        self.string_offset = self.read_long(0x34)
        self.file_count = self.read_short(0x62)
        self.metadata_offset = self.read_long(0x64)
        self.main_data_offset = self.read_long(0x68)
        self.resource_data_offset = self.read_long(0x70)

        toc_offset = 0x78
        results = []
        for i in range(self.file_count):
            entry_offset = toc_offset + (i * 0x12)
            file_num = self.read_short(entry_offset)
            offset = self.read_long(entry_offset + 2)
            size = self.read_long(entry_offset + 6)
            res_offset = self.read_long(entry_offset + 10)
            res_size = self.read_long(entry_offset + 14)

            name, file_type = self.get_file_info(file_num)
            results.append({
                'file_num': file_num, 'name': name, 'file_type': file_type,
                'offset': offset + self.main_data_offset, 'size': size,
                'is_resource': False, 'toc_entry_offset': entry_offset,
            })
            if res_size > 0:
                results.append({
                    'file_num': file_num, 'name': f'{name}.resource', 'file_type': file_type,
                    'offset': res_offset + self.resource_data_offset, 'size': res_size,
                    'is_resource': True, 'toc_entry_offset': entry_offset,
                })
        return results

    # ---- export / replace ----
    def extract_file(self, file_entry, output_dir):
        output_path = os.path.join(output_dir, file_entry['name'])
        folder = os.path.dirname(output_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        with open(output_path, 'wb') as f:
            f.write(self.read_bytes(file_entry['offset'], file_entry['size']))
        return output_path

    def replace_file_bytes(self, file_entry, new_bytes: bytes):
        """Patch new_bytes over file_entry in place. Same-size replacement only."""
        if len(new_bytes) != file_entry['size']:
            raise ValueError(
                f"replacement size {len(new_bytes)} != original {file_entry['size']}"
            )
        with open(self.filepath, 'r+b') as f:
            f.seek(file_entry['offset'])
            f.write(new_bytes)
        with open(self.filepath, 'rb') as f:
            self.file_data = f.read()
