#!/usr/bin/env python3

# lfgfx by Ethan Roseman (ethteck)

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type
from sty import Style, fg  # type: ignore
import threading

import n64img.image

from pygfxd import GfxdMacroId
from pygfxd import *  # type: ignore

DEBUG = True


def debug(msg: str) -> None:
    if DEBUG:
        print(msg)


class LFGFXLocal(threading.local):
    vram: int = 0
    found_objects: Dict[int, "Chunk"] = {}
    initialized: bool = False
    latest_macro: GfxdMacroId = None

    def __init__(self, **kw):
        if self.initialized:
            raise SystemError("__init__ called too many times")
        self.initialized = True
        self.__dict__.update(kw)

    def add_found_object(self, obj: "Chunk"):
        if (
            obj.start in self.found_objects
            and obj.type != self.found_objects[obj.start].type
        ):
            print(
                f"Duplicate objects found at 0x{obj.start:X}: {self.found_objects[obj.start]} and {obj} - ignoring the latter"
            )
        else:
            self.found_objects[obj.start] = obj


def auto_int(x):
    return int(x, 0)


parser = argparse.ArgumentParser(
    description="Analyze the Gfx / display lists in a binary file"
)
parser.add_argument("in_file", help="path to the input binary")
parser.add_argument(
    "vram",
    help="vram address at the given offset (or beginning of file)",
    type=auto_int,
)
parser.add_argument(
    "--mode",
    help="execution mode",
    choices=["simple", "splat"],
    default="simple",
)
parser.add_argument(
    "--start", help="start offset into the input file", default=0, type=auto_int
)
parser.add_argument(
    "--end",
    help="end offset into the input file. defaults to the end of the file",
    type=auto_int,
)
parser.add_argument(
    "--gfx-target",
    help="gfx target to use",
    choices=["f3d", "f3db", "f3dex", "f3dexb", "f3dex2"],
    default="f3dex2",
)
parser.add_argument(
    "--splat-rom-offset",
    help="start rom offset for this segment (for use with splat mode)",
    type=auto_int,
)
parser.add_argument(
    "--gfx",
    help="list of file offsets where known display lists begin",
    action="extend",
    nargs="+",
    type=auto_int,
)


class Chunk:
    start: int
    end: int
    type: str = "unmapped"
    splat_type: str = "type"
    display_color: Optional[Style] = None
    has_splat_extension: bool = False
    file_ext: str = "inc.c"

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def __repr__(self):
        return f"{self.type}: {self.start:X}-{self.end:X}"

    @property
    def addr(self):
        return thread_ctx.vram + self.start

    def type_color(self):
        if self.display_color:
            color = self.display_color
        elif self.type == "padding":
            color = fg.da_grey
        elif self.type == "unmapped":
            color = fg.li_red
        else:
            color = fg.white

        return color + self.type + fg.rs

    def symbol_name(self, rom_offset: int):
        return f"D_{self.addr:08X}_{rom_offset + self.start:06X}"

    def to_yaml(self, rom_offset):
        if self.has_splat_extension:
            return f"- [0x{rom_offset + self.start:X}, {self.splat_type}, {self.symbol_name(rom_offset)}]\n"
        else:
            return f"- [0x{rom_offset + self.start:X}] # {self.type}\n"

    def to_c(self, data: bytes, rom_offset: int):
        if self.has_splat_extension:
            return f'#include "effects/gfx/{self.symbol_name(rom_offset)}{self.file_ext}"\n'
        else:
            raw_c = f"u8 {self.symbol_name(rom_offset)}[] = " + "{\n"
            raw_c += "    " + ", ".join(f"0x{x:X}" for x in data[self.start : self.end])
            raw_c += "\n};\n"

            return raw_c

    def raw_chunk(self):
        return f"{self.start:X} - {self.end:X}"

    def __str__(self):
        ret = f"{self.type_color()}: 0x{self.start:X} - 0x{self.end:X}"

        if self.type == "unmapped":
            ret += f" (0x{self.end - self.start:X} bytes)"

        return ret


class Tlut(Chunk):
    type: str = "tlut"
    splat_type: str = "palette"
    display_color: Style = fg.li_green
    count: int
    has_splat_extension: bool = True
    file_ext: str = ".pal.inc.c"

    def __init__(self, start: int, end: int, idx: int, count: int):
        super().__init__(start, end)
        self.idx = idx
        self.count = count


class Timg(Chunk):
    type: str = "timg"
    display_color: Style = fg.li_cyan
    fmt: str
    size: int
    width: int
    height: int
    has_splat_extension: bool = True
    file_ext: str = ".png.inc.c"

    @property
    def splat_type(self):
        if self.fmt == 0:
            if self.size == 2:
                return "rgba16"
            elif self.size == 3:
                return "rgba32"
        elif self.fmt == 2:
            if self.size == 0:
                return "ci4"
            elif self.size == 1:
                return "ci8"
        elif self.fmt == 3:
            if self.size == 0:
                return "ia4"
            elif self.size == 1:
                return "ia8"
            elif self.size == 2:
                return "ia16"
        elif self.fmt == 4:
            if self.size == 0:
                return "i4"
            elif self.size == 1:
                return "i8"
        raise RuntimeError(f"Unknown format / size {self.fmt}, {self.size}")

    def resize(self, new_end: int):
        self.end = new_end
        self.height = (self.end - self.start) // self.width * (2 - self.size)

    def to_file(self, data: bytes) -> None:
        outname = Path("timg") / f"{self.start:X}.png"

        imgcls: Optional[Type[n64img.image.Image]] = None
        if self.splat_type == "rgba16":
            imgcls = n64img.image.RGBA16
        elif self.splat_type == "rgba32":
            imgcls = n64img.image.RGBA32
        elif self.splat_type == "ci4":
            imgcls = n64img.image.CI4
        elif self.splat_type == "ci8":
            imgcls = n64img.image.CI8
        elif self.splat_type == "ia4":
            imgcls = n64img.image.IA4
        elif self.splat_type == "ia8":
            imgcls = n64img.image.IA8
        elif self.splat_type == "ia16":
            imgcls = n64img.image.IA16
        elif self.splat_type == "i4":
            imgcls = n64img.image.I4
        elif self.splat_type == "i8":
            imgcls = n64img.image.I8

        if imgcls is None:
            raise RuntimeError(f"Unknown format / size {self.fmt}, {self.size}")

        imgcls(data[self.start : self.end], self.width, self.height).write(outname)

    def to_yaml(self, rom_offset):
        return f"- [0x{rom_offset + self.start:X}, {self.splat_type}, {self.symbol_name(rom_offset)}, {self.width}, {self.height}]\n"

    def __init__(
        self,
        start: int,
        end: int,
        fmt: str,
        size: int,
        width: int,
        height: int,
    ):
        super().__init__(start, end)
        self.fmt = fmt
        self.size = size
        self.width = width
        self.height = height


class Dlist(Chunk):
    type: str = "dlist"
    splat_type: str = "gfx"
    display_color: Style = fg.li_blue
    has_splat_extension: bool = True
    file_ext: str = ".gfx.inc.c"


class Vtx(Chunk):
    type: str = "vtx"
    splat_type: str = "vtx"
    display_color: Style = fg.li_green
    count: int
    has_splat_extension: bool = True
    file_ext: str = ".vtx.inc.c"

    def __init__(self, start: int, end: int, count: int):
        super().__init__(start, end)
        self.count = count


def macro_fn():
    gfxd_puts("    ")
    gfxd_macro_dflt()
    gfxd_puts(",\n")

    thread_ctx.latest_macro = gfxd_macro_id()
    return 0


def tlut_handler(addr, idx, count):
    gfxd_printf(f"D_{addr:08X}")

    start = addr - thread_ctx.vram
    end = start + count * 2
    tlut = Tlut(start, end, idx, count)
    thread_ctx.add_found_object(tlut)
    return 1


def timg_handler(addr, fmt, size, width, height, pal):
    gfxd_printf(f"D_{addr:08X}")

    if height == 0 or height == -1:
        # Guess height
        height = width

    num_bytes = width * height

    # Too small
    if num_bytes < 8:
        return 0

    if size == 0:
        num_bytes /= 2
    elif size == 1:
        num_bytes *= 1
    elif size == 2:
        num_bytes *= 2
    elif size == 3:
        num_bytes *= 4
    else:
        print(f"Unknown timg size format {size}")
        return 0

    start = addr - thread_ctx.vram

    if start < 0:
        return 0

    end = int(start + num_bytes)
    timg = Timg(start, end, fmt, size, width, height)
    thread_ctx.add_found_object(timg)
    return 1


def cimg_handler(addr, fmt, size, width):
    gfxd_printf(f"D_{addr:08X}")

    if addr < 0xFFFFFFFF:
        pass
        # print(f"cimg at 0x{addr:08X}, fmt {fmt}, size {size}, width {width}")
    return 1


def zimg_handler(addr):
    gfxd_printf(f"D_{addr:08X}")
    print(f"zimg at 0x{addr:08X}")
    return 1


def dl_handler(addr):
    gfxd_printf(f"D_{addr:08X}")
    start = addr - thread_ctx.vram
    thread_ctx.add_found_object(Dlist(start, start))
    return 1


def mtx_handler(addr):
    gfxd_printf(f"D_{addr:08X}")
    print(f"mtx at 0x{addr:08X}")
    return 1


def lookat_handler(addr, count):
    gfxd_printf(f"D_{addr:08X}")
    print(f"lookat at 0x{addr:08X}, count {count}")
    return 1


def light_handler(addr, count):
    gfxd_printf(f"D_{addr:08X}")
    print(f"light at 0x{addr:08X}, count {count}")
    return 1


def vtx_handler(addr, count):
    gfxd_printf(f"D_{addr:08X}")

    start = addr - thread_ctx.vram
    end = start + count * 0x10
    vtx = Vtx(start, end, count)
    thread_ctx.add_found_object(vtx)
    return 1


def vp_handler(addr):
    gfxd_printf(f"D_{addr:08X}")
    print(f"vp at 0x{addr:08X}")
    return 1


def gfxd_scan_bytes(data: bytes) -> int:
    gfxd_input_buffer(data)  # type: ignore
    return gfxd_execute()  # type: ignore


def is_bad_command(data: bytes) -> bool:
    gfxd_scan_bytes(data)

    if thread_ctx.latest_macro == GfxdMacroId.DPNoOp:
        return True

    if thread_ctx.latest_macro == GfxdMacroId.BranchZ:
        return True

    if thread_ctx.latest_macro == GfxdMacroId.SPModifyVertex:
        if data[0] == 0x02:
            if int.from_bytes(data[2:4], byteorder="big") > 79:
                return True

    return False


def valid_dlist(data: bytes) -> int:
    return gfxd_scan_bytes(data) != -1  # type: ignore


def find_earliest_start(
    data: bytes, min: int, end: int, known_dlists: List[int]
) -> int:
    for i in range(end - 8, min, -8):
        if i in known_dlists:
            # scan the first command since it may reference an object
            gfxd_scan_bytes(data[i : i + 8])
            return i
        if is_bad_command(data[i : i + 8]) or not valid_dlist(data[i:end]):
            return i + 8
    if i == min + 8:
        # We know the dlist starts at min
        # scan the first command since it may reference an object
        gfxd_scan_bytes(data[min : min + 8])
    return min


def get_end_dlist_cmd(gfx_target):
    if gfx_target == gfxd_f3dex2:
        return b"\xDF\x00\x00\x00\x00\x00\x00\x00"
    else:
        return b"\xB8\x00\x00\x00\x00\x00\x00\x00"


def collect_dlists(data: bytes, gfx_target, known_dlists: List[int]) -> List[Dlist]:
    ret: List[Dlist] = []
    ends: List[int] = []

    for i in range(0, len(data), 8):
        if data[i : i + 8] == get_end_dlist_cmd(gfx_target):
            ends.append(i + 8)

    min = 0
    for end in ends:
        start = find_earliest_start(data, min, end, known_dlists)
        ret.append(Dlist(start, end))
        min = end

    return ret


def is_zeros(data: bytes) -> bool:
    for i in range(len(data)):
        if data[i] != 0:
            return False
    return True


def pygfxd_init(target):
    gfxd_target(target)
    gfxd_macro_fn(macro_fn)

    # callbacks
    gfxd_tlut_callback(tlut_handler)
    gfxd_timg_callback(timg_handler)
    gfxd_cimg_callback(cimg_handler)  # TODO
    gfxd_zimg_callback(zimg_handler)  # TODO
    gfxd_dl_callback(dl_handler)
    gfxd_mtx_callback(mtx_handler)  # TODO
    gfxd_lookat_callback(lookat_handler)  # TODO
    gfxd_light_callback(light_handler)  # TODO
    # gfxd_seg_callback ?
    gfxd_vtx_callback(vtx_handler)
    gfxd_vp_callback(vp_handler)  # TODO
    # gfxd_uctext_callback ?
    # gfxd_ucdata_callback ?
    # gfxd_dram_callback ?


def target_arg_to_object(arg: str):
    if arg == "f3d":
        return gfxd_f3d  # type: ignore
    elif arg == "f3db":
        return gfxd_f3db  # type: ignore
    elif arg == "f3dex":
        return gfxd_f3dex  # type: ignore
    elif arg == "f3dexb":
        return gfxd_f3dexb  # type: ignore
    elif arg == "f3dex2":
        return gfxd_f3dex2  # type: ignore
    else:
        raise RuntimeError(f"Unknown target {arg}")


def splat_chunks(chunks: List[Chunk], data: bytes, rom_offset: int) -> Tuple[str, str]:
    yaml = ""
    c_code = ""

    empty_line = True
    for i, chunk in enumerate(chunks):
        yaml += chunk.to_yaml(rom_offset)

        if not chunk.has_splat_extension and not empty_line:
            c_code += "\n"

        c_code += chunk.to_c(data, rom_offset)

        if chunk.has_splat_extension:
            empty_line = False
        else:
            c_code += "\n"
            empty_line = True

    return yaml, c_code


def scan_binary(data: bytes, vram, gfx_target, known_dlists: List[int]) -> List[Chunk]:
    pygfxd_init(gfx_target)

    thread_ctx.found_objects.clear()
    thread_ctx.vram = vram

    chunks: List[Chunk] = []

    # dlists
    dlists: List[Dlist] = collect_dlists(data, gfx_target, known_dlists)

    chunks.extend(dlists)
    chunks.extend(thread_ctx.found_objects.values())

    chunks.sort(key=lambda x: x.start)

    # Truncate things as needed
    for i, chunk in enumerate(chunks):
        if i < len(chunks) - 1 and chunk.end > chunks[i + 1].start:
            if isinstance(chunk, Timg):
                # Allow truncation of images
                debug(f"Truncating {chunk} to end at 0x{chunks[i + 1].start:X}")
                chunk.resize(chunks[i + 1].start)
            elif isinstance(chunk, Tlut):
                # Allow truncation of palettes
                debug(f"Truncating {chunk} to end at 0x{chunks[i + 1].start:X}")
                chunk.end = chunks[i + 1].start
            elif isinstance(chunk, Vtx):
                # TODO conditionally truncate vtx blobs
                # raise RuntimeError("Vtx chunk overlap!")
                pass
            #     # Allow truncation of vtx blobs under certain conditions
            #     if (chunks[i + 1].start - chunk.start) % 0x10 == 0:
            #         debug(f"Truncating {chunk} to end at 0x{chunks[i + 1].start:X}")
            #         chunk.end = chunks[i + 1].start
            #     else:
            #         raise RuntimeError(f"Cannot truncate {chunk} to end at 0x{chunks[i + 1].start:X} - not aligned to 0x10 bytes!")
            else:
                # raise RuntimeError("Chunk overlap!")
                pass

    # Fill gaps
    to_add: List[Chunk] = []
    for i, chunk in enumerate(chunks):
        if i < len(chunks) - 1 and chunk.end < chunks[i + 1].start:
            to_add.append(Chunk(chunk.end, chunks[i + 1].start))
    chunks.extend(to_add)
    chunks.sort(key=lambda x: x.start)

    # Add a chunk for the beginning of the file
    if len(chunks) == 0:
        chunks.insert(0, Chunk(0, len(data)))
    elif chunks[0].start != 0:
        chunks.insert(0, Chunk(0, chunks[0].start))

    # Add a chunk for the rest of the file
    end_pos = chunks[len(chunks) - 1].end
    if end_pos < len(data):
        chunks.append(Chunk(end_pos, len(data)))

    # Remove chunks that are empty
    chunks = [chunk for chunk in chunks if chunk.start < chunk.end]

    # Mark chunks that are filled with 0s as padding
    for chunk in chunks:
        if chunk.type == "unmapped" and is_zeros(data[chunk.start : chunk.end]):
            chunk.type = "padding"

    return chunks


def main(args):
    with open(args.in_file, "rb") as f:
        input_bytes = f.read()

    start = args.start
    end = len(input_bytes) if args.end is None else args.end
    gfx_target = target_arg_to_object(args.gfx_target)

    print(f"Scanning input binary {args.in_file} from 0x{start:X} to 0x{end:X}")

    chunks = scan_binary(input_bytes[start:end], args.vram, gfx_target, args.gfx)

    if args.mode == "simple":
        for chunk in chunks:
            print(chunk)

            if isinstance(chunk, Timg):
                chunk.to_file(input_bytes)

    elif args.mode == "splat":
        if args.splat_rom_offset is None:
            raise RuntimeError("Must specify --splat-rom-offset with splat mode")
        splat_yaml, c_code = splat_chunks(chunks, input_bytes, args.splat_rom_offset)
        print("\nSplat yaml:\n")
        print(splat_yaml)
        print("C code:\n")
        print(c_code)
    else:
        raise RuntimeError(f"Unsupported mode {args.mode}")


thread_ctx = LFGFXLocal()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
