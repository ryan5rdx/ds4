#!/usr/bin/env python3
"""Requantise an already-built GLM 5.3 Flash GGUF in place of a full reconvert.

Why this exists.  `glm53_quantize.py` builds an artifact from the HF checkpoint,
which is a ~700 GB download the rig does not need for this.  The published
`GLM-5.3-Flash-Q4_K.gguf` predates the commit that started rewriting BF16 dense
weights to Q8_0 for the q4 artifact ("Quantize GLM 5.3 Q4 dense weights to Q8"),
so its KDA projections, output head and token embedding are still BF16 -- 459 of
its 1412 tensors, and the largest single block of avoidable decode traffic on a
tensor-parallel pair.  Those tensors are verbatim copies of the source BF16
weights, so quantising them here goes through exactly the same float32 -> Q8_0
path `glm53_quantize.py` would have used, from exactly the same inputs, and
lands on the same bytes.  Nothing else in the file is touched.

Usage:
    python3 glm53_requantize_gguf.py IN.gguf OUT.gguf [--artifact q4]
    python3 glm53_requantize_gguf.py --self-test

The shared quantiser library is built by `make libds4quants.dylib` (or .so) in
this directory; pass --library to override the search.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import struct
import sys

GGUF_MAGIC = b"GGUF"
GGUF_ALIGNMENT = 32

QTYPE_F32 = 0
QTYPE_F16 = 1
QTYPE_Q8_0 = 8
QTYPE_BF16 = 30

QTYPE_NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 10: "Q2_K", 12: "Q4_K",
               14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 24: "I8", 30: "BF16"}

# GGUF metadata value types -> fixed width, for the ones that have one.
KV_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
KV_STRING = 8
KV_ARRAY = 9


def fail(message):
    print(f"glm53-requantize: error: {message}", file=sys.stderr)
    sys.exit(1)


def align_up(value, alignment=GGUF_ALIGNMENT):
    return ((value + alignment - 1) // alignment) * alignment


# ---------------------------------------------------------------------------
# Role classification.  Mirrors glm53_quantize.py's build_plan/regular_qtype,
# but keyed on the GGUF tensor name because that is all this tool can see.
# Only BF16 tensors are ever rewritten, which makes the rule self-limiting: the
# F32 conv/bias/norm tensors in a KDA block cannot be caught by accident.
# ---------------------------------------------------------------------------

def target_qtype(name, source_qtype, artifact):
    if artifact not in ("q2", "q4") or source_qtype != QTYPE_BF16:
        return source_qtype
    if name == "token_embd.weight" or name == "output.weight":
        return QTYPE_Q8_0
    if ".kda_" in name:
        # q2 keeps kda_q/kda_k at Q4_K; this tool only ever promotes to Q8_0,
        # so refuse rather than silently produce a different artifact.
        if artifact == "q2" and (name.endswith(".kda_q.weight") or
                                 name.endswith(".kda_k.weight")):
            fail(f"{name}: q2 wants Q4_K here, which this tool does not emit; "
                 "rebuild that artifact with glm53_quantize.py")
        return QTYPE_Q8_0
    return source_qtype


# ---------------------------------------------------------------------------
# GGUF header parsing.  The metadata block is copied through byte for byte, so
# it is only walked far enough to find where it ends.
# ---------------------------------------------------------------------------

class Reader:
    def __init__(self, fp):
        self.fp = fp

    def raw(self, n):
        data = self.fp.read(n)
        if len(data) != n:
            fail("truncated GGUF header")
        return data

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self):
        return self.raw(self.u64())

    def skip_value(self, vtype):
        if vtype == KV_STRING:
            self.string()
        elif vtype == KV_ARRAY:
            etype = self.u32()
            count = self.u64()
            if etype in KV_FIXED:
                self.fp.seek(KV_FIXED[etype] * count, os.SEEK_CUR)
            else:
                for _ in range(count):
                    self.skip_value(etype)
        elif vtype in KV_FIXED:
            self.fp.seek(KV_FIXED[vtype], os.SEEK_CUR)
        else:
            fail(f"unknown GGUF metadata value type {vtype}")


def read_header(fp):
    r = Reader(fp)
    if r.raw(4) != GGUF_MAGIC:
        fail("not a GGUF file")
    version = r.u32()
    if version != 3:
        fail(f"unsupported GGUF version {version}")
    n_tensors = r.u64()
    n_kv = r.u64()
    kv_start = fp.tell()
    for _ in range(n_kv):
        r.string()
        r.skip_value(r.u32())
    kv_end = fp.tell()

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.u64() for _ in range(r.u32())]
        qtype = r.u32()
        offset = r.u64()
        tensors.append({"name": name, "dims": dims, "qtype": qtype,
                        "offset": offset})
    info_end = fp.tell()
    return version, kv_start, kv_end, tensors, info_end


def encode_tensor_info(t):
    out = struct.pack("<Q", len(t["name"])) + t["name"]
    out += struct.pack("<I", len(t["dims"]))
    for d in t["dims"]:
        out += struct.pack("<Q", d)
    out += struct.pack("<I", t["qtype"])
    out += struct.pack("<Q", t["offset"])
    return out


# ---------------------------------------------------------------------------
# Quantiser binding.
# ---------------------------------------------------------------------------

class Quantiser:
    def __init__(self, library_path):
        try:
            import numpy as np
        except ImportError as error:
            fail(f"NumPy is required: {error}")
        self.np = np
        self.lib = ctypes.CDLL(library_path)
        self.lib.ds4q_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.lib.ds4q_row_size.restype = ctypes.c_size_t
        self.lib.ds4q_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ds4q_quantize_chunk.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float)]
        self.lib.ds4q_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ds4q_quantize_init(QTYPE_Q8_0)

    def row_size(self, qtype, ncols):
        return self.lib.ds4q_row_size(qtype, ncols)

    def quantise_rows(self, qtype, f32_rows, nrows, ncols):
        """f32_rows is a contiguous float32 numpy array of nrows*ncols."""
        out = bytearray(self.row_size(qtype, ncols) * nrows)
        buf = (ctypes.c_char * len(out)).from_buffer(out)
        src = f32_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        written = self.lib.ds4q_quantize_chunk(
            qtype, src, ctypes.cast(buf, ctypes.c_void_p),
            0, nrows, ncols, None)
        del buf
        if written != len(out):
            fail(f"quantiser wrote {written} bytes, expected {len(out)}")
        return bytes(out)


def bf16_to_f32(np, raw):
    bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return bits.view(np.float32)


# ---------------------------------------------------------------------------
# Conversion.
# ---------------------------------------------------------------------------

def tensor_nbytes(quantiser, qtype, dims):
    ncols = dims[0]
    nrows = 1
    for d in dims[1:]:
        nrows *= d
    return quantiser.row_size(qtype, ncols) * nrows


def convert(in_path, out_path, artifact, library, chunk_rows, verbose=True):
    quantiser = Quantiser(library)
    np = quantiser.np

    with open(in_path, "rb") as fp:
        version, kv_start, kv_end, tensors, info_end = read_header(fp)
        fp.seek(0)
        prologue = fp.read(kv_end)      # magic..end of metadata, copied verbatim

    data_start_in = align_up(info_end)

    plan = []
    for t in tensors:
        name = t["name"].decode("utf-8", "replace")
        new_type = target_qtype(name, t["qtype"], artifact)
        plan.append({**t, "text": name, "new_qtype": new_type,
                     "in_nbytes": tensor_nbytes(quantiser, t["qtype"], t["dims"]),
                     "out_nbytes": tensor_nbytes(quantiser, new_type, t["dims"])})

    changed = [p for p in plan if p["new_qtype"] != p["qtype"]]
    if verbose:
        saved = sum(p["in_nbytes"] - p["out_nbytes"] for p in changed)
        print(f"tensors: {len(plan)}, rewriting {len(changed)}, "
              f"saving {saved / (1 << 30):.2f} GiB")

    # Lay out the new data section, then the new tensor-info table.  The table
    # is fixed-width per tensor regardless of offsets, so one pass is enough.
    cursor = 0
    for p in plan:
        p["new_offset"] = cursor
        cursor = align_up(cursor + p["out_nbytes"])
    data_bytes = cursor

    info_blob = b"".join(
        encode_tensor_info({"name": p["name"], "dims": p["dims"],
                            "qtype": p["new_qtype"], "offset": p["new_offset"]})
        for p in plan)
    data_start_out = align_up(kv_end + len(info_blob))

    with open(in_path, "rb") as src, open(out_path, "wb") as dst:
        dst.write(prologue)
        dst.write(info_blob)
        dst.write(b"\0" * (data_start_out - dst.tell()))

        for i, p in enumerate(plan):
            src.seek(data_start_in + p["offset"])
            assert dst.tell() == data_start_out + p["new_offset"]
            if p["new_qtype"] == p["qtype"]:
                remaining = p["in_nbytes"]
                while remaining:
                    n = min(remaining, 64 << 20)
                    dst.write(src.read(n))
                    remaining -= n
            else:
                ncols = p["dims"][0]
                nrows = p["in_nbytes"] // (ncols * 2)   # BF16 source
                done = 0
                while done < nrows:
                    batch = min(chunk_rows, nrows - done)
                    raw = src.read(batch * ncols * 2)
                    f32 = np.ascontiguousarray(bf16_to_f32(np, raw))
                    dst.write(quantiser.quantise_rows(
                        p["new_qtype"], f32, batch, ncols))
                    done += batch
                if verbose:
                    print(f"  [{i + 1}/{len(plan)}] {p['text']}: "
                          f"{QTYPE_NAMES.get(p['qtype'], p['qtype'])} -> "
                          f"{QTYPE_NAMES.get(p['new_qtype'], p['new_qtype'])}")
            pad = align_up(dst.tell()) - dst.tell()
            if pad:
                dst.write(b"\0" * pad)

    if verbose:
        print(f"wrote {out_path}: {os.path.getsize(out_path) / (1 << 30):.2f} GiB "
              f"(was {os.path.getsize(in_path) / (1 << 30):.2f} GiB), "
              f"data section {data_bytes / (1 << 30):.2f} GiB")
    return plan


# ---------------------------------------------------------------------------
# Self-test: build a miniature GGUF with the tensor names that matter, run the
# conversion, and check the types, the sizes and the round-trip error.  Runs
# without the real model, which is the only way this is testable off the rig.
# ---------------------------------------------------------------------------

def self_test(library):
    import tempfile
    try:
        import numpy as np
    except ImportError as error:
        fail(f"NumPy is required: {error}")

    quantiser = Quantiser(library)
    rng = np.random.default_rng(7)
    specs = [
        ("token_embd.weight", [256, 32], QTYPE_BF16, QTYPE_Q8_0),
        ("blk.0.kda_q.weight", [256, 64], QTYPE_BF16, QTYPE_Q8_0),
        ("blk.0.kda_output.weight", [64, 256], QTYPE_BF16, QTYPE_Q8_0),
        ("blk.0.kda_a_log", [64], QTYPE_F32, QTYPE_F32),
        ("blk.3.indexer.attn_k.weight", [256, 32], QTYPE_BF16, QTYPE_BF16),
        ("blk.3.attn_output.weight", [256, 64], QTYPE_Q8_0, QTYPE_Q8_0),
        ("output.weight", [256, 48], QTYPE_BF16, QTYPE_Q8_0),
    ]

    payloads = {}
    for name, dims, qtype, _ in specs:
        n = 1
        for d in dims:
            n *= d
        if qtype == QTYPE_BF16:
            values = rng.standard_normal(n).astype(np.float32)
            bits = (values.view(np.uint32) >> 16).astype("<u2")
            payloads[name] = (bits.tobytes(), values)
        elif qtype == QTYPE_F32:
            values = rng.standard_normal(n).astype(np.float32)
            payloads[name] = (values.tobytes(), values)
        else:
            values = rng.standard_normal(n).astype(np.float32)
            rows = n // dims[0]
            payloads[name] = (quantiser.quantise_rows(qtype, values, rows,
                                                      dims[0]), values)

    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, "mini.gguf")
        dst_path = os.path.join(tmp, "mini.q8.gguf")

        header = bytearray()
        header += GGUF_MAGIC + struct.pack("<I", 3)
        header += struct.pack("<Q", len(specs)) + struct.pack("<Q", 1)
        key = b"general.architecture"
        header += struct.pack("<Q", len(key)) + key
        header += struct.pack("<I", KV_STRING)
        val = b"glm5-next"
        header += struct.pack("<Q", len(val)) + val

        infos = bytearray()
        cursor = 0
        for name, dims, qtype, _ in specs:
            infos += encode_tensor_info({"name": name.encode(), "dims": dims,
                                         "qtype": qtype, "offset": cursor})
            cursor = align_up(cursor + len(payloads[name][0]))
        data_start = align_up(len(header) + len(infos))

        with open(src_path, "wb") as fp:
            fp.write(header)
            fp.write(infos)
            fp.write(b"\0" * (data_start - fp.tell()))
            for name, dims, qtype, _ in specs:
                fp.write(payloads[name][0])
                pad = align_up(fp.tell()) - fp.tell()
                if pad:
                    fp.write(b"\0" * pad)

        convert(src_path, dst_path, "q4", library, 64, verbose=False)

        with open(dst_path, "rb") as fp:
            _, _, kv_end, tensors, info_end = read_header(fp)
            got = {t["name"].decode(): t for t in tensors}
            for name, dims, _, expect in specs:
                if got[name]["qtype"] != expect:
                    fail(f"self-test: {name} became "
                         f"{QTYPE_NAMES.get(got[name]['qtype'])}, expected "
                         f"{QTYPE_NAMES.get(expect)}")
                if got[name]["dims"] != dims:
                    fail(f"self-test: {name} shape changed")
            # Q8_0 of BF16 must land within a block quantisation step of the
            # original, which is what proves we fed the quantiser real values
            # and not, say, byte-swapped ones.
            data_start = align_up(info_end)
            t = got["blk.0.kda_q.weight"]
            fp.seek(data_start + t["offset"])
            ncols = t["dims"][0]
            nrows = 1
            for d in t["dims"][1:]:
                nrows *= d
            blob = fp.read(quantiser.row_size(QTYPE_Q8_0, ncols) * nrows)
            blocks = np.frombuffer(blob, dtype=np.uint8).reshape(-1, 34)
            scales = blocks[:, :2].copy().view("<f2").astype(np.float32).ravel()
            qs = blocks[:, 2:].view(np.int8).astype(np.float32)
            recon = (qs * scales[:, None]).ravel()
            original = payloads["blk.0.kda_q.weight"][1]
            err = float(np.max(np.abs(recon - original)))
            step = float(np.max(np.abs(original))) / 127.0
            if not err <= step:
                fail(f"self-test: round-trip error {err:.5f} exceeds one "
                     f"quantisation step {step:.5f}")

    print("glm53-requantize self-test: ok")


def default_library():
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in ("libds4quants.dylib", "libds4quants.so"):
        path = os.path.join(here, candidate)
        if os.path.exists(path):
            return path
    fail("libds4quants not found; run `make libds4quants.dylib` in gguf-tools")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", help="input GGUF")
    parser.add_argument("dest", nargs="?", help="output GGUF")
    parser.add_argument("--artifact", default="q4", choices=("q2", "q4"))
    parser.add_argument("--library", default=None)
    parser.add_argument("--chunk-rows", type=int, default=4096,
                        help="rows converted per read; bounds peak memory")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    library = args.library or default_library()
    if args.self_test:
        self_test(library)
        return
    if not args.source or not args.dest:
        parser.error("source and dest are required unless --self-test")
    if os.path.exists(args.dest):
        fail(f"{args.dest} already exists")
    convert(args.source, args.dest, args.artifact, library, args.chunk_rows)


if __name__ == "__main__":
    main()
