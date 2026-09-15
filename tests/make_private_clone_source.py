#!/usr/bin/env python3
"""Assemble a Metal source the Xcode 14.2 frontend can compile.

SGASYNC needs three controls per experiment: modern shipping, an Xcode-14.2
NON-async clone, and an Xcode-14.2 async clone. Without the middle one a
compiler-version difference is indistinguishable from an async-copy result.

Getting the middle one means compiling the real kernels with the old frontend.
A handful of them use device-scope atomics (`thread_scope_device`,
`thread_scope_system`, three-argument `atomic_thread_fence`,
`mem_device_and_threadgroup`) that the 14.2 frontend does not have. None of them
is an SGASYNC target.

Rather than edit the shipping .metal files -- where a misplaced guard silently
drops a kernel from the PRODUCTION library -- this elides the offending
definitions from a copy, and derives which ones by compiling in a loop:
compile, take each error line, expand it to its enclosing top-level definition,
elide, repeat. Shipping sources are never touched, and the elision cannot go
stale against them.

An elided kernel is simply absent from the private library, so asking for it
returns nil and the host falls back. That is the intended failure: a missing
function is loud, whereas a weakened memory fence is not.

Usage: make_private_clone_source.py <out.metal> [--manifest out.json]
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

XCODE14_CANDIDATES = [
    "/Users/rschu/p/xcode-14.2-extract/expanded/Xcode.app",
    "/Applications/Xcode_14.2.app",
]
MAX_ROUNDS = 40


def find_metal_bin():
    import os
    if os.environ.get("XCODE14_APP"):
        XCODE14_CANDIDATES.insert(0, os.environ["XCODE14_APP"])
    for app in XCODE14_CANDIDATES:
        p = (Path(app) / "Contents/Developer/Toolchains/XcodeDefault.xctoolchain"
             / "usr/metal/macos/bin/metal")
        if p.is_file():
            return str(p)
    return None


def assemble_corpus():
    """Exactly ds4_gpu_full_source(): base string, then required_sources in order."""
    m = Path("ds4_metal.m").read_text()
    b0 = m.index("static const char *ds4_gpu_source =")
    b1 = m.index("static NSString *ds4_gpu_full_source(void)")
    lits = re.findall(r'^"((?:[^"\\]|\\.)*)"', m[b0:b1], re.M)
    base = "".join(lits).encode().decode("unicode_escape")
    order = re.findall(r'@\[@"DS4_METAL_[A-Z0-9_]+",\s*@"([^"]+)"\]', m)
    return "\n".join([base] + [Path(p).read_text() for p in order])


def enclosing_block(lines, idx):
    """Expand a 0-based line index to its enclosing TOP-LEVEL definition.

    Top level is identified by column-zero `{` ... `}` bracketing, which is the
    house style throughout the corpus. Preprocessor blocks (`#if` at column
    zero) are treated the same way so a whole conditional region goes together.
    Returns (start, end) inclusive, or None if it cannot be bounded -- in which
    case the caller must not guess.
    """
    # A column-zero #if...#endif region wins if we are inside one.
    depth = 0
    start_if = None
    for i in range(idx, -1, -1):
        ln = lines[i]
        if ln.startswith("#endif"):
            depth += 1
        elif ln.startswith("#if"):
            if depth == 0:
                start_if = i
                break
            depth -= 1
    if start_if is not None:
        depth = 0
        for j in range(start_if, len(lines)):
            if lines[j].startswith("#if"):
                depth += 1
            elif lines[j].startswith("#endif"):
                depth -= 1
                if depth == 0:
                    return start_if, j
    # Otherwise the enclosing column-zero brace block.
    start = None
    for i in range(idx, -1, -1):
        if lines[i].startswith("}"):
            break
        if re.match(r"^[A-Za-z_#\[]", lines[i]) and not lines[i].startswith("//"):
            start = i
            if lines[i].rstrip().endswith("{") or "(" in lines[i]:
                # keep scanning up through a multi-line signature
                continue
    if start is None:
        return None
    for j in range(max(start, idx), len(lines)):
        if lines[j].startswith("}"):
            return start, j
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out_path = Path(sys.argv[1])
    manifest_path = None
    if "--manifest" in sys.argv:
        manifest_path = Path(sys.argv[sys.argv.index("--manifest") + 1])

    metal = find_metal_bin()
    if not metal:
        print("VOID: Xcode 14.2 metal frontend not found; set XCODE14_APP",
              file=sys.stderr)
        return 2

    src = assemble_corpus()
    lines = src.split("\n")
    elided = []

    with tempfile.TemporaryDirectory() as td:
        tmp_src = Path(td) / "clone.metal"
        tmp_air = Path(td) / "clone.air"
        for rnd in range(MAX_ROUNDS):
            tmp_src.write_text("\n".join(lines))
            r = subprocess.run(
                [metal, "-std=metal3.0", "-mmacosx-version-min=13.0",
                 "-ferror-limit=400", "-DDS4_PRIVATE_CLONE=1",
                 "-c", str(tmp_src), "-o", str(tmp_air)],
                capture_output=True, text=True)
            if r.returncode == 0:
                print(f"clone source compiles under Xcode 14.2 after {rnd} "
                      f"elision round(s); {len(elided)} definition(s) removed")
                break
            bad = sorted({int(m.group(1)) for m in
                          re.finditer(r"^[^\s:]+:(\d+):\d+: error:", r.stderr, re.M)})
            if not bad:
                print("VOID: compile failed with no attributable line:\n" +
                      r.stderr[:2000], file=sys.stderr)
                return 1
            spans = []
            for L in bad:
                blk = enclosing_block(lines, L - 1)
                if blk is None:
                    print(f"VOID: cannot bound the definition at line {L}; "
                          f"refusing to guess.\n  {lines[L-1][:100]}",
                          file=sys.stderr)
                    return 1
                spans.append(blk)
            # merge and blank out, keeping line numbers stable within a round
            merged = []
            for a, b in sorted(spans):
                if merged and a <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], b))
                else:
                    merged.append((a, b))
            for a, b in merged:
                head = next((lines[i] for i in range(a, b + 1)
                             if lines[i].strip()), "")
                body = "\n".join(lines[a:b + 1])
                # REFUSE to elide the async primitive itself.
                #
                # Eliding a definition the old frontend cannot compile is the
                # point of this pass -- for device atomics and the like, whose
                # absence is inert. It is the opposite for anything containing
                # simdgroup_async_copy: removing that silently turns an ASYNC
                # arm into its own manual control, and the run then reports a
                # null for the primitive while never having used it. That
                # happened once, to the routed-MoE staging arms, and it is
                # exactly the class of failure this campaign keeps paying for.
                #
                # A genuine compile error in async code is a bug to fix in the
                # source, not to route around here.
                if "simdgroup_async_copy" in body or "simdgroup_future" in body:
                    print("VOID: the elision pass tried to remove code "
                          "containing the async primitive:", file=sys.stderr)
                    print(f"      lines {a + 1}-{b + 1}: {head[:120]}",
                          file=sys.stderr)
                    print("      Eliding this would silently downgrade an async "
                          "arm to its manual control.", file=sys.stderr)
                    print("      Fix the source; do not route around it.",
                          file=sys.stderr)
                    return 1
                elided.append({"first_line": a + 1, "last_line": b + 1,
                               "head": head[:120]})
                for i in range(a, b + 1):
                    lines[i] = ""
        else:
            print(f"VOID: still failing after {MAX_ROUNDS} rounds", file=sys.stderr)
            return 1

    out_path.write_text("\n".join(lines))
    # Both plain definitions and templated [[host_name]] entry points count --
    # the async arms are the latter, and a manifest that missed them reported
    # "0 sgasync kernels" while the metallib was simply built without the
    # define. Emitting the artifact here rather than leaving it to the caller is
    # the actual fix; this is belt and braces.
    body = "\n".join(lines)
    kernels = set(re.findall(r"^kernel void (\w+)", body, re.M))
    kernels |= set(re.findall(r'host_name\("([^"]+)"\)', body))
    if manifest_path:
        manifest_path.write_text(json.dumps(
            {"elided": elided, "kernels": sorted(kernels)}, indent=1))
    # Build the metallib here, with the SAME define the validation compile used.
    # Leaving this to the caller is how the first attempt produced an artifact
    # with no async kernels in it: the source compiled fine without
    # -DDS4_PRIVATE_CLONE, it just had the async arms preprocessed away.
    lib_path = out_path.with_suffix(".metallib")
    air = out_path.with_suffix(".air")
    metallib_bin = str(Path(metal).with_name("metallib"))
    for cmd in ([metal, "-std=metal3.0", "-mmacosx-version-min=13.0",
                 "-DDS4_PRIVATE_CLONE=1", "-c", str(out_path), "-o", str(air)],
                [metallib_bin, str(air), "-o", str(lib_path)]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("VOID: " + " ".join(cmd[:1]) + " failed:\n" + r.stderr[:2000],
                  file=sys.stderr)
            return 1
    for e in elided:
        print(f"  elided lines {e['first_line']}-{e['last_line']}: {e['head']}")
    n_async = len([k for k in kernels if "sgasync" in k])
    print(f"{len(kernels)} entry points survive ({n_async} sgasync) -> {lib_path}")
    if n_async == 0:
        print("WARNING: no sgasync entry points. The async arms are absent and "
              "any A/B against them is a null, not a negative.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
