#!/usr/bin/env python3
"""Threadgroup-memory census: does every kernel get the scratch it indexes?

WHY THIS EXISTS.  Metal does not bounds-check threadgroup memory.  A kernel that
writes past its `setThreadgroupMemoryLength:` allocation aliases silently -- no
fault, no validation error at build time, just wrong numbers or corrupted
neighbours.  Upstream's 8fcd61d found exactly that: two dispatch sites gave
`kernel_mul_mm_mpp_direct_rhs` 4096 bytes while its second A tile starts at
offset 4096, so `tA1` wrote 4096 bytes past the end.  It had been that way
across a whole measurement campaign, and fixing it cost ~285 ms per prefill
chunk -- which means the numbers taken before it were both wrong AND fast.

Nobody had checked the other sites.  This does.

WHAT IT CHECKS, precisely.  For each compute kernel in metal/*.metal it finds
every constant offset the kernel takes into its threadgroup pointer -- the
`shmem + N` / `scratch + N` idiom -- and takes the largest.  For each
`setThreadgroupMemoryLength:` call site in ds4_metal.m it resolves the byte
count where that is a constant expression, and attributes it to the kernel named
by the nearest preceding pipeline lookup.  A kernel whose largest offset is
>= the smallest length bound at any of its sites is REPORTED: at that offset the
kernel is writing at or past the end.

WHAT IT DOES NOT CHECK, and why that is stated rather than papered over:
  - Offsets computed from template parameters or function constants (NR0*NK,
    NSG*D) are invisible here.  Those are the harder half and need the dynamic
    pass below.
  - A kernel whose largest offset is comfortably inside the bound can still
    write past it from a runtime index.
  - Call sites whose length is a runtime expression are listed as UNRESOLVED
    rather than assumed safe.

So a clean run is not a proof.  It is the cheap half, and it is the half that
would have caught 8fcd61d.

THE DYNAMIC HALF, which this does not do: run the dispatching probes under
`MTL_SHADER_VALIDATION=1`, which does bounds-check threadgroup access.  That
covers the computed offsets this cannot see, at the cost of needing the kernel
to actually execute.  See the census entry in the rig plan.

Usage: python3 tools/tgmem_census.py [--verbose]   (or: make check-threadgroup-memory)
Exit 1 if any kernel is reported.
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERBOSE = "--verbose" in sys.argv

TGPARAM_RE = re.compile(r"threadgroup\s+[A-Za-z_]\w*\s*\*\s*(\w+)\s*(?:\[\[threadgroup\(\d+\)\]\])?")
KERNEL_RE = re.compile(r"^\s*(?:\[\[host_name\(\"([^\"]+)\"\)\]\]\s*)?kernel\s+void\s+(\w+)\s*\(",
                       re.M)
TEMPLATE_HOST_RE = re.compile(r'template\s*\[\[host_name\("([^"]+)"\)\]\]\s*kernel\s+(\w+)\s+(\w+)<')

# `atIndex:` is on the NEXT line at most v4-era sites, and call_sites()
# matches line by line, so an `atIndex|\n` terminator matched neither and
# the census found 1 site of 160.  End-of-string terminates a split line.
LEN_RE = re.compile(r"setThreadgroupMemoryLength:\s*([^\n]*?)\s*(?:atIndex|$)")
PIPE_RE = re.compile(r'"(kernel_[A-Za-z0-9_]+)"')

# `threadgroup T * name = (threadgroup T *)shmem;`  or `... *)(shmem + 4096);`
ALIAS_RE = re.compile(
    r"threadgroup\s+([A-Za-z_]\w*)\s*\*\s*(\w+)\s*=\s*\(\s*threadgroup\s+[A-Za-z_]\w*\s*\*\s*\)"
    r"\s*\(?\s*(\w+)\s*(?:\+\s*([^)\;]+?))?\s*\)?\s*;")
# `auto tX = tensor(sa + NR0*NK, ...)` and any `base + expr` in an index position
USE_RE = re.compile(r"\b(\w+)\s*\+\s*([A-Za-z_]\w*(?:\s*\*\s*[A-Za-z_0-9]\w*)*)\b")
CONST_RE = re.compile(r"constexpr\s+(?:short|int|uint)\s+(\w+)\s*=\s*([^;]+);")

# Byte size of the element types that appear as threadgroup tile types.  A
# template parameter (SA, S0, S1) has no size here, and that is the point: the
# report says so and prices the offset at 2 and 4 bytes rather than guessing.
ELEM_BYTES = {"char": 1, "uchar": 1, "half": 2, "ushort": 2, "short": 2,
              "float": 4, "uint": 4, "int": 4, "half2": 4, "half4": 8,
              "float2": 8, "float4": 16, "uint4": 16, "half4x4": 32,
              "float4x4": 64}


ACCESSOR_RE = re.compile(
    r"static\s+NSUInteger\s+(\w+)\(void\)\s*\{(.*?)\n\}", re.S)


def accessor_values():
    """`static NSUInteger f(void)` -> its constant value, where it has one.

    Routing five call sites through ds4_gpu_mm_nax_tg_mem() was the right change
    for the engine and it made every one of them invisible to a byte-count
    check -- the expression stopped being a literal.  This resolves the common
    shape back: a helper whose body names one constant
    (`const NSUInteger correct = ...`) or returns one directly.  A helper whose
    value genuinely depends on runtime state stays UNRESOLVED, which is correct.
    """
    src = open(os.path.join(ROOT, "ds4_metal.m")).read()
    out = {}
    for m in ACCESSOR_RE.finditer(src):
        name, body = m.group(1), m.group(2)
        cand = re.findall(r"const\s+NSUInteger\s+\w+\s*=\s*([^;]+);", body)
        cand += re.findall(r"return\s+([^;]+);", body)
        for c in cand:
            v = _const_eval_raw(c)
            if v is not None:
                out[name] = v
                break
    return out


SIZEOF_BYTES = {
    "char": 1, "uint8_t": 1, "int8_t": 1,
    "half": 2, "short": 2, "ushort": 2, "uint16_t": 2, "int16_t": 2,
    "float": 4, "int": 4, "uint": 4, "uint32_t": 4, "int32_t": 4,
    "half2": 4, "float2": 8, "half4": 8, "uint2": 8,
    "uint64_t": 8, "int64_t": 8, "double": 8,
    "float4": 16, "uint4": 16, "half4x4": 32, "float4x4": 64,
}

TG16_OPEN_RE = re.compile(r"\bDS4_TG16\s*\(")


def _tg16_inner(expr):
    """Text inside a `DS4_TG16(...)`, matched with balanced parentheses.

    ds4_metal.m:66 defines DS4_TG16 as the central 16-byte round-up, and 152 of
    the 153 binding sites go through it.  A regex that stops at the first `)`
    cannot read `DS4_TG16(656u * sizeof(float))` or
    `DS4_TG16(2048u * sizeof(uint64_t) + 96u)`, so before this the census
    resolved a bound for ZERO kernels and still exited 0 -- the guard against the
    8fcd61d bug class was checking nothing.  Returns None when there is no
    DS4_TG16 or its parentheses do not close.
    """
    m = TG16_OPEN_RE.search(expr)
    if not m:
        return None
    depth, start, i = 1, m.end(), m.end()
    while i < len(expr):
        if expr[i] == "(":
            depth += 1
        elif expr[i] == ")":
            depth -= 1
            if depth == 0:
                return expr[start:i]
        i += 1
    return None


def _const_eval_raw(expr):
    """Evaluate a byte-count expression if it is made only of literals."""
    e = expr.strip().rstrip("]").strip()
    for tname, tbytes in SIZEOF_BYTES.items():
        e = re.sub(r"\bsizeof\(%s\)" % re.escape(tname), str(tbytes), e)
    e = re.sub(r"(\d+)[uU]\b", r"\1", e)
    e = e.replace("(NSUInteger)", "")
    if not re.fullmatch(r"[\d\s\+\*\(\)/-]+", e):
        return None
    try:
        return int(eval(e, {"__builtins__": {}}))
    except Exception:
        return None


_ACCESSORS = None


def const_eval(expr):
    """Constant byte count, resolving a known accessor call if that is all it is."""
    global _ACCESSORS
    inner = _tg16_inner(expr)
    if inner is not None:
        v = const_eval(inner)
        # Re-apply the macro's round-up.  It can only raise the bound, so a
        # census that uses the rounded value stays conservative.
        return None if v is None else ((v + 15) & ~15)
    v = _const_eval_raw(expr)
    if v is not None:
        return v
    if _ACCESSORS is None:
        _ACCESSORS = accessor_values()
    e = expr.strip().rstrip("]").strip()
    m = re.fullmatch(r"(\w+)\(\)", e)
    if m and m.group(1) in _ACCESSORS:
        return _ACCESSORS[m.group(1)]
    return None


def kernel_bodies():
    """kernel name -> source text of its body."""
    out = {}
    alias = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "metal", "*.metal"))):
        src = open(path).read()
        for m in TEMPLATE_HOST_RE.finditer(src):
            alias.setdefault(m.group(1), m.group(3))
        starts = [(m.start(), m.group(1) or m.group(2)) for m in KERNEL_RE.finditer(src)]
        for i, (pos, name) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(src)
            out[name] = src[pos:end]
    for host, tmpl in alias.items():
        if host in out:
            continue
        for path in sorted(glob.glob(os.path.join(ROOT, "metal", "*.metal"))):
            src = open(path).read()
            m = re.search(r"\n(?:static\s+)?(?:inline\s+)?\w[\w\s\*<>,]*\b" +
                          re.escape(tmpl) + r"\s*\(", src)
            if m:
                nxt = re.search(r"\n(?:kernel|template)\s", src[m.start() + 1:])
                out[host] = src[m.start(): m.start() + 1 + (nxt.start() if nxt else len(src))]
                break
    return out


def body_consts(body):
    """constexpr name -> value, for the simple literal and product forms."""
    vals = {}
    for _ in range(3):                     # resolve chains like NL0 = NK/16
        for m in CONST_RE.finditer(body):
            name, expr = m.group(1), m.group(2)
            e = expr
            for k, v in vals.items():
                e = re.sub(r"\b%s\b" % re.escape(k), str(v), e)
            v = const_eval(e)
            if v is not None:
                vals[name] = v
    return vals


def tg_extent(body):
    """Largest byte offset the body can be shown to take into threadgroup memory.

    Returns (bytes, evidence, exact) where exact is False when the element size
    of the pointer type is unknown -- a template parameter -- in which case the
    figure is priced at 2 bytes per element and the caller must treat it as a
    lower bound.
    """
    tg = set(TGPARAM_RE.findall(body))
    if not tg:
        return 0, None, True
    consts = body_consts(body)
    # alias -> (byte base, element type)
    alias = {}
    for name in tg:
        alias[name] = (0, "char")
    for m in ALIAS_RE.finditer(body):
        etype, name, base, off = m.group(1), m.group(2), m.group(3), m.group(4)
        if base not in alias:
            continue
        base_bytes, base_type = alias[base]
        add = 0
        if off:
            e = off
            for k, v in consts.items():
                e = re.sub(r"\b%s\b" % re.escape(k), str(v), e)
            n = const_eval(e)
            if n is None:
                continue
            add = n * ELEM_BYTES.get(base_type, 1)
        alias[name] = (base_bytes + add, etype)

    best, ev, exact = 0, None, True
    for name, (base_bytes, etype) in alias.items():
        if base_bytes > best:
            best, ev, exact = base_bytes, "%s at byte %d" % (name, base_bytes), True
        for m in USE_RE.finditer(body):
            if m.group(1) != name:
                continue
            e = m.group(2)
            for k, v in consts.items():
                e = re.sub(r"\b%s\b" % re.escape(k), str(v), e)
            n = const_eval(e)
            if n is None:
                continue
            sz = ELEM_BYTES.get(etype)
            this_exact = sz is not None
            # The offset is where a region STARTS; the bound has to cover the
            # region too.  For the double-buffer idiom -- `tA1 = tensor(sa +
            # NR0*NK, ...)` -- the second tile is the same size as the offset,
            # so the extent is 2x it.  Assuming that is the safe direction for a
            # report: it over-requires for a small trailing scalar and is exactly
            # right for the pattern 8fcd61d was about.
            elem = sz if sz else 2
            end = base_bytes + 2 * n * elem
            if end > best:
                best = end
                exact = this_exact
                ev = "%s + %s = %d element(s) of %s, x2 for the region itself%s" % (
                    name, m.group(2), n, etype,
                    "" if this_exact else " (element size unknown, priced at 2 B)")
    return best, ev, exact


def call_sites():
    """kernel name -> list of (line, resolved bytes or None, raw expr)."""
    src = open(os.path.join(ROOT, "ds4_metal.m")).read().split("\n")
    sites = {}
    for i, line in enumerate(src):
        m = LEN_RE.search(line)
        if not m:
            continue
        val = const_eval(m.group(1))
        # nearest preceding kernel name mentioned in this encoder's setup
        name = None
        for j in range(i, max(-1, i - 40), -1):
            p = PIPE_RE.findall(src[j])
            if p:
                name = p[-1]
                break
        sites.setdefault(name, []).append((i + 1, val, m.group(1).strip()))
    return sites


def main():
    bodies = kernel_bodies()
    sites = call_sites()
    reported, unresolved, checked = [], [], 0

    for name, entries in sorted(sites.items(), key=lambda kv: kv[0] or ""):
        if name is None:
            for ln, val, raw in entries:
                unresolved.append((ln, "no kernel name within 40 lines", raw))
            continue
        body = bodies.get(name)
        if body is None:
            for ln, val, raw in entries:
                unresolved.append((ln, "kernel %s not found in metal/" % name, raw))
            continue
        need, where, exact = tg_extent(body)
        vals = [v for _, v, _ in entries if v is not None]
        for ln, val, raw in entries:
            if val is None:
                unresolved.append((ln, "%s: length is not a constant" % name, raw))
        if not vals:
            continue
        checked += 1
        low = min(vals)
        if need and low < need:
            reported.append((name, low, need, where, exact,
                             [ln for ln, v, _ in entries if v == low]))
        elif VERBOSE:
            print("  ok       %-52s bound %-6d largest constant offset %d" % (name, low, need))

    if unresolved and VERBOSE:
        print("\nUNRESOLVED (not assumed safe, just not decidable here):")
        for ln, why, raw in unresolved:
            print("  ds4_metal.m:%-6d %-46s  %s" % (ln, why, raw))

    print("\nthreadgroup-memory census: %d kernel(s) with a constant bound checked, "
          "%d site(s) unresolved" % (checked, len(unresolved)))

    # A census that resolves nothing is indistinguishable from a clean one unless
    # it says so.  That is exactly what happened when DS4_TG16 was introduced:
    # 0 checked / 133 unresolved, exit 0, for an unknown number of releases.  Any
    # future wrapper that blinds the parser must break the build, not pass it.
    if checked == 0:
        print("\nFATAL: the census resolved a constant bound for ZERO kernels.")
        print("It cannot have checked anything, so its exit status is meaningless.")
        print("Something now wraps or computes the length expressions -- teach")
        print("const_eval() to see through it (as _tg16_inner does for DS4_TG16)")
        print("or the 8fcd61d bug class is unguarded. Re-run with --verbose to")
        print("see what the unresolved sites look like.")
        return 1

    if reported:
        print("\n%d KERNEL(S) BOUND BELOW WHAT THEY INDEX:" % len(reported))
        for name, low, need, where, exact, lines in reported:
            print("  %s" % name)
            print("      smallest bound %d bytes at ds4_metal.m:%s" %
                  (low, ",".join(str(l) for l in lines)))
            print("      needs %d bytes: %s" % (need, where))
            if not exact:
                print("      element size is a template parameter: this is a LOWER"
                      " bound, the real extent may be 2x or 4x it")
            print("      -> at that offset the kernel writes at or past the end")
        print("\nThis is the 8fcd61d bug class.  Metal does not bounds-check")
        print("threadgroup memory, so the symptom is wrong numbers, not a fault.")
        return 1
    print("No kernel is bound below what it can be shown to index.")
    print("Not a proof: computed offsets (NR0*NK, NSG*D) are invisible to a")
    print("static pass -- run the dispatching probes under MTL_SHADER_VALIDATION=1")
    print("for those.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
