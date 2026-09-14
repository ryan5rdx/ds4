#!/usr/bin/env python3
"""Assemble the runtime Metal corpus exactly as ds4_gpu_full_source() does.

The shaders are concatenated at runtime from ds4_metal.m's base string plus the
required_sources list, in order -- so a probe that wants to compile against the
real corpus has to reproduce that assembly rather than compiling one .metal in
isolation. Reads the order out of ds4_metal.m so it cannot drift.
"""
import re, sys
# Reproduce ds4_gpu_full_source(): base source + the required_sources list, in order.
m = open('ds4_metal.m').read()
base_start = m.index('static const char *ds4_gpu_source =')
base_end   = m.index('static NSString *ds4_gpu_full_source(void)')
lits = re.findall(r'^"((?:[^"\\]|\\.)*)"', m[base_start:base_end], re.M)
base = ''.join(lits).encode().decode('unicode_escape')
order = re.findall(r'@\[@"DS4_METAL_[A-Z0-9_]+",\s*@"([^"]+)"\]', m)
out = [base]
for path in order:
    out.append(open(path).read())
out_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/full.metal'
open(out_path,'w').write('\n'.join(out))
print("sources:", len(order), "bytes:", sum(len(x) for x in out))
