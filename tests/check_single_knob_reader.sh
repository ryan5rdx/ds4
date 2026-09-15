#!/usr/bin/env bash
# Knobs that MUST have exactly one reader.
#
# U64TOP1-TP broke three times in a row the same way: a decision implemented in
# one caller while the measurement ran through another. The arming call, the
# eligibility predicate and the kill switch each lived in ds4_server.c while the
# A/B ran ds4-bench. A second getenv() for one of these is that bug reappearing,
# so it is a build failure rather than something to notice in a rig report.
set -euo pipefail
cd "$(dirname "$0")/.."
fail=0
for knob in DS4_TP_COMPACT_TOP1 DS4_MOE_RAW_STAGE DS4_SGASYNC_ARM; do
    # `|| true` on every count: grep exits 1 on no match, and under pipefail
    # that aborted the loop silently -- so a knob RENAMED out of existence
    # reported nothing at all, which is the same blind spot in miniature.
    hits=$( { grep -c "getenv(\"$knob\")" ./*.c ./*.m 2>/dev/null || true; } | grep -v ':0$' || true)
    total=$( { grep -o "getenv(\"$knob\")" ./*.c ./*.m 2>/dev/null || true; } | wc -l | tr -d ' ')
    if [ "$total" -ne 1 ]; then
        echo "FAIL $knob has $total reader(s), want exactly 1:"
        if [ -n "$hits" ]; then echo "$hits" | sed 's/^/    /'; else echo "    (none -- renamed or deleted?)"; fi
        fail=1
    else
        echo "ok   $knob: 1 reader ($(echo "$hits" | cut -d: -f1))"
    fi
done
# --- compact-logit invariants --------------------------------------------
#
# Two things that were wrong and would be invisible if they came back.

# 1. ONE PLACE APPLIES THE SAMPLER FLOOR. ds4_session_sample() returned the raw
#    key index while ds4_session_argmax() applied the floor, so the two
#    disagreed on exactly the sub-floor vectors the floor exists for.
#    ds4_session_argmax_excluding() deliberately uses the UNFLOORED key -- its
#    reference seeds from a real element and has no floor -- so the invariant is
#    about the floor comparison, not about decoding the key.
n=$( { grep -o "compact_top1_key >> 32" ds4.c || true; } | wc -l | tr -d ' ')
if [ "$n" -ne 1 ]; then
    echo "FAIL the compact sampler floor is applied in $n places (want 1: ds4_session_compact_argmax)"
    grep -n "compact_top1_key >> 32" ds4.c | sed 's/^/    /'
    fail=1
else
    echo "ok   compact sampler floor applied in exactly 1 place"
fi

# 2. The rollback snapshot must carry EVERY compact field. It stored top-1 and
#    the length and silently dropped the runner-up and both NaN flags, which
#    would restore stale exclusion metadata beside a correctly restored top-1.
missing=""
for f in compact_top1_key compact_top2_key compact_nan_at_0 compact_nan_at_1 compact_top1_len logits_compact; do
    grep -Eq "s->${f} += +s->glm53_rollback_" ds4.c || missing="$missing $f"
    grep -Eq "s->glm53_rollback_[a-z0-9_]+ += +s->${f};" ds4.c || missing="$missing ${f}(capture)"
done
if [ -n "$missing" ]; then
    echo "FAIL rollback does not round-trip:$missing"
    fail=1
else
    echo "ok   rollback captures and restores every compact field"
fi

# 3. EVERY prefill driver emits the per-chunk census.
#
# There are two: metal_graph_prefill_chunked_range (layer-major) and
# glm_graph_prefill_range (compact-indexed). All three validity emitters were
# wired only to the first, and the rig runs the second -- so ANEPROC 8a reported
# chunk_lines=0 twice and PERFONLY3's +3.19% arrived with no counter line, from
# call sites that exist, are correct, and are unreachable. A compiler cannot see
# this; only a check that names both drivers can.
python3 - <<'PY' || fail=1
import re, sys
src = open("ds4.c").read().split("\n")
drivers = ["metal_graph_prefill_chunked_range", "glm_graph_prefill_range"]
bad = []
for d in drivers:
    # find the definition (a line starting the function, with a brace body)
    start = next((i for i, l in enumerate(src)
                  if re.match(r"^static .*\b" + d + r"\(", l)), None)
    if start is None:
        bad.append(f"{d}: definition not found (renamed?)")
        continue
    depth, end, seen = 0, None, False
    for i in range(start, len(src)):
        depth += src[i].count("{") - src[i].count("}")
        if src[i].count("{"):
            seen = True
        if seen and depth <= 0:
            end = i
            break
    body = "\n".join(src[start:(end or len(src)) + 1])
    if "ds4_prefill_chunk_census" not in body:
        bad.append(f"{d}: no ds4_prefill_chunk_census call")
if bad:
    print("FAIL prefill census is not emitted by every driver:")
    for b in bad:
        print("    " + b)
    print("    A validity gate that is unreachable on the path the rig runs")
    print("    reports nothing and reads as success.")
    sys.exit(1)
print("ok   both prefill drivers emit the per-chunk census")
PY

[ "$fail" -eq 0 ] && echo "PASS: knob readers and compact-logit invariants hold"
exit $fail
