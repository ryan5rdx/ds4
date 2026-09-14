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
[ "$fail" -eq 0 ] && echo "PASS: every single-reader knob has exactly one reader"
exit $fail
