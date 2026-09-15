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
# Two tiers, because "zero readers" means two different things.
#
# REQUIRED: the feature ships on this branch, so exactly one reader. Zero is a
# knob renamed or deleted out from under its feature.
# OPTIONAL: the feature is not on every branch in the stack -- MoE arm B is
# unbanked, so DS4_MOE_RAW_STAGE is legitimately absent on the v4 bank. Zero is
# reported and allowed; TWO is still a build failure, because the multi-reader
# case is the one that actually burned us and it does not care which branch it
# happens on.
REQUIRED="DS4_TP_COMPACT_TOP1 DS4_SGASYNC_ARM"
OPTIONAL="DS4_MOE_RAW_STAGE"
for knob in $REQUIRED $OPTIONAL; do
    case " $OPTIONAL " in *" $knob "*) opt=1 ;; *) opt=0 ;; esac
    # `|| true` on every count: grep exits 1 on no match, and under pipefail
    # that aborted the loop silently -- so a knob RENAMED out of existence
    # reported nothing at all, which is the same blind spot in miniature.
    hits=$( { grep -c "getenv(\"$knob\")" ./*.c ./*.m 2>/dev/null || true; } | grep -v ':0$' || true)
    total=$( { grep -o "getenv(\"$knob\")" ./*.c ./*.m 2>/dev/null || true; } | wc -l | tr -d ' ')
    if [ "$total" -eq 0 ] && [ "$opt" -eq 1 ]; then
        echo "ok   $knob: absent on this branch (optional; its feature is not banked here)"
    elif [ "$total" -ne 1 ]; then
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

[ "$fail" -eq 0 ] && echo "PASS: knob readers and compact-logit invariants hold"
exit $fail
