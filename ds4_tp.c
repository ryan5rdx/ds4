/* Tensor-parallel transport and lockstep protocol.  See ds4_tp.h and
 * misc/METAL_TENSOR_PARALLELISM.md for the design.
 *
 * Wire notes: both ranks are identical Apple Silicon machines by
 * definition, so the wire format is host little-endian; the hello magic
 * doubles as a byte-order check.  The control socket is a plain blocking
 * TCP stream carrying framed commands.  Gate traffic goes over RDMA
 * (Thunderbolt UC queue pair, two-sided send/recv — see the driver quirks
 * note at ds4_tp_rdma) or over a dedicated full-duplex TCP socket at 16KB
 * per direction as the fallback. */

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <poll.h>
#include <stdarg.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "ds4_tp.h"
#include "ds4_gpu.h"

#if defined(__APPLE__) && defined(__has_include)
#if __has_include(<infiniband/verbs.h>)
#include <infiniband/verbs.h>
#include <dlfcn.h>
#define DS4_TP_HAVE_VERBS 1
#endif
#endif

#define DS4_TP_MAGIC UINT32_C(0x44533454) /* "DS4T" */
#define DS4_TP_BATCH_MAGIC UINT32_C(0x44533442) /* "DS4B" */
/* Base 7; upstream 9 adds the multimodal sync frame, and this fork adds the
 * prefill cancel frame and the verify flags word on top of it.
 *
 * 11 adds the GLM-5.3 rollback protocol: a mode word on REWIND (reusing its
 * `reserved` field, so the frame size is unchanged), an acknowledged REWIND,
 * and the ROLLBACK_CAPTURE frame.  The version bump is the only thing that can
 * refuse a mixed pair here: the frame layouts did not change, and `split_flags`
 * bit 11 only says whether the feature is enabled, not which ACK semantics the
 * peer speaks.  Three revisions shipped with different semantics under version
 * 10 and all would have passed bring-up, then disagreed at the first rewind.
 *
 * 12 is the upstream merge.  Upstream independently reached 10 and gave frames
 * 19/20/21 to RDMA_WARM / RDMA_POSTED / GLM_MTP, which this fork had already
 * spent on CANCEL / ROLLBACK_CAPTURE (renumbered to 22/23 in ds4_tp.h).  Both
 * 10 and 11 therefore name two incompatible wire protocols, so the merged
 * binary must refuse BOTH -- hence 12 rather than 11-and-keep-going. */
#define DS4_TP_PROTOCOL_VERSION 12u

#define DS4_TP_DEFAULT_TIMEOUT_SEC 300
/* Once both ranks enter a Metal gate, a live exchange normally completes in
 * microseconds. Fail well before Metal's command-buffer watchdog if the peer
 * stalls while keeping its sockets open. */
#define DS4_TP_DEFAULT_GATE_TIMEOUT_MS 750
/* Floor when an ANE sidecar is configured: it adds ~14-17 ms per layer of skew
 * for the barrier to absorb, and over 42 layers that is well past 750 ms. */
#define DS4_TP_ANE_GATE_TIMEOUT_MS 3000
/* "Once both ranks enter" is the whole caveat, and it does not cover how they
 * get there.  A batch or big gate opens with a header pair on data_fd, and that
 * pair IS the barrier that puts both ranks inside the gate: until it completes
 * the peer may still be an entire layer-0 encode behind.  On the first gate of a
 * session that is first-use Metal pipeline compilation plus cold expert pages --
 * ds4_session_gpu_warmup() prices it at ~1.1 s and only the ranks that call it
 * are spared -- so budgeting the arrival barrier at the in-gate 750 ms fails the
 * pair on its very first request, silently, before any token is produced.
 *
 * Still bounded, because a gate holds a committed command buffer open on both
 * sides.  Ten seconds sits above every arrival skew measured here and an order
 * of magnitude below nothing: single prefill command buffers of 7-11 s are
 * measured to survive the watchdog (see ds4_prefill_watchdog_chunk in ds4.c), so
 * a genuinely wedged peer is still reported by us rather than by the watchdog. */
#define DS4_TP_DEFAULT_MEET_TIMEOUT_MS 10000

typedef struct {
    uint32_t magic;
    uint32_t type;
    uint32_t bytes;
} ds4_tp_frame_header;

typedef struct {
    uint32_t magic;      /* also detects byte-order mismatch */
    uint32_t version;
    uint32_t role;
    uint32_t rdma_ok;    /* this side has a usable verbs device */
    uint64_t gguf_bytes;
    uint32_t model_id;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t quant_bits;
    uint32_t ctx_size;
    uint32_t gate_slot_start;
    uint32_t gate_slot_step;
    uint32_t gates_per_token;
    uint32_t split_flags;   /* was pad; same wire size */
    uint64_t gate_slot_mask[DS4_TP_GATE_MASK_WORDS];
} ds4_tp_hello_fixed;

typedef struct {
    uint64_t slab_base;
    uint32_t rkey;
    uint32_t qpn;
    uint32_t psn;
    uint32_t mtu;
    uint16_t lid;
    uint8_t gid[16];
    uint8_t link_layer;
} ds4_tp_rdma_info;

/* TCP gate frames carry a small header so a desynchronized pair fails loudly
 * instead of silently mixing partials. */
typedef struct {
    uint32_t magic;
    uint16_t layer;
    uint16_t gate;
    uint64_t seq;
} ds4_tp_gate_header;

#ifdef DS4_TP_HAVE_VERBS
/* librdma is loaded at runtime so builds and machines without the RDMA
 * stack (or with it disabled) fall back to TCP with no link-time cost.
 * ibv_post_send()/ibv_poll_cq() are header inlines over context->ops, so
 * only the setup entry points need dlsym. */
typedef struct {
    void *handle;
    struct ibv_device **(*get_device_list)(int *);
    void (*free_device_list)(struct ibv_device **);
    const char *(*get_device_name)(struct ibv_device *);
    struct ibv_context *(*open_device)(struct ibv_device *);
    int (*close_device)(struct ibv_context *);
    int (*query_device)(struct ibv_context *, struct ibv_device_attr *);
    int (*query_port)(struct ibv_context *, uint8_t, struct ibv_port_attr *);
    int (*query_gid)(struct ibv_context *, uint8_t, int, union ibv_gid *);
    struct ibv_pd *(*alloc_pd)(struct ibv_context *);
    int (*dealloc_pd)(struct ibv_pd *);
    struct ibv_mr *(*reg_mr)(struct ibv_pd *, void *, size_t, int);
    int (*dereg_mr)(struct ibv_mr *);
    struct ibv_cq *(*create_cq)(struct ibv_context *, int, void *, struct ibv_comp_channel *, int);
    int (*destroy_cq)(struct ibv_cq *);
    struct ibv_qp *(*create_qp)(struct ibv_pd *, struct ibv_qp_init_attr *);
    int (*destroy_qp)(struct ibv_qp *);
    int (*modify_qp)(struct ibv_qp *, struct ibv_qp_attr *, int);
    int (*query_qp)(struct ibv_qp *, struct ibv_qp_attr *, int, struct ibv_qp_init_attr *);
} ds4_tp_verbs_api;

/* AppleThunderboltRDMA quirks (validated with scratchpad probes,
 * ): only UC queue pairs exist (RC/UD: ENOTSUP); RDMA WRITE work
 * requests are accepted but never execute, so the data plane is two-sided
 * SEND/RECV like Apple's own JACCL; messages above 16KB are not delivered;
 * RTR requires GRH addressing with the IPv4-mapped GID that appears only
 * once the Thunderbolt member interface has an IPv4 address of its own.
 * UC delivery is in-order and the gate sequence is globally deterministic
 * (a model-fixed number of gates per token). After any initial bulk prefill,
 * decode keeps a receive window posted by sequence number: recv for seq s
 * lands in the slab in-slot (s-1) % slots and its completion is the arrival
 * signal. */
/* The Thunderbolt NHI frames everything at 4 KB: a message is a run of <=4 KB
 * frames ending in one whose `eof` nibble is 3.  Queue depths are counted in
 * FRAMES, so every max_send_wr / max_recv_wr in this file is a frame budget and
 * must be divided by this before it means "messages". */
/* Frame budget requested from create_qp, in 4 KB frames.
 *
 * UN-BANKED 2026-09-07, same day.  Default is back to 1024.
 *
 * It was banked on a PREFILL-ONLY measurement -- harness-windep runs gen 0, so
 * the +1.3% below came from a run with zero decode tokens -- and the first
 * decode ladder to carry it (NORMFUSE, v3 8921373 with shipping-base) read a
 * flat ~27 t/s against ~35 on the previous baseline, about -23% at every
 * context.  The fuse arm was OFF in that run, so NORM-FUSE is exonerated and
 * this is the live suspect: it is the only new thing in the shared baseline
 * that touches the transport, and the CQ is sized from it.
 *
 * Not yet proven -- the decisive A/B is 4095 vs 1024 on a decode ladder, one
 * env var, queued -- but a production default that may be costing 23% of decode
 * does not get to sit in the tree while we find out.
 *
 * The prefill result stands as a prefill result.  What was wrong was banking a
 * change to SHARED TRANSPORT STATE on one workload's measurement.
 *
 * Was: default 4095 (P2'-WIDE, 2026-09-07): +1.3% prefill @65536 on the TP2 pair,
 * 393.30 -> 398.01 t/s, 2 counterbalanced reps, bytes identical across arms and
 * no transport errors.  4095 is the ring maximum -- 4096 descriptors with one
 * slot held free -- and gives 1023 outstanding 16 KiB messages against the old
 * 1024 frames / 256 messages, so windows per 64 MiB exchange fall 16 -> 5 and
 * the per-window rendezvous count falls with them (22272 -> 6960 for the run).
 * Barrier time drops 29% (129.8 -> 92.1 ms/chunk) and the WR-build and post
 * columns drop with it, which is why throughput moves more than the barrier
 * line alone predicts.
 *
 * FIVE WINDOWS IS THE FLOOR, not a tuning choice.  A window carries at most
 * 4095 x 4096 = 15.996 MiB, and a 64 MiB exchange divided by that is 4.001 --
 * so it always rounds to 5, and the one reserved descriptor is exactly what
 * puts it over 4.  Larger messages do not help: the window cap is a frame
 * budget, so it is 15.996 MiB whatever the message size.
 *
 * An earlier run of this same arm reported +21.4%.  That was measured across a
 * desynced pair (see the send flow control above) and was an artifact; the
 * harness's fail-closed flag was right and the honest figure is +1.3%.
 *
 * DECLINED, 2026-09-07.  The default stays 1024 and the +1.3% is not taken.
 *
 * The arm is sound and the earlier suspicion against it was wrong: FRAMESDEC
 * showed 1024 and 4095 decode-equivalent from 2k to 131k, and the ~23% collapse
 * that got 4095 reverted was the TG16 threadgroup-rounding bug, not ring depth.
 * So this is not a rejection on evidence.
 *
 * It is a rejection on price.  Shipping 4095 reopens transport state: the CQ is
 * sized from this budget, the send flow control is calibrated against it, and
 * the rollback/resume suite has only ever certified 1024 -- P2'-GATE passed on a
 * build running the reverted default, so it certified the wrong thing.  Claiming
 * 1.3% of prefill costs a re-certification of the interrupt/resume/rollback path
 * and puts the deepest possible ring into production, and that trade was
 * declined deliberately.
 *
 * Set DS4_TP_RDMA_RECV_FRAMES=4095 to measure it again.  If it is ever taken,
 * `EXTRA_ENV="DS4_TP_RDMA_RECV_FRAMES=4095" bash harness-rb1.sh A` is the run
 * that has to pass first, and a 310k decode rung closes FRAMESDEC's one gap. */
static uint32_t tp_rdma_want_frames(void) {
    static uint32_t cached;
    if (!cached) {
        /* Explicit, not inherited: this value is a decision, not a leftover. */
        cached = 1024u;
        const char *e = getenv("DS4_TP_RDMA_RECV_FRAMES");
        if (e && e[0]) {
            const int v = atoi(e);
            /* Floor at one whole message: below 16 KiB / 4 KiB = 4 frames the
             * window rounds to zero messages and the bulk path posts nothing,
             * which presents as a hang rather than a refusal. */
            if (v > 0) {
                uint32_t want = (uint32_t)v > 4095u ? 4095u : (uint32_t)v;
                if (want < 4u) {
                    fprintf(stderr, "ds4-tp: DS4_TP_RDMA_RECV_FRAMES=%d is below "
                                    "one 16 KiB message; using 4\n", v);
                    want = 4u;
                }
                cached = want;
            }
        }
    }
    return cached;
}

#define DS4_TP_RDMA_FRAME_BYTES 4096u
#define DS4_TP_RDMA_MAX_MSG 16384
#define DS4_TP_RDMA_RECV_WINDOW 16
#define DS4_TP_RDMA_BULK_SLOTS 64
#define DS4_TP_RDMA_BULK_WR_TAG (UINT64_C(1) << 63)
#define DS4_TP_RDMA_BLOCK_WR_TAG (UINT64_C(1) << 61)

typedef struct {
    ds4_tp_verbs_api api;
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    struct ibv_port_attr port;
    union ibv_gid gid;
    int gid_index;
    uint32_t max_inline;
    ds4_tp_rdma_info peer;
    uint32_t send_outstanding;  /* signaled sends not yet reaped */
    uint32_t send_since_signal; /* unsignaled sends since the last reaped one */
    uint64_t recv_done;         /* highest gate seq whose recv completed */
    uint64_t last_gate_seq;     /* last real decode receive consumed */
    bool recv_window_active;    /* decode recvs are queued ahead */
    int warm_failed;            /* last warm-up round failed (dead direction) */
    uint32_t setup_attempt;     /* queue pair recreations so far */
    pthread_mutex_t post_lock;
    uint32_t recv_depth;        /* queue pair receive depth actually granted */
    uint32_t send_depth;        /* queue pair send depth actually granted */
    struct ibv_sge *win_sge;    /* window work request arrays (recv_depth) */
    struct ibv_recv_wr *win_rwr;
    struct ibv_send_wr *win_swr;
    /* Verify-block window (speculative decoding): one batch gate per layer,
     * receives posted a layer ahead of every send, no per-gate control
     * traffic (see ds4_tp_batch_block_begin). */
    bool block_active;
    uint32_t block_rows;
    uint32_t block_layers;
    uint32_t block_posted;      /* layers whose receives are posted */
    uint64_t block_recv_done;   /* row messages received in this block */
} ds4_tp_rdma;
#endif

struct ds4_tp {
    ds4_tp_options opt;
    int rank;                   /* 0 leader, 1 worker */
    int control_fd;
    int data_fd;                /* TCP fallback, headers, and verify gates */
    bool rdma_active;
    uint32_t peer_ctx;
    uint32_t n_layer;
    uint32_t n_embd;
    uint64_t vec_bytes;
    uint32_t n_slots;
    /* Decode gate schedule (see ds4_tp_identity). */
    uint32_t gate_slot_start;
    uint32_t gate_slot_step;
    uint32_t gates_per_token;
    uint64_t gate_slot_mask[DS4_TP_GATE_MASK_WORDS];
    uint8_t *slab;
    uint64_t slab_bytes;
    /* Slab regions, see ds4_tp.h layout comment. */
    uint64_t out_off;
    uint64_t in_off;
    uint64_t in_flags_off;
    uint64_t token_off;
    uint64_t out_flags_off;     /* local staging for RDMA flag writes */
    uint64_t gpu_flags_off;     /* GPU-written gate-ready flags (u32/slot) */
    uint64_t batch_out_off;     /* [layer][row] verify-block local partials */
    uint64_t batch_in_off;      /* [layer][row] verify-block peer partials */
    uint64_t prefill_bounce_out_off; /* one prefill chunk, local partial   */
    uint64_t prefill_bounce_in_off;  /* one prefill chunk, peer partial    */
    uint64_t timeout_sec;
    uint64_t gate_timeout_ms;   /* exchange, both ranks already in the gate */
    /* Where gate_timeout_ms came from. Printed at connect so a missing
     * override is visible in one line instead of inferred from a crash. */
    const char *gate_timeout_src;
    uint64_t meet_timeout_ms;   /* arrival barrier, peer may still be behind */
    atomic_bool failed;
    /* Serializes control-plane (control_fd) sends/receives and the paired
     * (send command -> wait ack) round trips.  Recursive so a caller may hold
     * it across a whole batch round trip while the per-call wrappers below
     * re-acquire it.  The data plane (data_fd) carries only TCP-fallback gate
     * exchanges from the service thread and is not covered by this lock. */
    pthread_mutex_t control_lock;
#ifdef DS4_TP_HAVE_VERBS
    ds4_tp_rdma rdma;
#endif
};

/* ------------------------------------------------------------------------
 * Small socket helpers (same conventions as ds4_distributed.c).
 * --------------------------------------------------------------------- */

static double tp_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* The two gate budgets, as absolute deadlines.  Which one a wait gets is the
 * whole distinction drawn at DS4_TP_DEFAULT_MEET_TIMEOUT_MS: everything from the
 * peer's gate header onwards is an exchange between two ranks that are both
 * already inside the gate; the wait for that header is not. */
static double tp_gate_deadline(const ds4_tp *tp) {
    return tp_now_sec() + (double)tp->gate_timeout_ms / 1000.0;
}

static double tp_meet_deadline(const ds4_tp *tp) {
    return tp_now_sec() + (double)tp->meet_timeout_ms / 1000.0;
}

static void tp_set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || !errlen) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

/* data_fd carries SO_SNDTIMEO/SO_RCVTIMEO (tp_socket_set_gate_timeout) so that
 * no gate can wedge inside an uninterruptible read.  That makes EAGAIN a clock
 * tick rather than an error, and only the caller knows which clock applies:
 * waiting for the peer to ARRIVE at a gate is a different budget from finishing
 * an exchange both ranks are already inside.  TP_IO_NO_DEADLINE means the
 * socket is the only bound, which is what every control_fd caller wants -- that
 * socket has no timeout set and can never report EAGAIN.
 *
 * Distinguishing the three failures matters more than it looks: collapsing them
 * into a bare 0 is what let a 750 ms arrival timeout surface as an unexplained
 * "TP gate exchange failed", with nothing in either rank's log to say why. */
#define TP_IO_NO_DEADLINE 0.0

typedef enum {
    TP_IO_OK = 0,
    TP_IO_TIMEOUT,  /* deadline passed with the transfer unfinished */
    TP_IO_CLOSED,   /* orderly close, possibly mid-frame */
    TP_IO_ERROR,    /* errno is set */
} tp_io_status;

/* A peer that trickles never returns EAGAIN, so the deadline has to be checked
 * after progress too or a slow drip outlives any budget. */
static int tp_io_expired(double deadline) {
    return deadline > TP_IO_NO_DEADLINE && tp_now_sec() >= deadline;
}

static const char *tp_io_why(tp_io_status status) {
    switch (status) {
    case TP_IO_OK:      return "ok";
    case TP_IO_TIMEOUT: return "timed out waiting for the peer";
    case TP_IO_CLOSED:  return "peer closed the gate socket";
    default:            return strerror(errno);
    }
}

static tp_io_status tp_write_until(int fd, const void *buf, size_t len,
                                   double deadline) {
    const char *p = buf;
    while (len) {
#ifdef MSG_NOSIGNAL
        ssize_t w = send(fd, p, len, MSG_NOSIGNAL);
#else
        ssize_t w = send(fd, p, len, 0);
#endif
        if (w < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                /* The socket timeout already slept for us, so this loop costs
                 * one iteration per SO_SNDTIMEO period, not a spin. */
                if (deadline > TP_IO_NO_DEADLINE && !tp_io_expired(deadline))
                    continue;
                return TP_IO_TIMEOUT;
            }
            return TP_IO_ERROR;
        }
        if (w == 0) return TP_IO_CLOSED;
        p += w;
        len -= (size_t)w;
        if (len && tp_io_expired(deadline)) return TP_IO_TIMEOUT;
    }
    return TP_IO_OK;
}

static tp_io_status tp_read_until(int fd, void *buf, size_t len,
                                  double deadline) {
    char *p = buf;
    while (len) {
        ssize_t r = read(fd, p, len);
        if (r < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                if (deadline > TP_IO_NO_DEADLINE && !tp_io_expired(deadline))
                    continue;
                return TP_IO_TIMEOUT;
            }
            return TP_IO_ERROR;
        }
        if (r == 0) return TP_IO_CLOSED;
        p += r;
        len -= (size_t)r;
        if (len && tp_io_expired(deadline)) return TP_IO_TIMEOUT;
    }
    return TP_IO_OK;
}

static int tp_write_full(int fd, const void *buf, size_t len) {
    return tp_write_until(fd, buf, len, TP_IO_NO_DEADLINE) == TP_IO_OK;
}

static int tp_read_full(int fd, void *buf, size_t len) {
    return tp_read_until(fd, buf, len, TP_IO_NO_DEADLINE) == TP_IO_OK;
}

static void tp_socket_tune(int fd) {
    int one = 1;
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    /* Gate exchanges are latency-critical 16KB messages; large socket
     * buffers only matter for the TCP fallback's pipelining. */
    int sz = 4 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
}

static int tp_socket_set_gate_timeout(int fd, uint64_t timeout_ms) {
    struct timeval tv = {
        .tv_sec = (time_t)(timeout_ms / 1000u),
        .tv_usec = (suseconds_t)((timeout_ms % 1000u) * 1000u),
    };
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0)
        return 0;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0)
        return 0;
    return 1;
}

#ifdef DS4_TP_HAVE_VERBS
/* UC queue pairs do not report a dead remote reliably.  The control socket
 * does, so sample it while polling an RDMA completion and abort before the
 * Metal command-buffer watchdog fires. */
static int tp_peer_closed(const ds4_tp *tp) {
    char byte;
    const ssize_t n = recv(tp->control_fd, &byte, 1,
                           MSG_PEEK | MSG_DONTWAIT);
    if (n == 0) return 1;
    if (n > 0) return 0;
    return errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR;
}
#endif

static int tp_listen(const char *host, int port, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    int rc = getaddrinfo(host && host[0] ? host : NULL, portbuf, &hints, &res);
    if (rc != 0) {
        tp_set_err(err, errlen, "tp listen resolve %s:%d: %s", host, port, gai_strerror(rc));
        return -1;
    }
    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && listen(fd, 2) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) tp_set_err(err, errlen, "tp listen %s:%d: %s", host, port, strerror(errno));
    return fd;
}

static int tp_dial(const char *host, int port, double timeout_sec, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    double deadline = tp_now_sec() + timeout_sec;
    int last_errno = 0;
    uint32_t attempts = 0;
    do {
        struct addrinfo hints = {0}, *res = NULL;
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        int gai = getaddrinfo(host, portbuf, &hints, &res);
        if (gai == 0) {
            for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
                int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
                if (fd < 0) continue;
                if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
                    freeaddrinfo(res);
                    return fd;
                }
                last_errno = errno;
                close(fd);
            }
            freeaddrinfo(res);
        }
        /* Retrying is normal while the peer loads its model; still say why
         * every ~10s so a wrong address or a policy block is visible. */
        if (attempts++ % 50 == 0) {
            fprintf(stderr, "ds4-tp: connecting to %s:%d ... (%s)\n", host, port,
                    gai != 0 ? gai_strerror(gai) :
                    last_errno ? strerror(last_errno) : "no address worked");
        }
        usleep(200 * 1000);
    } while (tp_now_sec() < deadline);
    tp_set_err(err, errlen, "tp connect %s:%d: %s", host, port,
               last_errno ? strerror(last_errno) : "unreachable");
    return -1;
}

static int tp_send_frame(int fd, uint32_t type, const void *payload, uint32_t bytes) {
    ds4_tp_frame_header h = { DS4_TP_MAGIC, type, bytes };
    if (!tp_write_full(fd, &h, sizeof(h))) return 0;
    if (bytes && !tp_write_full(fd, payload, bytes)) return 0;
    return 1;
}

/* Worker-side cancel poll for a mirrored prefill.
 *
 * Installed as the session cancel predicate only for the duration of one SYNC,
 * and called between prefill chunks.  Reading control_fd here is exclusive:
 * the worker's command loop is the sole reader and it is blocked inside this
 * very sync, and the leader is blocked awaiting our COMMAND_ACK, so CANCEL is
 * the only frame that can legitimately arrive.  Anything else means the stream
 * is misframed, which we report rather than try to interpret.
 *
 * Consuming the frame here is what keeps framing intact: the command loop must
 * not see a CANCEL for a sync that has already returned. */
typedef struct {
    ds4_tp *tp;
    uint64_t session_id;
    bool cancelled;
    bool broken;
} tp_worker_cancel_state;

static int tp_read_frame_header(int fd, uint32_t *type, uint32_t *bytes);

static bool tp_worker_prefill_cancelled(void *ud) {
    tp_worker_cancel_state *st = (tp_worker_cancel_state *)ud;
    if (!st) return false;
    if (st->cancelled) return true;
    if (st->broken) return false;
    /* Kill switch.  This poll is the only always-on part of the cancel
     * protocol, so it is the first thing to eliminate when the control stream
     * misbehaves. */
    if (getenv("DS4_TP_DISABLE_CANCEL_POLL") != NULL) return false;

    const int fd = st->tp->control_fd;
    /* control_lock, not bare access.  An earlier version read control_fd with
     * no lock on the reasoning that the command loop is the sole reader and is
     * blocked inside this very sync.  That is wrong: other control-plane
     * round trips run from this thread during a sync -- ds4_tp_hash_check()
     * sends and reads a HASH frame under control_lock, and the verify-commit
     * path does the same -- so an unlocked read here can steal a frame another
     * reader is waiting for and wedge both ranks.
     *
     * trylock rather than lock: this runs between prefill chunks on the hot
     * path, and a cancel that arrives while the lock is held is simply seen at
     * the next chunk boundary.  Blocking here to catch it marginally sooner
     * would be trading a certain stall for an uncertain one. */
    if (pthread_mutex_trylock(&st->tp->control_lock) != 0) return false;

    struct pollfd pfd = { .fd = fd, .events = POLLIN, .revents = 0 };
    const int ready = poll(&pfd, 1, 0);
    if (ready <= 0 || !(pfd.revents & POLLIN)) {
        pthread_mutex_unlock(&st->tp->control_lock);
        return false;
    }

    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(fd, &type, &bytes)) {
        pthread_mutex_unlock(&st->tp->control_lock);
        st->broken = true;
        ds4_log(stderr, DS4_LOG_ERROR,
                "tp worker: bad frame header while polling for cancel");
        ds4_tp_mark_failed(st->tp);
        return false;
    }
    if (type != (uint32_t)DS4_TP_FRAME_CANCEL || bytes != sizeof(uint64_t)) {
        pthread_mutex_unlock(&st->tp->control_lock);
        st->broken = true;
        ds4_log(stderr, DS4_LOG_ERROR,
                "tp worker: unexpected frame %u (%u bytes) during mirrored prefill",
                type, bytes);
        ds4_tp_mark_failed(st->tp);
        return false;
    }
    uint64_t session_id = 0;
    if (!tp_read_full(fd, &session_id, sizeof(session_id))) {
        pthread_mutex_unlock(&st->tp->control_lock);
        st->broken = true;
        ds4_tp_mark_failed(st->tp);
        return false;
    }
    pthread_mutex_unlock(&st->tp->control_lock);
    if (session_id != st->session_id) {
        /* Only one sync is in flight at a time, so a cancel for a different
         * session means the leader and worker disagree about which session is
         * running -- worse than a stale cancel, so do not silently drop it. */
        st->broken = true;
        ds4_log(stderr, DS4_LOG_ERROR,
                "tp worker: cancel for session %llu during sync of %llu",
                (unsigned long long)session_id,
                (unsigned long long)st->session_id);
        ds4_tp_mark_failed(st->tp);
        return false;
    }
    st->cancelled = true;
    return true;
}

static int tp_read_frame_header(int fd, uint32_t *type, uint32_t *bytes) {
    ds4_tp_frame_header h;
    if (!tp_read_full(fd, &h, sizeof(h))) return 0;
    if (h.magic != DS4_TP_MAGIC) return 0;
    *type = h.type;
    *bytes = h.bytes;
    return 1;
}

/* ------------------------------------------------------------------------
 * Options and CLI.
 * --------------------------------------------------------------------- */

bool ds4_tp_enabled(const ds4_tp_options *opt) {
    return opt && opt->role != DS4_TP_NONE;
}

void ds4_tp_usage(FILE *fp) {
    fprintf(fp,
        "Tensor parallelism (two identical machines):\n"
        "  --tensor-parallel           Use --role/--listen/--coordinator for a 50/50 TP pair.\n"
        "  --transport <auto|rdma|tcp> Gate transport (default auto).\n"
        "  --rdma-device <name>        Select a verbs device such as rdma_en1.\n"
        "  --rdma-gid-index <n>        Select the local verbs GID index.\n"
        "  --tensor-parallel-token-prefill\n"
        "                              GLM diagnostic: prefill one token at a time.\n"
        "  --debug-hash <n>            Cross-check hidden state every n tokens.\n");
}

int ds4_tp_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_tp_options *opt,
        char *err,
        size_t errlen)
{
    int i = *index;
    if (!strcmp(arg, "--tensor-parallel")) {
        opt->requested = true;
    } else if (!strcmp(arg, "--transport")) {
        if (i + 1 >= argc) goto missing;
        const char *v = argv[++i];
        if (!strcmp(v, "auto")) opt->transport = DS4_TP_TRANSPORT_AUTO;
        else if (!strcmp(v, "rdma")) opt->transport = DS4_TP_TRANSPORT_RDMA;
        else if (!strcmp(v, "tcp")) opt->transport = DS4_TP_TRANSPORT_TCP;
        else {
            tp_set_err(err, errlen, "invalid %s value: %s", arg, v);
            return DS4_TP_CLI_ERROR;
        }
    } else if (!strcmp(arg, "--rdma-device")) {
        if (i + 1 >= argc) goto missing;
        opt->rdma_device = argv[++i];
    } else if (!strcmp(arg, "--rdma-gid-index")) {
        if (i + 1 >= argc) goto missing;
        char *end = NULL;
        errno = 0;
        long value = strtol(argv[++i], &end, 10);
        if (errno != 0 || !end || *end != '\0' || value < 0 || value > INT_MAX) {
            tp_set_err(err, errlen, "invalid --rdma-gid-index %s", argv[i]);
            return DS4_TP_CLI_ERROR;
        }
        opt->rdma_gid_index = (int)value;
        opt->rdma_gid_index_set = true;
    } else if (!strcmp(arg, "--tensor-parallel-token-prefill")) {
        opt->glm_token_prefill = true;
    } else if (!strcmp(arg, "--debug-hash")) {
        if (i + 1 >= argc) goto missing;
        opt->debug_hash = atoi(argv[++i]);
    } else {
        return DS4_TP_CLI_NOT_MATCHED;
    }
    *index = i;
    return DS4_TP_CLI_MATCHED;
missing:
    tp_set_err(err, errlen, "%s requires an argument", arg);
    return DS4_TP_CLI_ERROR;
}

int ds4_tp_adopt_distributed_options(
        ds4_tp_options *tp,
        ds4_distributed_options *dist,
        char *err,
        size_t errlen)
{
    if (!tp || !dist || !tp->requested) return 1;
    if (tp->role != DS4_TP_NONE) {
        tp_set_err(err, errlen,
                   "--tensor-parallel selects its role through --role");
        return 0;
    }
    if (dist->role == DS4_DISTRIBUTED_NONE) {
        tp_set_err(err, errlen,
                   "--tensor-parallel requires --role coordinator or --role worker");
        return 0;
    }
    if (dist->layers.set) {
        tp_set_err(err, errlen,
                   "tensor parallelism always uses one 50/50 worker; omit --layers");
        return 0;
    }
    if (dist->prefill_chunk || dist->prefill_window || dist->activation_bits ||
        dist->replay_check || dist->debug) {
        tp_set_err(err, errlen,
                   "--dist-* and distributed debug options cannot be used with --tensor-parallel");
        return 0;
    }

    if (dist->role == DS4_DISTRIBUTED_COORDINATOR) {
        if (!dist->listen_host || dist->listen_port <= 0) {
            tp_set_err(err, errlen,
                       "--role coordinator --tensor-parallel requires --listen HOST PORT");
            return 0;
        }
        if (dist->coordinator_host || dist->coordinator_port) {
            tp_set_err(err, errlen,
                       "--role coordinator must not use --coordinator");
            return 0;
        }
        tp->role = DS4_TP_LEADER;
        tp->listen_host = dist->listen_host;
        tp->listen_port = dist->listen_port;
    } else if (dist->role == DS4_DISTRIBUTED_WORKER) {
        if (!dist->coordinator_host || dist->coordinator_port <= 0) {
            tp_set_err(err, errlen,
                       "--role worker --tensor-parallel requires --coordinator HOST PORT");
            return 0;
        }
        if (dist->listen_host || dist->listen_port) {
            tp_set_err(err, errlen,
                       "--role worker --tensor-parallel must not use --listen");
            return 0;
        }
        tp->role = DS4_TP_WORKER;
        tp->leader_host = dist->coordinator_host;
        tp->leader_port = dist->coordinator_port;
    } else {
        tp_set_err(err, errlen, "invalid tensor-parallel role");
        return 0;
    }

    memset(dist, 0, sizeof(*dist));
    return 1;
}

int ds4_tp_validate_engine_options(
        const ds4_engine_options *opt,
        char *err,
        size_t errlen)
{
    if (!ds4_tp_enabled(&opt->tp)) {
        if (opt->tp.requested || opt->tp.transport != DS4_TP_TRANSPORT_AUTO ||
            opt->tp.rdma_device || opt->tp.rdma_gid_index_set ||
            opt->tp.glm_token_prefill || opt->tp.debug_hash != 0) {
            tp_set_err(err, errlen,
                       "tensor-parallel options require --tensor-parallel and --role");
            return 0;
        }
        return 1;
    }
    if (opt->backend != DS4_BACKEND_METAL) {
        tp_set_err(err, errlen, "tensor parallelism requires the Metal backend");
        return 0;
    }
    if (opt->distributed.role != DS4_DISTRIBUTED_NONE) {
        tp_set_err(err, errlen, "tensor parallelism and --role distributed modes are exclusive");
        return 0;
    }
    /* Speculative drafting (DSpark/MTP) is allowed on the leader: the
     * verify block is mirrored to the worker via DS4_TP_FRAME_VERIFY and
     * the legacy MTP path falls back to per-token decode under TP. */
    if (opt->load_slice) {
        tp_set_err(err, errlen, "tensor parallelism does not use distributed layer slices");
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Slab layout.
 * --------------------------------------------------------------------- */

/* One prefill chunk's worth of rows, each direction.  DS4_GLM53_PREFILL_CHUNK
 * caps the chunk at 4096, so this is 64 MiB per direction at n_embd 4096 -- paid
 * once, against 2.28 GB of CPU memcpy per 1.14 GB exchange if the bounce lives
 * outside the registered region. */
#define DS4_TP_PREFILL_BOUNCE_ROWS 4096u

uint64_t ds4_tp_slab_bytes(uint32_t n_layer, uint32_t n_embd) {
    uint64_t vec = (uint64_t)n_embd * sizeof(float);
    uint64_t slots = (uint64_t)n_layer * DS4_TP_GATES_PER_LAYER;
    return slots * vec * 2 +    /* out + in vectors */
           slots * 8 * 2 +      /* in flags + out flag staging */
           16 +                 /* token slot */
           (uint64_t)DS4_TP_GPU_FLAG_BANK_SLOTS * 4u * 2u +
                                /* row + verifier GPU-ready flag banks */
           (uint64_t)n_layer * DS4_TP_BATCH_MAX_ROWS * vec * 2 + /* batch out+in */
           (uint64_t)DS4_TP_PREFILL_BOUNCE_ROWS * vec * 2;  /* prefill bounce */
}

static void tp_slab_layout(ds4_tp *tp) {
    uint64_t vec = tp->vec_bytes;
    uint64_t slots = tp->n_slots;
    tp->out_off = 0;
    tp->in_off = slots * vec;
    tp->in_flags_off = tp->in_off + slots * vec;
    tp->token_off = tp->in_flags_off + slots * 8;
    tp->out_flags_off = tp->token_off + 16;
    tp->gpu_flags_off = tp->out_flags_off + slots * 8;
    tp->batch_out_off = tp->gpu_flags_off +
                        (uint64_t)DS4_TP_GPU_FLAG_BANK_SLOTS * 4u * 2u;
    tp->batch_in_off = tp->batch_out_off +
                       (uint64_t)tp->n_layer * DS4_TP_BATCH_MAX_ROWS * vec;
    tp->prefill_bounce_out_off = tp->batch_in_off +
                     (uint64_t)tp->n_layer * DS4_TP_BATCH_MAX_ROWS * vec;
    tp->prefill_bounce_in_off = tp->prefill_bounce_out_off +
                     (uint64_t)DS4_TP_PREFILL_BOUNCE_ROWS * vec;
    tp->slab_bytes = tp->prefill_bounce_in_off +
                     (uint64_t)DS4_TP_PREFILL_BOUNCE_ROWS * vec;
}

uint64_t ds4_tp_slab_prefill_bounce_out_offset(const ds4_tp *tp) {
    return tp->prefill_bounce_out_off;
}

uint64_t ds4_tp_slab_prefill_bounce_in_offset(const ds4_tp *tp) {
    return tp->prefill_bounce_in_off;
}

uint64_t ds4_tp_slab_prefill_bounce_bytes(const ds4_tp *tp) {
    return (uint64_t)DS4_TP_PREFILL_BOUNCE_ROWS * tp->vec_bytes;
}

uint64_t ds4_tp_slab_gpu_flags_offset(const ds4_tp *tp) {
    return tp->gpu_flags_off;
}

static uint32_t tp_slot(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    (void)tp;
    return layer * DS4_TP_GATES_PER_LAYER + gate;
}

uint64_t ds4_tp_slab_out_offset(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    return tp->out_off + (uint64_t)tp_slot(tp, layer, gate) * tp->vec_bytes;
}

uint64_t ds4_tp_slab_in_offset(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    return tp->in_off + (uint64_t)tp_slot(tp, layer, gate) * tp->vec_bytes;
}

uint64_t ds4_tp_slab_batch_out_offset(const ds4_tp *tp, uint32_t layer) {
    return tp->batch_out_off +
           (uint64_t)layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
}

uint64_t ds4_tp_slab_batch_in_offset(const ds4_tp *tp, uint32_t layer) {
    return tp->batch_in_off +
           (uint64_t)layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
}

static uint32_t tp_gate_mask_count(
        const uint64_t mask[DS4_TP_GATE_MASK_WORDS]) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < DS4_TP_GATE_MASK_WORDS; i++)
        count += (uint32_t)__builtin_popcountll(mask[i]);
    return count;
}

static int tp_gate_mask_fits(
        const uint64_t mask[DS4_TP_GATE_MASK_WORDS],
        uint32_t n_slots) {
    for (uint32_t word = 0; word < DS4_TP_GATE_MASK_WORDS; word++) {
        uint64_t bits = mask[word];
        while (bits) {
            const uint32_t slot =
                word * 64u + (uint32_t)__builtin_ctzll(bits);
            if (slot >= n_slots) return 0;
            bits &= bits - 1u;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * RDMA path.
 * --------------------------------------------------------------------- */

#ifdef DS4_TP_HAVE_VERBS

static int tp_rdma_load_api(ds4_tp_verbs_api *api) {
    if (api->handle) return 1;
    void *h = dlopen("/usr/lib/librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) return 0;
#define TP_SYM(field, name) \
    do { \
        api->field = (__typeof__(api->field))dlsym(h, name); \
        if (!api->field) { dlclose(h); return 0; } \
    } while (0)
    TP_SYM(get_device_list, "ibv_get_device_list");
    TP_SYM(free_device_list, "ibv_free_device_list");
    TP_SYM(get_device_name, "ibv_get_device_name");
    TP_SYM(open_device, "ibv_open_device");
    TP_SYM(close_device, "ibv_close_device");
    TP_SYM(query_device, "ibv_query_device");
    TP_SYM(query_port, "ibv_query_port");
    TP_SYM(query_gid, "ibv_query_gid");
    TP_SYM(alloc_pd, "ibv_alloc_pd");
    TP_SYM(dealloc_pd, "ibv_dealloc_pd");
    TP_SYM(reg_mr, "ibv_reg_mr");
    TP_SYM(dereg_mr, "ibv_dereg_mr");
    TP_SYM(create_cq, "ibv_create_cq");
    TP_SYM(destroy_cq, "ibv_destroy_cq");
    TP_SYM(create_qp, "ibv_create_qp");
    TP_SYM(destroy_qp, "ibv_destroy_qp");
    TP_SYM(modify_qp, "ibv_modify_qp");
    TP_SYM(query_qp, "ibv_query_qp");
#undef TP_SYM
    api->handle = h;
    return 1;
}

/* Probe only: does this machine expose a verbs device right now? */
static int tp_rdma_probe(ds4_tp_verbs_api *api) {
    if (!tp_rdma_load_api(api)) return 0;
    int num = 0;
    struct ibv_device **devs = api->get_device_list(&num);
    if (!devs) return 0;
    api->free_device_list(devs);
    return num > 0;
}

/* Completion-queue entries for a given frame budget.
 *
 * The CQ was a flat 512 and that is now far too small.  Two changes stacked:
 * the frame budget went 1024 -> 4095 frames (256 -> 1023 messages per window),
 * and the bulk send loop now signals EVERY work request rather than one per
 * 64-message batch, because its flow control compares completions against
 * sends.  Signalling is right -- depending on a provider to complete
 * unsignalled sends is a hang the day that behaviour changes -- but it took the
 * send side from ~16 completions per window to 1023.
 *
 * Worst case in flight is a full receive window plus a full send window plus
 * the decode path's own traffic, so size from both directions with slack.
 * Overrunning a CQ is not a clean error on any provider: it is lost completions
 * and a spin to the deadline. */
static uint32_t tp_rdma_cq_entries(uint32_t frames, uint32_t max_cqe) {
    const uint32_t fpm =
        (DS4_TP_RDMA_MAX_MSG + DS4_TP_RDMA_FRAME_BYTES - 1u) / DS4_TP_RDMA_FRAME_BYTES;
    const uint32_t msgs = frames / fpm ? frames / fpm : 1u;
    uint64_t want = (uint64_t)msgs * 2u + 256u;
    if (want < 512u) want = 512u;            /* never below the old floor */
    if (max_cqe && want > max_cqe) want = max_cqe;
    return (uint32_t)want;
}

/* Shared by bring-up and the retry path so the two cannot drift apart. */
static struct ibv_cq *tp_rdma_make_cq(ds4_tp_rdma *r) {
    struct ibv_device_attr cqa;
    memset(&cqa, 0, sizeof(cqa));
    uint32_t max_cqe = 0;
    if (r->api.query_device && r->api.query_device(r->ctx, &cqa) == 0)
        max_cqe = (uint32_t)cqa.max_cqe;
    const uint32_t cqe = tp_rdma_cq_entries(tp_rdma_want_frames(), max_cqe);
    struct ibv_cq *cq = r->api.create_cq(r->ctx, (int)cqe, NULL, NULL, 0);
    if (!cq && cqe > 512u) {
        fprintf(stderr, "ds4-tp: create_cq(%u) refused; retrying at 512\n", cqe);
        cq = r->api.create_cq(r->ctx, 512, NULL, NULL, 0);
    }
    static int announced;
    if (cq && !announced) {
        announced = 1;
        fprintf(stderr, "ds4-tp: completion queue %u entries (frame budget %u, "
                        "device max_cqe %u)\n", cqe, tp_rdma_want_frames(), max_cqe);
    }
    return cq;
}

static int tp_rdma_open(ds4_tp *tp, char *err, size_t errlen) {
    ds4_tp_rdma *r = &tp->rdma;
    int num = 0;
    struct ibv_device **devs = r->api.get_device_list(&num);
    if (!devs || num == 0) {
        tp_set_err(err, errlen, "tp rdma: no verbs devices");
        if (devs) r->api.free_device_list(devs);
        return 0;
    }
    /* One verbs device per Thunderbolt port (rdma_enN); pick the active one
     * unless the caller selected a device explicitly. */
    const char *want_name = tp->opt.rdma_device;
    char states[256] = "";
    for (int i = 0; i < num && !r->ctx; i++) {
        const char *name = r->api.get_device_name(devs[i]);
        if (want_name && strcmp(want_name, name) != 0) continue;
        struct ibv_context *ctx = r->api.open_device(devs[i]);
        if (!ctx) continue;
        struct ibv_port_attr pa;
        if (r->api.query_port(ctx, 1, &pa) == 0 &&
            (pa.state == IBV_PORT_ACTIVE || want_name)) {
            r->ctx = ctx;
            r->port = pa;
            fprintf(stderr, "ds4-tp: rdma device %s (port state %d)\n", name, (int)pa.state);
            break;
        }
        size_t off = strlen(states);
        snprintf(states + off, sizeof(states) - off, "%s%s=%d",
                 off ? ", " : "", name, (int)pa.state);
        r->api.close_device(ctx);
    }
    r->api.free_device_list(devs);
    if (!r->ctx) {
        tp_set_err(err, errlen,
                   "tp rdma: no device with an active port (%s); is the peer up "
                   "and rdma_ctl enabled on both machines?", states);
        return 0;
    }
    /* The driver only connects through the IPv4-mapped GID
     * (::ffff:a.b.c.d), which exists only when the Thunderbolt member
     * interface carries an IPv4 address (the bridge's address does not
     * count). */
    r->gid_index = -1;
    if (tp->opt.rdma_gid_index_set) {
        r->gid_index = tp->opt.rdma_gid_index;
        if (r->api.query_gid(r->ctx, 1, r->gid_index, &r->gid) != 0) {
            tp_set_err(err, errlen, "tp rdma: query_gid(%d): %s",
                       r->gid_index, strerror(errno));
            return 0;
        }
    } else {
        for (int i = 0; i < r->port.gid_tbl_len; i++) {
            union ibv_gid tmp;
            if (r->api.query_gid(r->ctx, 1, i, &tmp) != 0) continue;
            uint64_t hi;
            uint16_t mid, v4tag;
            memcpy(&hi, &tmp.raw[0], 8);
            memcpy(&mid, &tmp.raw[8], 2);
            memcpy(&v4tag, &tmp.raw[10], 2);
            if (hi == 0 && mid == 0 && v4tag == 0xffff) {
                r->gid = tmp;
                r->gid_index = i;
                break;
            }
        }
        if (r->gid_index < 0) {
            tp_set_err(err, errlen,
                       "tp rdma: no IPv4-mapped GID on the active port; give the "
                       "Thunderbolt interface its own IPv4 (e.g. sudo ifconfig en1 "
                       "inet 10.99.0.2/30 alias) on both machines");
            return 0;
        }
    }
    r->pd = r->api.alloc_pd(r->ctx);
    if (!r->pd) {
        tp_set_err(err, errlen, "tp rdma: alloc_pd failed");
        return 0;
    }
    {
        struct ibv_device_attr da;
        memset(&da, 0, sizeof(da));
        if (r->api.query_device && r->api.query_device(r->ctx, &da) == 0) {
            fprintf(stderr, "ds4-tp: rdma device limits: max_qp_wr %d, max_sge %d, max_cqe %d, max_mr_size %llu\n",
                    da.max_qp_wr, da.max_sge, da.max_cqe, (unsigned long long)da.max_mr_size);
        }
    }
    r->cq = tp_rdma_make_cq(r);
    if (!r->cq) {
        tp_set_err(err, errlen, "tp rdma: create_cq failed");
        return 0;
    }
    struct ibv_qp_init_attr qia = {0};
    qia.send_cq = r->cq;
    qia.recv_cq = r->cq;
    qia.qp_type = IBV_QPT_UC;
    const uint32_t want_frames = tp_rdma_want_frames();
    qia.cap.max_send_wr = want_frames;
    qia.cap.max_recv_wr = want_frames;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    qia.cap.max_inline_data = 0;
    r->qp = r->api.create_qp(r->pd, &qia);
    if (!r->qp && want_frames > 1024u) {
        /* A refused larger ask must NOT silently become the 256/64 floor --
         * that is a 4x SMALLER window than the default, so a wide arm that
         * failed to get its frames would read as "wider windows made it
         * worse" when the wide window never existed. */
        fprintf(stderr, "ds4-tp: create_qp refused %u frames; retrying at 1024\n",
                want_frames);
        qia.cap.max_send_wr = qia.cap.max_recv_wr = 1024u;
        r->qp = r->api.create_qp(r->pd, &qia);
    }
    if (!r->qp) {
        qia.cap.max_send_wr = 256;
        qia.cap.max_recv_wr = 64;
        r->qp = r->api.create_qp(r->pd, &qia);
    }
    if (!r->qp) {
        tp_set_err(err, errlen, "tp rdma: create_qp(UC): %s", strerror(errno));
        return 0;
    }
    r->max_inline = qia.cap.max_inline_data;
    r->recv_depth = qia.cap.max_recv_wr ? qia.cap.max_recv_wr : 64u;
    r->send_depth = qia.cap.max_send_wr ? qia.cap.max_send_wr : 256u;
    if (r->recv_depth > 4096u) r->recv_depth = 4096u;
    if (r->send_depth > 4096u) r->send_depth = 4096u;

    pthread_mutex_init(&r->post_lock, NULL);
    return 1;
}

#define DS4_TP_RDMA_WARM_WR_TAG (UINT64_C(1) << 62)

static const char *tp_wc_status_str(int status);

/* Warm-up round after the ready barrier.  A fresh UC queue pair over
 * Thunderbolt RDMA sometimes cannot deliver in one direction at all, and UC
 * never reports a lost message: such runs timed out at the first bulk round
 * with no completions.  This round proves both directions before any real
 * traffic (the caller recreates the queue pair when it fails).  Each side posts exactly one full-size receive and resends
 * a tagged message until the peer confirms receipt over the control
 * channel.  A dropped UC message never arrives late, so once both sides
 * have received, no stray message can reach the receive queue. */
/* "Receives posted" barrier.  On Apple's Thunderbolt RDMA a UC send that
 * reaches a queue pair with no receive posted is not just lost: that
 * direction of the pair stops delivering for good.  Every sender therefore
 * waits for the peer's word that its receives are posted before posting
 * sends (one control-channel round trip). */
static int tp_rdma_posted_barrier(ds4_tp *tp, uint32_t tag) {
    uint32_t t = 0, b = 0, theirs = 0;
    if (!tp_send_frame(tp->control_fd, DS4_TP_FRAME_RDMA_POSTED, &tag, sizeof(tag))) return 0;
    if (!tp_read_frame_header(tp->control_fd, &t, &b) ||
        t != DS4_TP_FRAME_RDMA_POSTED || b != sizeof(theirs) ||
        !tp_read_full(tp->control_fd, &theirs, sizeof(theirs)) || theirs != tag) {
        return 0;
    }
    return 1;
}

static int tp_rdma_warm_up(ds4_tp *tp, char *err, size_t errlen) {
    ds4_tp_rdma *r = &tp->rdma;
    uint8_t *wsend = tp->slab + tp->batch_out_off;
    uint8_t *wrecv = tp->slab + tp->batch_in_off;
    /* Full-size message: 64 B warm messages pass on a fresh direction while
     * the first 16 KB chunks are still dropped. */
    const uint32_t bytes = DS4_TP_RDMA_MAX_MSG;
    memset(wsend, 0x5A, bytes);
    memset(wrecv, 0, bytes);
    struct ibv_sge rsge = {
        .addr = (uintptr_t)wrecv, .length = bytes, .lkey = r->mr->lkey,
    };
    struct ibv_recv_wr rwr;
    memset(&rwr, 0, sizeof(rwr));
    rwr.wr_id = DS4_TP_RDMA_WARM_WR_TAG;
    rwr.sg_list = &rsge;
    rwr.num_sge = 1;
    struct ibv_recv_wr *bad_r = NULL;
    if (ibv_post_recv(r->qp, &rwr, &bad_r) != 0) {
        tp_set_err(err, errlen, "tp rdma: warm-up post_recv: %s", strerror(errno));
        return 0;
    }
    if (!tp_rdma_posted_barrier(tp, 0xfeedu)) {
        tp_set_err(err, errlen, "tp rdma: warm-up posted barrier failed");
        return 0;
    }
    int got_recv = 0, peer_got = 0, attempts = 0;
    for (int attempt = 0; attempt < 20 && !(got_recv && peer_got); attempt++) {
        attempts = attempt + 1;
        int send_done = peer_got;
        if (!peer_got) {
            struct ibv_sge ssge = {
                .addr = (uintptr_t)wsend, .length = bytes, .lkey = r->mr->lkey,
            };
            struct ibv_send_wr swr;
            memset(&swr, 0, sizeof(swr));
            swr.wr_id = DS4_TP_RDMA_WARM_WR_TAG;
            swr.sg_list = &ssge;
            swr.num_sge = 1;
            swr.opcode = IBV_WR_SEND;
            swr.send_flags = IBV_SEND_SIGNALED;
            struct ibv_send_wr *bad_s = NULL;
            if (ibv_post_send(r->qp, &swr, &bad_s) != 0) {
                tp_set_err(err, errlen, "tp rdma: warm-up post_send: %s", strerror(errno));
                return 0;
            }
        }
        const double deadline = tp_now_sec() + 0.1;
        while (tp_now_sec() < deadline && !(send_done && got_recv)) {
            struct ibv_wc wc[8];
            int n = ibv_poll_cq(r->cq, 8, wc);
            if (n < 0) {
                tp_set_err(err, errlen, "tp rdma: warm-up poll_cq failed");
                return 0;
            }
            for (int i = 0; i < n; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    tp_set_err(err, errlen, "tp rdma: warm-up completion error: %s",
                               tp_wc_status_str(wc[i].status));
                    return 0;
                }
                if (wc[i].wr_id != DS4_TP_RDMA_WARM_WR_TAG) continue;
                if (wc[i].opcode & IBV_WC_RECV) got_recv = 1;
                else send_done = 1;
            }
        }
        uint32_t mine = (uint32_t)got_recv, theirs = 0, t = 0, b = 0;
        if (!tp_send_frame(tp->control_fd, DS4_TP_FRAME_RDMA_WARM, &mine, sizeof(mine)) ||
            !tp_read_frame_header(tp->control_fd, &t, &b) ||
            t != DS4_TP_FRAME_RDMA_WARM || b != sizeof(theirs) ||
            !tp_read_full(tp->control_fd, &theirs, sizeof(theirs))) {
            tp_set_err(err, errlen, "tp rdma: warm-up status exchange failed");
            return 0;
        }
        peer_got = theirs != 0;
    }
    if (!(got_recv && peer_got)) {
        tp_set_err(err, errlen, "tp rdma: warm-up failed after %d attempts (recv %d, peer %d)",
                   attempts, got_recv, peer_got);
        r->warm_failed = 1;
        return 0;
    }
    fprintf(stderr, "ds4-tp: rdma warm-up ok (%d attempt%s), queue depth recv %u send %u\n",
            attempts, attempts == 1 ? "" : "s", r->recv_depth, r->send_depth);
    return 1;
}

static int tp_rdma_post_gate_recv(ds4_tp *tp, uint64_t seq);

/* A freshly created UC queue pair over Thunderbolt RDMA sometimes cannot
 * deliver in one direction at all (every send silently dropped, observed
 * right after a working run and after role swaps).  Recreating the pair
 * (new queue pair number) and re-exchanging the info fixes it; both sides
 * take this path in lockstep because the warm-up status exchange tells
 * them the same outcome. */
static int tp_rdma_recreate_qp(ds4_tp *tp, char *err, size_t errlen) {
    ds4_tp_rdma *r = &tp->rdma;
    if (r->qp) { r->api.destroy_qp(r->qp); r->qp = NULL; }
    if (r->cq) { r->api.destroy_cq(r->cq); r->cq = NULL; }
    r->cq = tp_rdma_make_cq(r);
    if (!r->cq) {
        tp_set_err(err, errlen, "tp rdma: create_cq (retry) failed");
        return 0;
    }
    struct ibv_qp_init_attr qia = {0};
    qia.send_cq = r->cq;
    qia.recv_cq = r->cq;
    qia.qp_type = IBV_QPT_UC;
    const uint32_t want_frames = tp_rdma_want_frames();
    qia.cap.max_send_wr = want_frames;
    qia.cap.max_recv_wr = want_frames;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    qia.cap.max_inline_data = 0;
    r->qp = r->api.create_qp(r->pd, &qia);
    if (!r->qp && want_frames > 1024u) {
        /* A refused larger ask must NOT silently become the 256/64 floor --
         * that is a 4x SMALLER window than the default, so a wide arm that
         * failed to get its frames would read as "wider windows made it
         * worse" when the wide window never existed. */
        fprintf(stderr, "ds4-tp: create_qp refused %u frames; retrying at 1024\n",
                want_frames);
        qia.cap.max_send_wr = qia.cap.max_recv_wr = 1024u;
        r->qp = r->api.create_qp(r->pd, &qia);
    }
    if (!r->qp) {
        qia.cap.max_send_wr = 256;
        qia.cap.max_recv_wr = 64;
        r->qp = r->api.create_qp(r->pd, &qia);
    }
    if (!r->qp) {
        tp_set_err(err, errlen, "tp rdma: create_qp (retry): %s", strerror(errno));
        return 0;
    }
    r->max_inline = qia.cap.max_inline_data;
    r->recv_depth = qia.cap.max_recv_wr ? qia.cap.max_recv_wr : 64u;
    r->send_depth = qia.cap.max_send_wr ? qia.cap.max_send_wr : 256u;
    if (r->recv_depth > 4096u) r->recv_depth = 4096u;
    if (r->send_depth > 4096u) r->send_depth = 4096u;
    r->send_outstanding = 0;
    r->recv_done = 0;
    return 1;
}

static int tp_rdma_register_and_exchange(ds4_tp *tp, char *err, size_t errlen) {
    ds4_tp_rdma *r = &tp->rdma;
    if (!r->mr) {
        r->mr = r->api.reg_mr(r->pd, tp->slab, tp->slab_bytes,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_REMOTE_WRITE);
    }
    if (!r->mr) {
        tp_set_err(err, errlen, "tp rdma: reg_mr(%llu bytes): %s",
                   (unsigned long long)tp->slab_bytes, strerror(errno));
        return 0;
    }
    ds4_tp_rdma_info mine = {0};
    mine.slab_base = (uint64_t)(uintptr_t)tp->slab;
    mine.rkey = r->mr->rkey;
    mine.qpn = r->qp->qp_num;
    mine.psn = ((uint32_t)(getpid() ^ (uintptr_t)tp) + r->setup_attempt * 0x1000u) & 0xffffff;
    mine.mtu = (uint32_t)r->port.active_mtu;
    mine.lid = r->port.lid;
    memcpy(mine.gid, r->gid.raw, 16);
    mine.link_layer = r->port.link_layer;
    if (!tp_send_frame(tp->control_fd, DS4_TP_FRAME_RDMA_INFO, &mine, sizeof(mine))) {
        tp_set_err(err, errlen, "tp rdma: info send failed");
        return 0;
    }
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_RDMA_INFO || bytes != sizeof(r->peer) ||
        !tp_read_full(tp->control_fd, &r->peer, sizeof(r->peer))) {
        tp_set_err(err, errlen, "tp rdma: info recv failed");
        return 0;
    }

    /* INIT -> RTR -> RTS with the exact recipe the driver accepts (same as
     * JACCL): MTU 1024 and GRH via the IPv4-mapped GID. */
    struct ibv_qp_attr a = {0};
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = 1;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                        IBV_ACCESS_REMOTE_WRITE;
    if (r->api.modify_qp(r->qp, &a,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify INIT: %s", strerror(errno));
        return 0;
    }
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = IBV_MTU_1024;
    a.dest_qp_num = r->peer.qpn;
    a.rq_psn = r->peer.psn;
    a.ah_attr.dlid = (uint16_t)r->peer.lid;
    a.ah_attr.port_num = 1;
    a.ah_attr.is_global = 1;
    memcpy(a.ah_attr.grh.dgid.raw, r->peer.gid, 16);
    a.ah_attr.grh.sgid_index = (uint8_t)r->gid_index;
    a.ah_attr.grh.hop_limit = 1;
    if (r->api.modify_qp(r->qp, &a,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify RTR: %s", strerror(errno));
        return 0;
    }
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS;
    a.sq_psn = mine.psn;
    if (r->api.modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify RTS: %s", strerror(errno));
        return 0;
    }
    if (tp->vec_bytes > 2ull * DS4_TP_RDMA_MAX_MSG) {
        tp_set_err(err, errlen,
                   "tp rdma: gate vector %llu bytes exceeds twice the driver's "
                   "%u message limit",
                   (unsigned long long)tp->vec_bytes, DS4_TP_RDMA_MAX_MSG);
        return 0;
    }
    if (tp->vec_bytes > DS4_TP_RDMA_MAX_MSG) {
        fprintf(stderr,
                "ds4-tp: rdma gate vectors ride as 2 chunked messages "
                "(%llu bytes > %u limit)\n",
                (unsigned long long)tp->vec_bytes, DS4_TP_RDMA_MAX_MSG);
    }
    /* Leave the receive queue empty for an initial bulk prefill.  The first
     * decode gate arms the normal lookahead window after prefill finishes. */
    if (!tp_send_frame(tp->control_fd, DS4_TP_FRAME_RDMA_READY, NULL, 0)) {
        tp_set_err(err, errlen, "tp rdma: ready send failed");
        return 0;
    }
    uint32_t rtype = 0, rbytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &rtype, &rbytes) ||
        rtype != DS4_TP_FRAME_RDMA_READY || rbytes != 0) {
        tp_set_err(err, errlen, "tp rdma: ready barrier failed");
        return 0;
    }
    if (getenv("DS4_TP_DISABLE_RDMA_WARMUP") == NULL &&
        !tp_rdma_warm_up(tp, err, errlen)) return 0;
    return 1;
}

/* ibv_wc_status_str lives in librdma; resolve lazily to keep the dlopen-only
 * linkage discipline. */
static const char *tp_wc_status_str(int status) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "wc status %d", status);
    return buf;
}

/* Slab slot a given gate seq lands in.  DS4 fires every slot in order
 * (identity mapping); GLM's schedule from the hello skips dense layers
 * and the ATTN slots. */
static uint32_t tp_gate_slot(const ds4_tp *tp, uint64_t seq) {
    /* Loop rather than naming three words: with four gates a GLM 5.3 mask
     * reaches slot 183, and a hardcoded three-word test is one gate away from
     * silently treating a masked schedule as an unmasked one. */
    uint64_t any = 0;
    for (uint32_t w = 0; w < DS4_TP_GATE_MASK_WORDS; w++) any |= tp->gate_slot_mask[w];
    if (any) {
        uint32_t ordinal = (uint32_t)((seq - 1) % tp->gates_per_token);
        for (uint32_t word = 0; word < DS4_TP_GATE_MASK_WORDS; word++) {
            uint64_t bits = tp->gate_slot_mask[word];
            const uint32_t count = (uint32_t)__builtin_popcountll(bits);
            if (ordinal >= count) {
                ordinal -= count;
                continue;
            }
            while (ordinal > 0) {
                bits &= bits - 1u;
                ordinal--;
            }
            return word * 64u + (uint32_t)__builtin_ctzll(bits);
        }
        return tp->n_slots;
    }
    if (tp->gates_per_token == 0)
        return (uint32_t)((seq - 1) % tp->n_slots);
    return tp->gate_slot_start +
           (uint32_t)((seq - 1) % tp->gates_per_token) * tp->gate_slot_step;
}

/* Reap completions: send CQEs free send-queue slots, recv CQEs advance the
 * arrival watermark (UC is in-order, so gate seq recv completions arrive
 * monotonically).  Returns 0 on any completion error. */
static int tp_rdma_drain_cq(ds4_tp *tp) {
    ds4_tp_rdma *r = &tp->rdma;
    struct ibv_wc wc[16];
    int n = ibv_poll_cq(r->cq, 16, wc);
    if (n < 0) {
        fprintf(stderr, "ds4-tp: rdma poll_cq failed: %s\n", strerror(errno));
        return 0;
    }
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            fprintf(stderr, "ds4-tp: rdma completion error: %s (wr_id %llu)\n",
                    tp_wc_status_str(wc[i].status),
                    (unsigned long long)wc[i].wr_id);
            return 0;
        }
        if (wc[i].opcode & IBV_WC_RECV) {
            if (wc[i].wr_id & DS4_TP_RDMA_BLOCK_WR_TAG) {
                if (wc[i].byte_len == 0) {
                    fprintf(stderr, "ds4-tp: rdma verify-block receive of 0 bytes\n");
                    return 0;
                }
                r->block_recv_done++;
            } else if (wc[i].wr_id > r->recv_done) {
                r->recv_done = wc[i].wr_id;
            }
        } else if (r->send_outstanding > 0) {
            r->send_outstanding--;
        }
    }
    return 1;
}

/* Verify-block window helpers. */
static int tp_rdma_block_post_layer(ds4_tp *tp, uint32_t layer) {
    ds4_tp_rdma *r = &tp->rdma;
    const uint32_t rows = r->block_rows;
    const uint32_t chunks = (uint32_t)((tp->vec_bytes + DS4_TP_RDMA_MAX_MSG - 1u) /
                                       DS4_TP_RDMA_MAX_MSG);
    const uint32_t n = rows * chunks;
    if (n == 0 || n > r->recv_depth) return 0;
    struct ibv_sge *sge = r->win_sge;
    struct ibv_recv_wr *wr = r->win_rwr;
    memset(wr, 0, (size_t)n * sizeof(*wr));
    const uintptr_t base =
        (uintptr_t)(tp->slab + ds4_tp_slab_batch_in_offset(tp, layer));
    uint32_t wi = 0;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint64_t off = 0; off < tp->vec_bytes; ) {
            const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
                DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
            const int last = off + len == tp->vec_bytes;
            sge[wi].addr = base + (uintptr_t)row * tp->vec_bytes + off;
            sge[wi].length = (uint32_t)len;
            sge[wi].lkey = r->mr->lkey;
            wr[wi].wr_id = last ?
                (DS4_TP_RDMA_BLOCK_WR_TAG | ((uint64_t)layer * rows + row + 1u)) : 0;
            wr[wi].sg_list = &sge[wi];
            wr[wi].num_sge = 1;
            if (wi != 0) wr[wi - 1u].next = &wr[wi];
            wi++;
            off += len;
        }
    }
    struct ibv_recv_wr *bad = NULL;
    if (ibv_post_recv(r->qp, wr, &bad) != 0) {
        fprintf(stderr, "ds4-tp: rdma verify-block post_recv(layer %u): %s\n",
                layer, strerror(errno));
        return 0;
    }
    r->block_posted = layer + 1u;
    return 1;
}

static int tp_rdma_block_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t rows) {
    ds4_tp_rdma *r = &tp->rdma;
    if (!r->block_active || rows != r->block_rows || layer >= r->block_layers ||
        layer + 1u != r->block_posted) {
        fprintf(stderr, "ds4-tp: verify-block gate out of order (layer %u rows %u, posted %u/%u rows %u)\n",
                layer, rows, r->block_posted, r->block_layers, r->block_rows);
        return 0;
    }
    pthread_mutex_lock(&r->post_lock);
    int ok = 1;
    /* Post the next layer's receives BEFORE this layer's sends: the peer
     * can only send layer+1 after it has received this send, so its send
     * never reaches an unposted queue. */
    if (layer + 1u < r->block_layers) ok = tp_rdma_block_post_layer(tp, layer + 1u);
    const uint32_t chunks = (uint32_t)((tp->vec_bytes + DS4_TP_RDMA_MAX_MSG - 1u) /
                                       DS4_TP_RDMA_MAX_MSG);
    const uint32_t n = rows * chunks;
    if (ok) {
        struct ibv_sge *sge = r->win_sge + r->recv_depth / 2u;   /* separate half of the arrays */
        struct ibv_send_wr *wr = r->win_swr;
        if (n > r->recv_depth / 2u || n > r->send_depth) ok = 0;
        if (ok) {
            memset(wr, 0, (size_t)n * sizeof(*wr));
            const uintptr_t base =
                (uintptr_t)(tp->slab + ds4_tp_slab_batch_out_offset(tp, layer));
            uint32_t wi = 0;
            for (uint32_t row = 0; row < rows; row++) {
                for (uint64_t off = 0; off < tp->vec_bytes; ) {
                    const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
                        DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
                    sge[wi].addr = base + (uintptr_t)row * tp->vec_bytes + off;
                    sge[wi].length = (uint32_t)len;
                    sge[wi].lkey = r->mr->lkey;
                    wr[wi].wr_id = DS4_TP_RDMA_BLOCK_WR_TAG | 1u;
                    wr[wi].sg_list = &sge[wi];
                    wr[wi].num_sge = 1;
                    wr[wi].opcode = IBV_WR_SEND;
                    wr[wi].send_flags = wi + 1u == n ? IBV_SEND_SIGNALED : 0;
                    if (wi != 0) wr[wi - 1u].next = &wr[wi];
                    wi++;
                    off += len;
                }
            }
            struct ibv_send_wr *bad = NULL;
            if (ibv_post_send(r->qp, wr, &bad) != 0) {
                fprintf(stderr, "ds4-tp: rdma verify-block post_send(layer %u): %s\n",
                        layer, strerror(errno));
                ok = 0;
            } else {
                r->send_outstanding++;
            }
        }
    }
    pthread_mutex_unlock(&r->post_lock);
    const uint64_t want = (uint64_t)(layer + 1u) * rows;
    double deadline = 0.0;
    uint32_t peer_poll = 0;
    while (ok && r->block_recv_done < want) {
        ok = tp_rdma_drain_cq(tp);
        if (ok && (++peer_poll & 0x3fffu) == 0 && tp_peer_closed(tp)) {
            fprintf(stderr, "ds4-tp: peer disconnected during verify-block gate\n");
            ok = 0;
        }
        if (deadline == 0.0) deadline = tp_now_sec() + (double)tp->gate_timeout_ms / 1000.0;
        else if (tp_now_sec() > deadline) {
            fprintf(stderr, "ds4-tp: timeout waiting verify-block gate layer %u (%llu/%llu rows)\n",
                    layer, (unsigned long long)r->block_recv_done, (unsigned long long)want);
            ok = 0;
        }
    }
    return ok;
}

/* Arm the receive for gate seq: UC delivery order pairs the peer's seq'th
 * send with our seq'th posted recv, landing it in the in-slot the combine
 * kernel reads. */
static int tp_rdma_post_gate_recv(ds4_tp *tp, uint64_t seq) {
    ds4_tp_rdma *r = &tp->rdma;
    const uint32_t slot = tp_gate_slot(tp, seq);
    const uintptr_t base =
        (uintptr_t)(tp->slab + tp->in_off + (uint64_t)slot * tp->vec_bytes);
    /* Vectors above the driver's 16KB message cap ride as two chunks
     * landing contiguously in the slot. UC delivery is in-order and both
     * sides post/send strictly in seq order, so the k'th send always
     * matches the k'th recv; only the FINAL chunk carries the seq as
     * wr_id, so the arrival watermark advances when the slot is whole. */
    struct ibv_sge sge[2];
    struct ibv_recv_wr wr[2];
    memset(wr, 0, sizeof(wr));
    uint32_t count = 0;
    for (uint64_t off = 0; off < tp->vec_bytes; count++) {
        const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
            DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
        const int last = off + len == tp->vec_bytes;
        sge[count].addr = base + off;
        sge[count].length = (uint32_t)len;
        sge[count].lkey = r->mr->lkey;
        wr[count].wr_id = last ? seq : 0;
        wr[count].sg_list = &sge[count];
        wr[count].num_sge = 1;
        if (count != 0) wr[count - 1u].next = &wr[count];
        off += len;
    }
    struct ibv_recv_wr *bad = NULL;
    if (ibv_post_recv(r->qp, wr, &bad) != 0) {
        fprintf(stderr, "ds4-tp: rdma post_recv(seq %llu): %s\n",
                (unsigned long long)seq, strerror(errno));
        return 0;
    }
    return 1;
}

/* T1: two latency changes to the decode gate critical path, together behind
 * one flag.  Default off.
 *
 * M3 measured hardware half-RTT at 14.5-15.5 us for the gate's 16 KB payload
 * against ds4's observed 29.9-49.7 us exchange, so most of the gate is
 * software, not fabric.  86 gates/token makes that 1.3-3.0 ms/token at 131k
 * (the range is wide because M0 has not yet said which of the two observed
 * wire numbers was taken on a correctly-pinned shard).
 *
 * (a) Arm the next receive BEFORE waiting instead of after.  The slot posted
 *     is DS4_TP_RDMA_RECV_WINDOW seqs ahead, so it is neither the slot this
 *     gate consumes nor one already posted; posting a receive earlier is
 *     never unsafe.  It removes one verb call from between "peer data
 *     arrived" and "GPU unblocked".  Costs at most one extra seq in flight:
 *     17 x 2 chunks = 34 WRs against max_recv_wr = 64.
 *
 * (b) Stop signalling every send.  Every gate currently posts with
 *     IBV_SEND_SIGNALED, so tp_rdma_drain_cq() must chew a send CQE before
 *     reaching the recv CQE it is actually waiting for.  Completions are
 *     ordered per QP, so signalling every Nth reclaims N send-queue slots at
 *     once; at N = 16 and <=2 WRs per gate that is <=32 unsignaled WRs
 *     against max_send_wr = 256.  Nothing reads send_outstanding (it is a
 *     diagnostic counter, only ever incremented and decremented), but keep it
 *     accurate anyway.
 *
 * The one thing (b) gives up is per-send completion-time error detection.
 * ibv_post_send still reports synchronous failures, and UC on this stack is
 * treated as lossless (see the M3 note in the test plan), so the residual
 * exposure is a completion-time error on an unsignaled WR -- which the next
 * signalled completion or the gate timeout still surfaces, just later. */
#define DS4_TP_GATE_SIGNAL_EVERY 16u

/* One decode gate: ensure the receive window is armed, send our partial,
 * wait for the peer's receive completion, and advance the window. */
/* Per-gate-slot RDMA timing (DS4_TP_GATE_PROFILE): post cost and the wait for
 * the peer's partial, printed every 430 gates per slot.
 *
 * Sized and bounded by DS4_TP_GATES_PER_LAYER, not 2.  Upstream has two gates
 * per layer; this branch has three (ATTN / ROUTER / FFN), so the imported
 * `gate < 2u` guard silently dropped every FFN gate from the profile.  That is
 * the slot that matters most -- it is the only one sitting behind the
 * routed-expert shard, so it is where a straggler shows up, and a profile that
 * omits it reads as "the two gates agree" when the interesting one was never
 * counted. */
static double g_rdma_stat_post_us[DS4_TP_GATES_PER_LAYER];
static double g_rdma_stat_wait_us[DS4_TP_GATES_PER_LAYER];
static uint64_t g_rdma_stat_count[DS4_TP_GATES_PER_LAYER];
static int g_rdma_stat_enabled = -1;
static const char *tp_gate_slot_name(uint32_t gate) {
    switch (gate) {
    case DS4_TP_GATE_INDEXER: return "indexer";
    case DS4_TP_GATE_ATTN:   return "attn";
    case DS4_TP_GATE_ROUTER: return "router";
    case DS4_TP_GATE_FFN:    return "ffn";
    default:                 return "?";
    }
}

/* ---- bulk-gate window accounting (DS4_TP_BIG_GATE_DEBUG=2) ----------------
 *
 * The merge brought in a windowed bulk path whose every window opens with
 * tp_rdma_window_barrier(): a blocking 1-byte write + 1-byte read on the TCP
 * data socket before that window's sends are posted.  The pre-merge path had
 * no such barrier -- `git show 4ec9f2d:ds4_tp.c` has zero occurrences.
 *
 * It is a symmetric RENDEZVOUS, not a request/response round trip: both ranks
 * write, then both read.  Its cost is therefore "how far apart the two ranks
 * arrived" plus one socket traversal, and it must NOT be priced by multiplying
 * a count by the LOGITS-TCP 0.236-0.267 ms figure -- that number was measured
 * as peer wait and was never pure TCP latency.  Hence this: measure it.
 *
 * At the shipping 4096-row shape one chunk is 87 exchanges (45 attention + 42
 * sparse FFN) of 64 MiB each; a depth-256 window is 4 MiB, so ~16 windows per
 * exchange and ~1,392 barriers per chunk.  That is the number worth measuring
 * directly, and the reason a depth-64 arm (~5,568) is a usable causal probe.
 *
 * SIX phases, each timed per window with local variables.  There is no
 * residual: an earlier version computed "poll" as whole-minus-named against a
 * cumulative running total, which folded WR construction, send posting,
 * transfer and completion handling into one bucket and then described it as
 * "the peer being slow".  Every phase below is measured, and setup is the only
 * thing left over.
 *
 *   setup   WR/SGE construction for the window
 *   copyin  staging memcpy host->slab (one contiguous copy, staged path only)
 *   rpost   ibv_post_recv chains
 *   barrier tp_rdma_window_barrier -- the rendezvous under test
 *   spost   ibv_post_send sub-batches
 *   cwait   completion polling until the window's recvs and signalled sends land
 *   copyout staging memcpy slab->host (staged path only)
 */
typedef struct {
    uint64_t windows;
    uint64_t bytes;
    double setup_us, copyin_us, rpost_us, barrier_us, spost_us, cwait_us, copyout_us;
} ds4_tp_bulk_acct;

static ds4_tp_bulk_acct g_bulk;
static int g_bulk_dbg = -1;   /* 0 off, 1 gate lines, 2 accounting, 3 both */

/* Level 2 is the measurement mode and is deliberately QUIET: the per-gate
 * fprintf runs on the service thread inside the gate, and at 87 exchanges a
 * chunk it perturbs exactly what is being measured.  Level 3 asks for both. */
static int tp_bulk_dbg_level(void) {
    if (g_bulk_dbg < 0) {
        const char *e = getenv("DS4_TP_BIG_GATE_DEBUG");
        if (!e || !e[0]) g_bulk_dbg = 0;
        else if (e[0] == '3') g_bulk_dbg = 3;
        else if (e[0] == '2') g_bulk_dbg = 2;
        else g_bulk_dbg = 1;
    }
    return g_bulk_dbg;
}
static int tp_bulk_acct_on(void)  { const int l = tp_bulk_dbg_level(); return l >= 2; }
static int tp_bulk_gateline_on(void) { const int l = tp_bulk_dbg_level(); return l == 1 || l == 3; }

void ds4_tp_bulk_window_report(const char *when) {
    if (!g_bulk.windows) return;
    const double n = (double)g_bulk.windows;
    const double total = g_bulk.setup_us + g_bulk.copyin_us + g_bulk.rpost_us +
                         g_bulk.barrier_us + g_bulk.spost_us + g_bulk.cwait_us +
                         g_bulk.copyout_us;
    fprintf(stderr,
            "ds4-tp: bulk acct [%s] windows=%llu bytes=%llu (%.2f MiB) "
            "barrier=%.1fms/%.0fus_win/%.1f%% setup=%.1f copyin=%.1f "
            "rpost=%.1f spost=%.1f cwait=%.1f copyout=%.1f total=%.1fms\n",
            when ? when : "run", (unsigned long long)g_bulk.windows,
            (unsigned long long)g_bulk.bytes, (double)g_bulk.bytes / 1048576.0,
            g_bulk.barrier_us / 1000.0, g_bulk.barrier_us / n,
            100.0 * g_bulk.barrier_us / (total > 0.0 ? total : 1.0),
            g_bulk.setup_us / 1000.0, g_bulk.copyin_us / 1000.0,
            g_bulk.rpost_us / 1000.0, g_bulk.spost_us / 1000.0,
            g_bulk.cwait_us / 1000.0, g_bulk.copyout_us / 1000.0,
            total / 1000.0);
}

static int tp_rdma_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t gate, uint64_t seq) {
    ds4_tp_rdma *r = &tp->rdma;
    const uint32_t slot = layer * DS4_TP_GATES_PER_LAYER + gate;
    if (g_rdma_stat_enabled < 0) g_rdma_stat_enabled = getenv("DS4_TP_GATE_PROFILE") != NULL;
    const double st0 = g_rdma_stat_enabled ? tp_now_sec() : 0.0;
    double st1 = 0.0;
    if (getenv("DS4_TP_GATE_TRACE")) {
        fprintf(stderr, "ds4-tp: gate trace l=%u g=%u seq=%llu want_slot=%u\n",
                layer, gate, (unsigned long long)seq, tp_gate_slot(tp, seq));
    }
    if (slot != tp_gate_slot(tp, seq)) {
        /* Say what was expected, not just what arrived.  This mismatch means
         * the graph and ds4_engine_tp_gate_schedule() disagree about which
         * gates fire, and it has now cost two rig cycles (S6a, S6c) because the
         * old message named only the offending gate.  Naming the expected slot
         * and the per-token count identifies the disagreement directly: a gate
         * short means the graph skipped one the schedule mask includes, a gate
         * long means the reverse. */
        const uint32_t want = tp_gate_slot(tp, seq);
        fprintf(stderr,
                "ds4-tp: gate order broke: graph fired layer %u gate %u "
                "(slot %u), schedule expected slot %u (layer %u gate %u) "
                "at seq %llu; %u gates/token\n",
                layer, gate, slot,
                want, want / DS4_TP_GATES_PER_LAYER,
                want % DS4_TP_GATES_PER_LAYER,
                (unsigned long long)seq, tp->gates_per_token);
        return 0;
    }
    const uintptr_t send_base =
        (uintptr_t)(tp->slab + tp->out_off + (uint64_t)slot * tp->vec_bytes);
    pthread_mutex_lock(&r->post_lock);
    int ok = 1;
    /* Row gates keep the short in-gate budget even on the gate that arms the
     * window.  They have no header rendezvous, so this recv wait IS their
     * arrival barrier -- but the skew it absorbs is small: a decode gate is
     * only ever entered from lockstep, either from the previous gate or from
     * the bulk gate that drained the window.  Widening it to the meet budget
     * would also catch every row gate that follows a speculative verify batch,
     * which is a hot path with no skew to absorb and real watchdog margin to
     * lose.  If a first-decode-gate timeout is ever observed, measure it with
     * DS4_TP_GATE_TIMEOUT_MS before changing the default. */
    if (!r->recv_window_active) {
        for (uint64_t s = seq; ok && s < seq + DS4_TP_RDMA_RECV_WINDOW; s++)
            ok = tp_rdma_post_gate_recv(tp, s);
        if (ok) r->recv_window_active = true;
    }
    struct ibv_sge send_sge[2];
    struct ibv_send_wr send_wr[2];
    memset(send_wr, 0, sizeof(send_wr));
    uint32_t send_count = 0;
    for (uint64_t off = 0; ok && off < tp->vec_bytes; send_count++) {
        const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
            DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
        send_sge[send_count].addr = send_base + off;
        send_sge[send_count].length = (uint32_t)len;
        send_sge[send_count].lkey = r->mr->lkey;
        send_wr[send_count].wr_id = seq;
        send_wr[send_count].sg_list = &send_sge[send_count];
        send_wr[send_count].num_sge = 1;
        send_wr[send_count].opcode = IBV_WR_SEND;
        if (send_count != 0) send_wr[send_count - 1u].next = &send_wr[send_count];
        off += len;
    }
    if (ok) {
        send_wr[send_count - 1u].send_flags = IBV_SEND_SIGNALED;
        struct ibv_send_wr *bad = NULL;
        ok = ibv_post_send(r->qp, send_wr, &bad) == 0;
        if (!ok) {
            fprintf(stderr, "ds4-tp: rdma post_send: %s\n", strerror(errno));
        } else {
            r->send_outstanding++;
        }
    }
    if (g_rdma_stat_enabled) st1 = tp_now_sec();

    double deadline = 0.0;
    uint32_t peer_poll = 0;
    while (ok && r->recv_done < seq) {
        ok = tp_rdma_drain_cq(tp);
        if (ok && (++peer_poll & 0x3fffu) == 0 && tp_peer_closed(tp)) {
            fprintf(stderr, "ds4-tp: peer disconnected during RDMA gate\n");
            ok = 0;
        }
        if (deadline == 0.0) {
            deadline = tp_gate_deadline(tp);
        }
        else if (tp_now_sec() > deadline) {
            fprintf(stderr, "ds4-tp: timeout waiting gate seq %llu (recv_done %llu)\n",
                    (unsigned long long)seq, (unsigned long long)r->recv_done);
            ok = 0;
        }
    }
    if (g_rdma_stat_enabled && ok && gate < DS4_TP_GATES_PER_LAYER) {
        const double st2 = tp_now_sec();
        g_rdma_stat_post_us[gate] += (st1 - st0) * 1e6;
        g_rdma_stat_wait_us[gate] += (st2 - st1) * 1e6;
        if (++g_rdma_stat_count[gate] % 430 == 0) {
            fprintf(stderr,
                    "ds4-tp: rdma gate %u (%s): post %.1f us, peer wait %.1f us (n=%llu)\n",
                    gate, tp_gate_slot_name(gate),
                    g_rdma_stat_post_us[gate] / (double)g_rdma_stat_count[gate],
                    g_rdma_stat_wait_us[gate] / (double)g_rdma_stat_count[gate],
                    (unsigned long long)g_rdma_stat_count[gate]);
        }
    }
    if (ok) ok = tp_rdma_post_gate_recv(tp, seq + DS4_TP_RDMA_RECV_WINDOW);
    if (ok) r->last_gate_seq = seq;
    pthread_mutex_unlock(&r->post_lock);
    return ok;
}

static int tp_rdma_big_gate_capable(const ds4_tp *tp) {
    const uint64_t stage_bytes =
        (uint64_t)DS4_TP_RDMA_BULK_SLOTS * DS4_TP_RDMA_MAX_MSG;
    const uint64_t batch_region_bytes =
        (uint64_t)tp->n_layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
    return tp->rdma.qp && tp->rdma.mr && batch_region_bytes >= stage_bytes;
}

/* Decode keeps a lookahead window of receives on the latency QP. Before a
 * later prompt can reuse that QP for bulk rows, consume those receives with
 * dummy sends on both ranks. The TCP big-gate header exchange is the barrier
 * that guarantees both sides have reached this transition. */
static int tp_rdma_drain_decode_window(ds4_tp *tp) {
    ds4_tp_rdma *r = &tp->rdma;
    if (!r->recv_window_active) return 1;

    const uint32_t chunks_per_gate =
        (uint32_t)((tp->vec_bytes + DS4_TP_RDMA_MAX_MSG - 1u) /
                   DS4_TP_RDMA_MAX_MSG);
    const uint32_t nwr = DS4_TP_RDMA_RECV_WINDOW * chunks_per_gate;
    struct ibv_sge sge[DS4_TP_RDMA_RECV_WINDOW * 2u];
    struct ibv_send_wr wr[DS4_TP_RDMA_RECV_WINDOW * 2u];
    memset(wr, 0, sizeof(wr));
    uint8_t *scratch = tp->slab + tp->batch_out_off;
    uint32_t wi = 0;
    for (uint32_t gate = 0; gate < DS4_TP_RDMA_RECV_WINDOW; gate++) {
        for (uint64_t off = 0; off < tp->vec_bytes; ) {
            const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
                DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
            sge[wi] = (struct ibv_sge) {
                .addr = (uintptr_t)(scratch + off),
                .length = (uint32_t)len,
                .lkey = r->mr->lkey,
            };
            wr[wi].wr_id = DS4_TP_RDMA_BULK_WR_TAG | ((uint64_t)wi + 1u);
            wr[wi].sg_list = &sge[wi];
            wr[wi].num_sge = 1;
            wr[wi].opcode = IBV_WR_SEND;
            wr[wi].send_flags = wi + 1u == nwr ? IBV_SEND_SIGNALED : 0;
            if (wi > 0) wr[wi - 1u].next = &wr[wi];
            wi++;
            off += len;
        }
    }

    pthread_mutex_lock(&r->post_lock);
    struct ibv_send_wr *bad = NULL;
    if (ibv_post_send(r->qp, wr, &bad) != 0) {
        fprintf(stderr, "ds4-tp: rdma receive-window drain post failed: %s\n",
                strerror(errno));
        pthread_mutex_unlock(&r->post_lock);
        return 0;
    }

    uint32_t recv_done = 0;
    int send_done = 0;
    const double deadline =
        tp_now_sec() + (double)tp->gate_timeout_ms / 1000.0;
    uint32_t peer_poll = 0;
    while (recv_done < nwr || !send_done) {
        struct ibv_wc wc[DS4_TP_RDMA_RECV_WINDOW * 2u + 1u];
        int n = ibv_poll_cq(r->cq,
                           (int)(DS4_TP_RDMA_RECV_WINDOW * 2u + 1u), wc);
        if (n < 0) {
            fprintf(stderr,
                    "ds4-tp: rdma poll_cq failed draining the receive window: %s\n",
                    strerror(errno));
            pthread_mutex_unlock(&r->post_lock);
            return 0;
        }
        for (int i = 0; i < n; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                fprintf(stderr, "ds4-tp: rdma receive-window drain: %s\n",
                        tp_wc_status_str(wc[i].status));
                pthread_mutex_unlock(&r->post_lock);
                return 0;
            }
            if (wc[i].opcode & IBV_WC_RECV) {
                recv_done++;
            } else if (wc[i].wr_id & DS4_TP_RDMA_BULK_WR_TAG) {
                send_done = 1;
            } else if (r->send_outstanding > 0) {
                r->send_outstanding--;
            }
        }
        if ((peer_poll++ & 0x3fffu) == 0 && tp_peer_closed(tp)) {
            fprintf(stderr,
                    "ds4-tp: peer disconnected while draining RDMA receives\n");
            pthread_mutex_unlock(&r->post_lock);
            return 0;
        }
        if (tp_now_sec() > deadline) {
            fprintf(stderr,
                    "ds4-tp: timeout draining RDMA receive window (%u/%u)\n",
                    recv_done, nwr);
            pthread_mutex_unlock(&r->post_lock);
            return 0;
        }
    }
    r->recv_done = r->last_gate_seq;
    r->recv_window_active = false;
    pthread_mutex_unlock(&r->post_lock);
    return 1;
}

/* Large prefill row swaps share the latency QP.  No future decode receives
 * are queued, so each round can post its 1 MiB receive window before sending
 * the matching 16 KiB messages.  Verify scratch provides already-registered
 * staging memory and is idle during normal prefill. */
/* One-byte barrier on the batch socket: both sides have their receives
 * posted for the coming window before either posts a send (a UC send with
 * no receive posted kills that direction on this driver). */
static int tp_rdma_window_barrier(ds4_tp *tp, uint8_t tag) {
    uint8_t peer = 0;
    if (!tp_write_full(tp->data_fd, &tag, 1) || !tp_read_full(tp->data_fd, &peer, 1)) return 0;
    return peer == tag;
}

/* Big-gate exchange (prefill partials): windows of up to recv_depth
 * messages; every window posts its receives, crosses the barrier, then
 * streams sends in send-depth sub-batches. Metal-owned payloads are staged
 * through the registered slab because Apple's verbs provider rejects direct
 * registration of these buffers. */
static int tp_rdma_big_gate_exchange(ds4_tp *tp,
                                     const void *out,
                                     void *in,
                                     uint64_t bytes) {
    ds4_tp_rdma *r = &tp->rdma;
    if (!tp_rdma_big_gate_capable(tp) || r->recv_window_active) {
        /* Both conditions are caller contracts: the capability is checked
         * before this path is selected, and the decode window is drained by
         * the header exchange that precedes it.  Neither is recoverable here,
         * but silently failing the gate reads as a wire fault. */
        fprintf(stderr,
                "ds4-tp: bulk rdma gate unavailable (capable=%d, decode window %s)\n",
                tp_rdma_big_gate_capable(tp) ? 1 : 0,
                r->recv_window_active ? "still armed" : "clear");
        return 0;
    }

    /* Payloads already inside the registered slab (verify batches) can ride
     * directly. Ordinary prefill tensors use the idle verify regions as
     * registered staging because their standalone MTLBuffers are not in the
     * NIC memory region. */
    const uintptr_t slab_lo = (uintptr_t)tp->slab;
    const uintptr_t slab_hi = slab_lo + tp->slab_bytes;
    const uintptr_t out_lo = (uintptr_t)out;
    const uintptr_t in_lo = (uintptr_t)in;
    const bool in_slab =
        out_lo >= slab_lo && out_lo <= slab_hi && bytes <= slab_hi - out_lo &&
        in_lo >= slab_lo && in_lo <= slab_hi && bytes <= slab_hi - in_lo;
    struct ibv_mr *out_mr = r->mr;
    struct ibv_mr *in_mr = r->mr;
    const bool direct = in_slab;
    {
        /* Say which path this is, once.  `direct` versus staged is a 2x-volume
         * difference in CPU copies and is otherwise invisible from outside. */
        static int announced;
        if (!announced) {
            announced = 1;
            fprintf(stderr,
                    "ds4-tp: bulk gate path is %s (%.1f MiB payload)\n",
                    direct ? "DIRECT, no staging" : "STAGED through the slab",
                    (double)bytes / 1048576.0);
        }
    }
    /* Payloads already inside the registered slab (verify batches) can ride
     * directly. Ordinary prefill tensors use the idle verify regions as
     * registered staging because their standalone MTLBuffers are not in the
     * NIC memory region. */
    uint8_t *stage_send = tp->slab + tp->batch_out_off;
    uint8_t *stage_recv = tp->slab + tp->batch_in_off;
    const uint64_t stage_bytes = (uint64_t)tp->n_layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
    /* WINDOW DEPTH IS IN MESSAGES.  THE QUEUE IS IN FRAMES.  Getting this
     * wrong wedges both ranks.
     *
     * TN3205: "Queues are sized in units of 4 KB frames... two 4 KB sends and
     * a single 8 KB send would both take up 2 queue spaces."  A 16 KiB message
     * is therefore FOUR receive slots, not one, and the granted max_recv_wr of
     * 1024 buys 256 outstanding messages.
     *
     * That is where the bare `if (depth > 256u) depth = 256u;` came from -- it
     * was exactly this conversion, written as a constant with nothing saying
     * so, which reads as an arbitrary cap with headroom above it.  It is not.
     * Over-posting the ring does not error: per APPLE-RDMA.md 3.1 completions
     * simply never arrive and both ends spin forever.
     *
     * Derived from the grant now, so it cannot drift from the QP or from
     * DS4_TP_RDMA_MAX_MSG. */
    const uint32_t frames_per_msg =
        (DS4_TP_RDMA_MAX_MSG + DS4_TP_RDMA_FRAME_BYTES - 1u) / DS4_TP_RDMA_FRAME_BYTES;
    const uint32_t recv_frames = r->recv_depth ? r->recv_depth : 64u;
    const uint32_t depth_granted = recv_frames / frames_per_msg;
    uint32_t depth = depth_granted ? depth_granted : 1u;
    /* DS4_TP_RDMA_WINDOW_DEPTH exists to test the barrier hypothesis in the
     * only direction that is safe.  Barrier count is bytes/(depth x 16 KiB),
     * so LOWERING depth multiplies barriers: if the barrier owns the
     * regression, halving depth must roughly double the penalty, and that is a
     * causal test no amount of correlation gives.  Raising it past what the QP
     * was created with would overrun the receive queue, so it is clamped -- the
     * knob can make things worse on purpose, never silently worse by accident. */
    const uint32_t depth_base = depth;
    {
        static int win_override = -2;
        if (win_override == -2) {
            const char *e = getenv("DS4_TP_RDMA_WINDOW_DEPTH");
            win_override = (e && e[0]) ? atoi(e) : -1;
        }
        if (win_override > 0) {
            /* Both directions now.  Down multiplies barriers (the causal test
             * that can only make things worse, so it is always safe).  Up
             * divides them, and is clamped to the depth the QP actually
             * granted -- posting more receive WRs than the queue holds would
             * overrun it, and on a UC QP that is a silent drop, not an error. */
            uint32_t want = (uint32_t)win_override;
            if (want > depth_granted) {
                static int warned;
                if (!warned) {
                    warned = 1;
                    fprintf(stderr,
                            "ds4-tp: window depth %u messages exceeds what the "
                            "granted queue holds (%u frames / %u per message = "
                            "%u messages); clamping.  Over-posting this ring does "
                            "not error -- completions stop arriving and both "
                            "ranks spin.  Raise DS4_TP_RDMA_RECV_FRAMES instead.\n",
                            want, recv_frames, frames_per_msg, depth_granted);
                }
                want = depth_granted;
            }
            depth = want;
        }
    }
    if (!direct) {
        const uint32_t stage_slots = (uint32_t)(stage_bytes / DS4_TP_RDMA_MAX_MSG);
        if (depth > stage_slots) depth = stage_slots;
        if (depth == 0u) return 0;
        out_mr = in_mr = r->mr;
    }
    /* Announce the EFFECTIVE depth, once, after every clamp -- the granted
     * receive depth, the 256 cap, the override and the staging-slot clamp.
     * The previous announce printed the requested value before the staging
     * clamp could lower it, so an arm could report "forced to 64" while
     * actually running at 32 and the window count would not match the plan. */
    {
        static int announced_depth;
        if (!announced_depth) {
            announced_depth = 1;
            fprintf(stderr,
                    "ds4-tp: bulk window depth %u -> %u messages "
                    "(frames asked %u granted %u / %u per msg = %u max, "
                    "%.2f MiB/window, %s)\n",
                    depth_base, depth, tp_rdma_want_frames(), recv_frames,
                    frames_per_msg, depth_granted,
                    (double)depth * DS4_TP_RDMA_MAX_MSG / 1048576.0,
                    depth == depth_base ? "no override" : "override or clamp applied");
        }
    }
    if (!r->win_sge) {
        const uint32_t cap = r->recv_depth > r->send_depth ? r->recv_depth : r->send_depth;
        r->win_sge = calloc(2u * cap, sizeof(*r->win_sge));
        r->win_rwr = calloc(cap, sizeof(*r->win_rwr));
        r->win_swr = calloc(cap, sizeof(*r->win_swr));
        if (!r->win_sge || !r->win_rwr || !r->win_swr) return 0;
    }
    const uint32_t send_depth = r->send_depth ? r->send_depth : 256u;
    const int acct = tp_bulk_acct_on();
    /* One clock read per phase per WINDOW.  Timing each 16 KiB chunk would put
     * thousands of gettime calls inside the thing being measured. */
#define BULK_T(dst, expr) do { \
        if (acct) { const double t_0 = tp_now_sec(); expr; (dst) += (tp_now_sec() - t_0) * 1e6; } \
        else { expr; } \
    } while (0)
    uint64_t off = 0;
    uint8_t tag = 1;
    while (off < bytes) {
        double t_setup = 0, t_copyin = 0, t_rpost = 0, t_barrier = 0;
        double t_spost = 0, t_cwait = 0, t_copyout = 0;
        const uint64_t remaining = bytes - off;
        uint32_t chunks = (uint32_t)((remaining + DS4_TP_RDMA_MAX_MSG - 1u) / DS4_TP_RDMA_MAX_MSG);
        if (chunks > depth) chunks = depth;
        /* Known before the WR loop, which is what lets the staging copy be one
         * contiguous memcpy instead of `chunks` of them. */
        const uint64_t win_span = (uint64_t)chunks * DS4_TP_RDMA_MAX_MSG;
        const uint64_t win_total = remaining < win_span ? remaining : win_span;
        if (!direct) {
            BULK_T(t_copyin,
                   memcpy(stage_send, (const uint8_t *)out + off, (size_t)win_total));
        }
        const double s0 = acct ? tp_now_sec() : 0.0;
        uint64_t win_bytes = 0;
        for (uint32_t i = 0; i < chunks; i++) {
            const uint64_t left = remaining - win_bytes;
            const uint32_t len = (uint32_t)(left > DS4_TP_RDMA_MAX_MSG ? DS4_TP_RDMA_MAX_MSG : left);
            const uint64_t coff = win_bytes;
            r->win_sge[i] = (struct ibv_sge) {
                .addr = direct ? in_lo + off + coff : (uintptr_t)(stage_recv + coff),
                .length = len,
                .lkey = in_mr->lkey,
            };
            memset(&r->win_rwr[i], 0, sizeof(r->win_rwr[i]));
            r->win_rwr[i].wr_id = DS4_TP_RDMA_BULK_WR_TAG | ((uint64_t)i + 1u);
            r->win_rwr[i].sg_list = &r->win_sge[i];
            r->win_rwr[i].num_sge = 1;
            r->win_rwr[i].next = i + 1u < chunks ? &r->win_rwr[i + 1u] : NULL;
            r->win_sge[depth + i] = (struct ibv_sge) {
                .addr = direct ? out_lo + off + coff : (uintptr_t)(stage_send + coff),
                .length = len,
                .lkey = out_mr->lkey,
            };
            win_bytes += len;
        }
        /* Post in chains of 64 (the chain length the driver is known to
         * accept); the work requests are already linked, so cut the links at
         * chain ends. */
        if (acct) t_setup = (tp_now_sec() - s0) * 1e6;
        const double rp0 = acct ? tp_now_sec() : 0.0;
        for (uint32_t c0 = 0; c0 < chunks; c0 += 64u) {
            const uint32_t c1 = c0 + 64u < chunks ? c0 + 64u : chunks;
            r->win_rwr[c1 - 1u].next = NULL;
            struct ibv_recv_wr *bad_recv = NULL;
            if (ibv_post_recv(r->qp, &r->win_rwr[c0], &bad_recv) != 0) {
                fprintf(stderr, "ds4-tp: big gate post_recv(%u of %u): %s\n", c0, chunks, strerror(errno));
                return 0;
            }
        }
        atomic_thread_fence(memory_order_release);
        const double b0 = acct ? tp_now_sec() : 0.0;
        if (acct) t_rpost = (b0 - rp0) * 1e6;
        if (!tp_rdma_window_barrier(tp, tag)) {
            fprintf(stderr, "ds4-tp: big gate window barrier failed\n");
            return 0;
        }
        if (acct) t_barrier = (tp_now_sec() - b0) * 1e6;
        tag++;
        /* Sends in sub-batches bounded by the send depth; the last of each
         * sub-batch is signaled. */
        uint32_t sent = 0, send_done = 0, recv_done = 0, signaled = 0;
        /* How many MESSAGES the send queue can hold, from the frame budget.
         *
         * This provider completes EVERY send work request, whether or not
         * IBV_SEND_SIGNALED is set -- the same thin-shim behaviour that makes
         * it ignore wr->opcode and imm_data.  The old gate compared a count of
         * POSTS against a count of COMPLETIONS:
         *
         *     while (... send_done < signaled ...)
         *         if (sent < chunks && (signaled - send_done) < 4u)
         *
         * so send_done raced past signaled (WINDEP2 logged 1023 against 16),
         * the unsigned subtraction wrapped to ~4e9, and the gate stopped
         * gating.  At the shipping 256-message window that was 4 posts and the
         * race rarely opened; at 1023 messages it is 16 posts and the loop
         * fires everything at the queue at once.
         *
         * Both counters are in messages now, and sent >= send_done always, so
         * the subtraction cannot wrap. */
        /* Start from the frame budget, then LEARN the real capacity.
         *
         * The arithmetic says 4095 frames / 4 per message = 1023 messages, and
         * 1023 x 4 = 4092 of 4095 -- three frames of slack.  WINDEP2 shows the
         * true limit is lower: the worker took 1009 messages and then refused,
         * so something else occupies ~60 frames (a provider-reserved
         * descriptor, or accounting slack this side cannot see).  Guessing a
         * margin would be a magic number replacing a magic number, so instead
         * the first EAGAIN teaches us the capacity and every later window
         * respects it.  Static: one QP per process, and the answer does not
         * change within a run. */
        static uint32_t g_send_msgs_cap;
        const uint32_t send_msgs_from_frames =
            send_depth / frames_per_msg ? send_depth / frames_per_msg : 1u;
        if (!g_send_msgs_cap || g_send_msgs_cap > send_msgs_from_frames)
            g_send_msgs_cap = send_msgs_from_frames;
        /* Read live below, not snapshotted: an EAGAIN mid-window lowers the cap
         * and the remainder of THIS window should already respect it. */
#define SEND_MSGS_MAX (g_send_msgs_cap)
        const double deadline = tp_now_sec() + (double)tp->gate_timeout_ms / 1000.0 + 2.0;
        uint32_t peer_poll = 0;
        while (recv_done < chunks || send_done < sent) {
            const uint32_t in_flight = sent - send_done;
            if (sent < chunks && in_flight < SEND_MSGS_MAX) {
                const double ws0 = acct ? tp_now_sec() : 0.0;
                uint32_t n = chunks - sent;
                if (n > 64u) n = 64u;   /* chain length the driver accepts */
                /* Never offer more than the queue can still hold.  The old
                 * bound was `send_depth / 4u`, which is the frames-per-message
                 * divisor written as a bare 4 -- correct only while a message
                 * is 16 KiB, and it bounded the BATCH rather than the
                 * OUTSTANDING total, so it never prevented an overflow. */
                const uint32_t room = SEND_MSGS_MAX - in_flight;
                if (n > room) n = room;
                if (n == 0u) n = 1u;
                for (uint32_t i = 0; i < n; i++) {
                    struct ibv_send_wr *w = &r->win_swr[i];
                    memset(w, 0, sizeof(*w));
                    w->wr_id = DS4_TP_RDMA_BULK_WR_TAG | ((uint64_t)(sent + i) + 1u);
                    w->sg_list = &r->win_sge[depth + sent + i];
                    w->num_sge = 1;
                    w->opcode = IBV_WR_SEND;
                    /* SIGNAL EVERY REQUEST.  The flow control below compares
                     * completions against sends, so it needs one completion per
                     * send.  This provider already produces that -- it ignores
                     * send_flags exactly as it ignores wr->opcode -- but relying
                     * on that is a hang waiting to happen: if the flag were ever
                     * honoured, `send_done < sent` would never clear and the
                     * loop would spin to its deadline.  Asking for what the code
                     * depends on costs nothing here and cannot be wrong. */
                    w->send_flags = IBV_SEND_SIGNALED;
                    w->next = i + 1u < n ? &r->win_swr[i + 1u] : NULL;
                }
                if (acct) t_setup += (tp_now_sec() - ws0) * 1e6;
                struct ibv_send_wr *bad_send = NULL;
                int psrc = 0;
                BULK_T(t_spost, psrc = ibv_post_send(r->qp, r->win_swr, &bad_send));
                if (psrc != 0) {
                    /* A failed post_send is PARTIAL, not atomic: *bad_wr names
                     * the first request the queue would not take, and
                     * everything before it was accepted.  Discarding the whole
                     * batch loses those, and the peer then waits forever for
                     * chunks that were already on the wire -- which is exactly
                     * how WINDEP2 desynced (worker EAGAIN on post_send(63) with
                     * 49 accepted; coordinator stalled at 1009/1023 recvs, and
                     * 1023 - 1009 = 14 = 63 - 49).
                     *
                     * EAGAIN means "queue full, try again", so credit what was
                     * taken and let the loop drain completions and retry.  Only
                     * a genuine error is fatal. */
                    uint32_t accepted = 0;
                    if (bad_send >= r->win_swr && bad_send < r->win_swr + n)
                        accepted = (uint32_t)(bad_send - r->win_swr);
                    /* The RETURN CODE is the error: verbs returns errno by
                     * value and is not required to set the global.  Reading
                     * errno can pick up whatever the last unrelated syscall
                     * left, and a misclassified EAGAIN is a discarded partial
                     * post -- the desync this path exists to prevent. */
                    if (psrc == EAGAIN || psrc == ENOMEM) {
                        sent += accepted;
                        if (accepted) signaled++;
                        /* The queue told us its real size.  Back the cap off
                         * below what was in flight when it refused, so the next
                         * window paces itself instead of rediscovering this. */
                        const uint32_t observed = sent - send_done;
                        const uint32_t learned = observed > 8u ? observed - 8u : 1u;
                        if (learned < g_send_msgs_cap) {
                            fprintf(stderr,
                                    "ds4-tp: big gate send queue refused at %u "
                                    "messages in flight (frame budget implied "
                                    "%u); capping at %u for the rest of the run\n",
                                    observed, send_msgs_from_frames, learned);
                            g_send_msgs_cap = learned;
                        }
                        goto poll_completions;
                    }
                    fprintf(stderr,
                            "ds4-tp: big gate post_send(%u): %s (%u of %u accepted, "
                            "%u/%u sent, %u in flight of %u)\n",
                            n, strerror(psrc), accepted, n, sent, chunks,
                            in_flight, SEND_MSGS_MAX);
                    return 0;
                }
                {
                    /* Confirm the batched send loop upstream added actually
                     * engaged.  It replaced "post every chunk up front" with
                     * "post in sub-batches bounded by the send depth, signal
                     * the last of each", and the difference is invisible from
                     * throughput alone -- the old shape would simply have been
                     * slower, not wrong.  One line, once. */
                    static int announced_batch;
                    if (!announced_batch) {
                        announced_batch = 1;
                        fprintf(stderr,
                                "ds4-tp: bulk gate send BATCHED (%u chunks this "
                                "round, up to %u per post, send queue %u frames "
                                "= %u messages)\n",
                                chunks, n, send_depth, SEND_MSGS_MAX);
                    }
                }
                sent += n;
                signaled++;
            }
        poll_completions:;
            struct ibv_wc wc[64];
            int nwc = 0;
            BULK_T(t_cwait, nwc = ibv_poll_cq(r->cq, 64, wc));
            if (nwc < 0) {
                fprintf(stderr,
                        "ds4-tp: rdma poll_cq failed in a bulk round: %s\n",
                        strerror(errno));
                return 0;
            }
            for (int i = 0; i < nwc; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "ds4-tp: big gate completion error: %s\n",
                            tp_wc_status_str(wc[i].status));
                    return 0;
                }
                if ((wc[i].wr_id & DS4_TP_RDMA_BULK_WR_TAG) == 0) {
                    if (wc[i].opcode & IBV_WC_RECV) {
                        if (wc[i].wr_id > r->recv_done) r->recv_done = wc[i].wr_id;
                    } else if (r->send_outstanding > 0) {
                        r->send_outstanding--;
                    }
                    continue;
                }
                if (wc[i].opcode & IBV_WC_RECV) {
                    const uint64_t idx = (wc[i].wr_id & ~DS4_TP_RDMA_BULK_WR_TAG) - 1u;
                    const uint32_t want = idx + 1u < chunks || win_bytes % DS4_TP_RDMA_MAX_MSG == 0u
                        ? (uint32_t)DS4_TP_RDMA_MAX_MSG : (uint32_t)(win_bytes % DS4_TP_RDMA_MAX_MSG);
                    if (idx >= chunks || wc[i].byte_len != want) {
                        fprintf(stderr, "ds4-tp: big gate chunk %llu received %u bytes, expected %u\n",
                                (unsigned long long)idx, wc[i].byte_len, want);
                        return 0;
                    }
                    recv_done++;
                } else {
                    send_done++;
                }
            }
            if (nwc == 0 && (peer_poll++ & 0x3fffu) == 0 && tp_peer_closed(tp)) {
                fprintf(stderr, "ds4-tp: peer disconnected during big gate\n");
                return 0;
            }
            if (nwc == 0 && tp_now_sec() > deadline) {
                fprintf(stderr,
                        "ds4-tp: timeout in big gate window (%u/%u recvs, %u/%u send "
                        "completions, %u posts, queue cap %u messages)\n",
                        recv_done, chunks, send_done, sent, signaled, SEND_MSGS_MAX);
                return 0;
            }
        }
        atomic_thread_fence(memory_order_acquire);
        if (!direct) {
            BULK_T(t_copyout,
                   memcpy((uint8_t *)in + off, stage_recv, (size_t)win_bytes));
        }
#undef SEND_MSGS_MAX
        off += win_bytes;
        if (acct) {
            /* Committed only HERE, after the window completed.  A window that
             * returned early on an error never reaches this line, so every
             * counted barrier is a barrier belonging to a window that finished
             * -- which is what makes "windows" comparable across arms. */
            g_bulk.windows++;
            g_bulk.bytes      += win_bytes;
            g_bulk.setup_us   += t_setup;
            g_bulk.copyin_us  += t_copyin;
            g_bulk.rpost_us   += t_rpost;
            g_bulk.barrier_us += t_barrier;
            g_bulk.spost_us   += t_spost;
            g_bulk.cwait_us   += t_cwait;
            g_bulk.copyout_us += t_copyout;
            /* One safety report per ~47 chunks.  The spec is right that final
             * aggregates suffice, but a rank that gets pkill -9'd by a harness
             * timeout never reaches teardown, and a run that produced no
             * counters at all is indistinguishable from a run where the
             * accounting was never enabled. */
            if ((g_bulk.windows & 0xffffu) == 0u) ds4_tp_bulk_window_report("running");
        }
    }
    return 1;
#undef BULK_T
}

static void tp_rdma_close(ds4_tp *tp) {
    ds4_tp_rdma *r = &tp->rdma;
    ds4_tp_bulk_window_report("final");
    free(r->win_sge); free(r->win_rwr); free(r->win_swr);
    r->win_sge = NULL; r->win_rwr = NULL; r->win_swr = NULL;
    if (r->qp) r->api.destroy_qp(r->qp);
    if (r->mr) r->api.dereg_mr(r->mr);
    if (r->cq) r->api.destroy_cq(r->cq);
    if (r->pd) r->api.dealloc_pd(r->pd);
    if (r->ctx) r->api.close_device(r->ctx);
    r->qp = NULL; r->mr = NULL; r->cq = NULL; r->pd = NULL; r->ctx = NULL;
}

#endif /* DS4_TP_HAVE_VERBS */

/* ------------------------------------------------------------------------
 * Bring-up.
 * --------------------------------------------------------------------- */

static int tp_hello_exchange(ds4_tp *tp, const ds4_tp_identity *id, int rdma_ok,
                             char *err, size_t errlen) {
    ds4_tp_hello_fixed mine = {
        .magic = DS4_TP_MAGIC,
        .version = DS4_TP_PROTOCOL_VERSION,
        .role = (uint32_t)tp->opt.role,
        .rdma_ok = (uint32_t)rdma_ok,
        .gguf_bytes = id->gguf_bytes,
        .model_id = id->model_id,
        .n_layer = id->n_layer,
        .n_embd = id->n_embd,
        .n_vocab = id->n_vocab,
        .quant_bits = id->quant_bits,
        .ctx_size = id->ctx_size,
        .gate_slot_start = id->gate_slot_start,
        .gate_slot_step = id->gate_slot_step,
        .gates_per_token = id->gates_per_token,
        .split_flags = id->split_flags,
    };
    memcpy(mine.gate_slot_mask, id->gate_slot_mask,
           sizeof(mine.gate_slot_mask));
    ds4_tp_hello_fixed theirs;
    if (!tp_write_full(tp->control_fd, &mine, sizeof(mine)) ||
        !tp_read_full(tp->control_fd, &theirs, sizeof(theirs))) {
        tp_set_err(err, errlen, "tp hello exchange failed");
        return 0;
    }
    if (theirs.magic != DS4_TP_MAGIC) {
        tp_set_err(err, errlen, "tp hello: bad magic (mixed byte order or wrong peer?)");
        return 0;
    }
    if (theirs.version != DS4_TP_PROTOCOL_VERSION) {
        tp_set_err(err, errlen, "tp hello: protocol version %u != %u",
                   theirs.version, DS4_TP_PROTOCOL_VERSION);
        return 0;
    }
    /* Say it out loud on SUCCESS, not only on mismatch.  A version bump is a
     * silent property otherwise -- the only evidence both ranks agreed is that
     * nothing failed -- and version 12 exists precisely because 10 and 11 name
     * two incompatible wire protocols that would both have completed bring-up
     * against each other's frame numbering.  A post-merge smoke test has to be
     * able to READ the version, not infer it. */
    fprintf(stderr, "ds4-tp: hello ok, protocol version %u, %u gate slots/layer\n",
            (unsigned)DS4_TP_PROTOCOL_VERSION,
            (unsigned)DS4_TP_GATES_PER_LAYER);
    if (theirs.role == mine.role) {
        tp_set_err(err, errlen, "tp hello: both sides claim role %u", mine.role);
        return 0;
    }
    if (theirs.gguf_bytes != mine.gguf_bytes || theirs.model_id != mine.model_id ||
        theirs.n_layer != mine.n_layer || theirs.n_embd != mine.n_embd ||
        theirs.n_vocab != mine.n_vocab || theirs.quant_bits != mine.quant_bits ||
        theirs.gate_slot_start != mine.gate_slot_start ||
        theirs.gate_slot_step != mine.gate_slot_step ||
        theirs.gates_per_token != mine.gates_per_token ||
        theirs.split_flags != mine.split_flags ||
        memcmp(theirs.gate_slot_mask, mine.gate_slot_mask,
               sizeof(mine.gate_slot_mask)) != 0) {
        tp_set_err(err, errlen,
                   "tp hello: identity mismatch (peer gguf=%llu id=%u layers=%u "
                   "embd=%u vocab=%u qbits=%u gates/token=%u split_flags=0x%x; "
                   "ours gates/token=%u split_flags=0x%x)",
                   (unsigned long long)theirs.gguf_bytes, theirs.model_id,
                   theirs.n_layer, theirs.n_embd, theirs.n_vocab,
                   theirs.quant_bits, theirs.gates_per_token,
                   theirs.split_flags, mine.gates_per_token, mine.split_flags);
        return 0;
    }
    const uint32_t mask_count = tp_gate_mask_count(mine.gate_slot_mask);
    const uint32_t n_slots = mine.n_layer * DS4_TP_GATES_PER_LAYER;
    if ((mask_count != 0 && mask_count != mine.gates_per_token) ||
        !tp_gate_mask_fits(mine.gate_slot_mask, n_slots)) {
        tp_set_err(err, errlen,
                   "tp hello: invalid gate schedule mask (%u bits, %u gates, %u slots)",
                   mask_count, mine.gates_per_token, n_slots);
        return 0;
    }
    tp->peer_ctx = theirs.ctx_size;
    tp->n_layer = id->n_layer;
    tp->n_embd = id->n_embd;
    tp->vec_bytes = (uint64_t)id->n_embd * sizeof(float);
    tp->n_slots = n_slots;
    tp->gate_slot_start = id->gate_slot_start;
    tp->gate_slot_step = id->gate_slot_step;
    tp->gates_per_token = id->gates_per_token;
    memcpy(tp->gate_slot_mask, id->gate_slot_mask,
           sizeof(tp->gate_slot_mask));
    tp_slab_layout(tp);
    /* Transport decision: RDMA only when both sides can. */
    int want_rdma = tp->opt.transport != DS4_TP_TRANSPORT_TCP;
    tp->rdma_active = want_rdma && rdma_ok && theirs.rdma_ok;
    if (tp->opt.transport == DS4_TP_TRANSPORT_RDMA && !tp->rdma_active) {
        tp_set_err(err, errlen, "tp: --transport rdma but %s side has no active device",
                   rdma_ok ? "the peer" : "this");
        return 0;
    }
    return 1;
}

int ds4_tp_create(
        ds4_tp **out,
        const ds4_tp_options *opt,
        const ds4_tp_identity *id,
        char *err,
        size_t errlen)
{
    *out = NULL;
    ds4_tp *tp = calloc(1, sizeof(*tp));
    if (!tp) {
        tp_set_err(err, errlen, "tp: out of memory");
        return 0;
    }
    tp->opt = *opt;
    tp->rank = opt->role == DS4_TP_LEADER ? 0 : 1;
    tp->control_fd = -1;
    tp->data_fd = -1;
    tp->timeout_sec = DS4_TP_DEFAULT_TIMEOUT_SEC;
    const char *tmo = getenv("DS4_TP_TIMEOUT_SEC");
    if (tmo) tp->timeout_sec = (uint64_t)atoi(tmo);
    tp->gate_timeout_ms = DS4_TP_DEFAULT_GATE_TIMEOUT_MS;
    tp->gate_timeout_src = "default";
    /* An ANE shared-expert sidecar adds ~14-17 ms per layer for the gate window
     * to absorb, and 750 ms was sized without one. ANESIDE4 shipped a harness
     * that exported DS4_TP_GATE_TIMEOUT_MS=3000 and the process still came up
     * at 750, so the arms that needed the mitigation ran without it and failed
     * the same way as before -- a mitigation that depends on an env surviving
     * ssh, env(1) and nohup is a mitigation that can silently go missing.
     *
     * So the transport raises its own budget when a sidecar is configured. The
     * explicit knob still wins; this only moves the FLOOR. */
    const char *ane = getenv("DS4_METAL_ANE_SHEXP");
    const int ane_on = ane && ane[0] && strcmp(ane, "0") != 0 &&
                       strcmp(ane, "off") != 0;
    if (ane_on && tp->gate_timeout_ms < DS4_TP_ANE_GATE_TIMEOUT_MS) {
        tp->gate_timeout_ms = DS4_TP_ANE_GATE_TIMEOUT_MS;
        tp->gate_timeout_src = "ane-sidecar";
    }
    const char *gate_tmo = getenv("DS4_TP_GATE_TIMEOUT_MS");
    if (gate_tmo) {
        const long value = strtol(gate_tmo, NULL, 10);
        if (value > 0 && value <= 60000) {
            tp->gate_timeout_ms = (uint64_t)value;
            tp->gate_timeout_src = "env";
        } else {
            fprintf(stderr, "ds4-tp: DS4_TP_GATE_TIMEOUT_MS=%s out of range "
                            "(1..60000) -- IGNORED, staying at %llums\n",
                    gate_tmo, (unsigned long long)tp->gate_timeout_ms);
        }
    }
    tp->meet_timeout_ms = DS4_TP_DEFAULT_MEET_TIMEOUT_MS;
    const char *meet_tmo = getenv("DS4_TP_MEET_TIMEOUT_MS");
    if (meet_tmo) {
        const long value = strtol(meet_tmo, NULL, 10);
        /* Same 60 s ceiling as the in-gate override, and for a stronger reason:
         * the barrier holds a committed command buffer open, so anything past
         * the measured watchdog envelope hands the kill to macOS instead of
         * reporting it here. */
        if (value > 0 && value <= 60000) tp->meet_timeout_ms = (uint64_t)value;
    }
    /* Raising the in-gate budget past the arrival budget means "be more
     * patient", never "be patient in the exchange but not at the barrier". */
    if (tp->gate_timeout_ms > tp->meet_timeout_ms)
        tp->meet_timeout_ms = tp->gate_timeout_ms;
    pthread_mutexattr_t ca;
    pthread_mutexattr_init(&ca);
    pthread_mutexattr_settype(&ca, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&tp->control_lock, &ca);
    pthread_mutexattr_destroy(&ca);

    int rdma_ok = 0;
#ifdef DS4_TP_HAVE_VERBS
    if (opt->transport != DS4_TP_TRANSPORT_TCP &&
        (uint64_t)id->n_embd * sizeof(float) <= 2ull * DS4_TP_RDMA_MAX_MSG)
        rdma_ok = tp_rdma_probe(&tp->rdma.api);
#endif

    int listener = -1;
    if (tp->rank == 0) {
        listener = tp_listen(opt->listen_host, opt->listen_port, err, errlen);
        if (listener < 0) goto fail;
        fprintf(stderr, "ds4-tp: waiting for worker on %s:%d ...\n",
                opt->listen_host ? opt->listen_host : "0.0.0.0", opt->listen_port);
        tp->control_fd = accept(listener, NULL, NULL);
        if (tp->control_fd < 0) {
            tp_set_err(err, errlen, "tp accept: %s", strerror(errno));
            goto fail;
        }
    } else {
        tp->control_fd = tp_dial(opt->leader_host, opt->leader_port,
                                 (double)tp->timeout_sec, err, errlen);
        if (tp->control_fd < 0) goto fail;
    }
    tp_socket_tune(tp->control_fd);

    if (!tp_hello_exchange(tp, id, rdma_ok, err, errlen)) goto fail;

#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active) {
        if (!tp_rdma_open(tp, err, errlen)) goto fail;
    }
#endif
    {
        /* Second socket dedicated to gate traffic so control frames never
         * interleave with gate payloads.  Created under RDMA too for
         * headers, verify-block gates, and transport fallback. */
        if (tp->rank == 0) {
            tp->data_fd = accept(listener, NULL, NULL);
            if (tp->data_fd < 0) {
                tp_set_err(err, errlen, "tp data accept: %s", strerror(errno));
                goto fail;
            }
        } else {
            tp->data_fd = tp_dial(opt->leader_host, opt->leader_port,
                                  (double)tp->timeout_sec, err, errlen);
            if (tp->data_fd < 0) goto fail;
        }
        tp_socket_tune(tp->data_fd);
        if (!tp_socket_set_gate_timeout(tp->data_fd, tp->gate_timeout_ms)) {
            tp_set_err(err, errlen, "tp data socket timeout: %s", strerror(errno));
            goto fail;
        }
    }
    if (listener >= 0) close(listener);
    fprintf(stderr,
            "ds4-tp: %s connected, transport=%s gate-timeout=%llums(%s) "
            "meet-timeout=%llums\n",
            tp->rank == 0 ? "worker" : "leader",
            tp->rdma_active ? "rdma" : "tcp",
            (unsigned long long)tp->gate_timeout_ms, tp->gate_timeout_src,
            (unsigned long long)tp->meet_timeout_ms);
    *out = tp;
    return 1;
fail:
    if (listener >= 0) close(listener);
    ds4_tp_free(tp);
    return 0;
}

int ds4_tp_attach_slab(ds4_tp *tp, void *base, char *err, size_t errlen) {
    tp->slab = base;
    memset(tp->slab + tp->in_flags_off, 0, (uint64_t)tp->n_slots * 8);
    memset(tp->slab + tp->token_off, 0, 16);
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active) {
        for (;;) {
            tp->rdma.warm_failed = 0;
            if (tp_rdma_register_and_exchange(tp, err, errlen)) return 1;
            if (!tp->rdma.warm_failed || tp->rdma.setup_attempt >= 3) return 0;
            tp->rdma.setup_attempt++;
            fprintf(stderr, "ds4-tp: %s; recreating the queue pair (setup attempt %u)\n",
                    err, tp->rdma.setup_attempt + 1u);
            if (!tp_rdma_recreate_qp(tp, err, errlen)) return 0;
        }
    }
#endif
    (void)err; (void)errlen;
    return 1;
}

void ds4_tp_free(ds4_tp *tp) {
    if (!tp) return;
#ifdef DS4_TP_HAVE_VERBS
    tp_rdma_close(tp);
#endif
    if (tp->control_fd >= 0) close(tp->control_fd);
    if (tp->data_fd >= 0) close(tp->data_fd);
    pthread_mutex_destroy(&tp->control_lock);
    free(tp);
}

/* Hold the control-plane lock across a whole (send command -> wait ack)
 * round trip.  Guarantees one in-flight command per rank so the matching
 * COMMAND_ACK is never consumed by a different session's waiter.  Safe to
 * hold across GPU encode: the gate exchanges run on the service thread over
 * data_fd/RDMA and never acquire control_lock. */
void ds4_tp_control_lock(ds4_tp *tp) {
    if (tp) pthread_mutex_lock(&tp->control_lock);
}

void ds4_tp_control_unlock(ds4_tp *tp) {
    if (tp) pthread_mutex_unlock(&tp->control_lock);
}

int ds4_tp_rank(const ds4_tp *tp) { return tp->rank; }
bool ds4_tp_is_rdma(const ds4_tp *tp) { return tp->rdma_active; }
uint32_t ds4_tp_peer_ctx(const ds4_tp *tp) { return tp->peer_ctx; }
bool ds4_tp_failed(const ds4_tp *tp) {
    return tp && atomic_load_explicit(&tp->failed, memory_order_acquire);
}
void ds4_tp_mark_failed(ds4_tp *tp) {
    if (!tp) return;
    /* Set in several places and cleared in none: once the control stream is
     * misframed there is no in-band way to resynchronise, so the pair is done
     * for this process.  That is defensible, but it used to happen in total
     * silence -- every later ds4_session_rewind()/ds4_session_invalidate()
     * quietly skipped its mirror and the ranks drifted apart with no log line
     * to explain why the pair had gone useless.  Say it once, loudly. */
    if (!atomic_exchange_explicit(&tp->failed, true, memory_order_acq_rel)) {
        ds4_log(stderr, DS4_LOG_ERROR,
                "tp: transport marked failed; this pair can no longer stay in "
                "sync and both ranks must be restarted");
    }
}

/* ------------------------------------------------------------------------
 * Gate exchange.
 * --------------------------------------------------------------------- */

int ds4_tp_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t gate, uint64_t seq) {
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active) return tp_rdma_gate_exchange(tp, layer, gate, seq);
#endif
    /* TCP: both sides write their partial then read the peer's.  16KB per
     * direction fits comfortably in the socket buffers, so the symmetric
     * write-then-read cannot deadlock.  Header and payload go out in one
     * writev so NODELAY does not split them into two segments. */
    ds4_tp_gate_header h = { DS4_TP_MAGIC, (uint16_t)layer, (uint16_t)gate, seq };
    struct iovec iov[2] = {
        { &h, sizeof(h) },
        { tp->slab + ds4_tp_slab_out_offset(tp, layer, gate), tp->vec_bytes },
    };
    size_t want = sizeof(h) + tp->vec_bytes;
    ssize_t w = writev(tp->data_fd, iov, 2);
    /* SO_SNDTIMEO can report EAGAIN with nothing written; that is the same
     * "finish with the plain path" case as a short writev, only at zero. */
    if (w < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) w = 0;
    tp_io_status io = TP_IO_OK;
    if (w < 0) {
        io = TP_IO_ERROR;
    } else if ((size_t)w != want) {
        const double send_by = tp_gate_deadline(tp);
        size_t done = (size_t)w;
        if (done < sizeof(h)) {
            io = tp_write_until(tp->data_fd, (char *)&h + done,
                                sizeof(h) - done, send_by);
            done = sizeof(h);
        }
        if (io == TP_IO_OK) {
            const uint64_t payload_done = done - sizeof(h);
            io = tp_write_until(tp->data_fd,
                                tp->slab + ds4_tp_slab_out_offset(tp, layer, gate) +
                                    payload_done,
                                tp->vec_bytes - payload_done, send_by);
        }
    }
    if (io != TP_IO_OK) {
        fprintf(stderr, "ds4-tp: gate l=%u g=%u seq=%llu send failed: %s\n",
                layer, gate, (unsigned long long)seq, tp_io_why(io));
        return 0;
    }
    ds4_tp_gate_header ph;
    io = tp_read_until(tp->data_fd, &ph, sizeof(ph), tp_meet_deadline(tp));
    if (io != TP_IO_OK) {
        fprintf(stderr, "ds4-tp: gate l=%u g=%u seq=%llu header: %s\n",
                layer, gate, (unsigned long long)seq, tp_io_why(io));
        return 0;
    }
    if (ph.magic != DS4_TP_MAGIC || ph.layer != layer || ph.gate != gate || ph.seq != seq) {
        fprintf(stderr, "ds4-tp: gate desync: got l=%u g=%u seq=%llu, want l=%u g=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, gate, (unsigned long long)seq);
        return 0;
    }
    io = tp_read_until(tp->data_fd, tp->slab + ds4_tp_slab_in_offset(tp, layer, gate),
                       tp->vec_bytes, tp_gate_deadline(tp));
    if (io != TP_IO_OK) {
        fprintf(stderr, "ds4-tp: gate l=%u g=%u seq=%llu payload: %s\n",
                layer, gate, (unsigned long long)seq, tp_io_why(io));
        return 0;
    }
    return 1;
}

/* Verify-block batch gate: one exchange per layer moving all block rows at
 * once. The payload lives in the registered slab, so RDMA sends it directly;
 * TCP remains the symmetric write-then-read fallback. */
/* Verify-block window: called on both ranks right before a speculative
 * verify block whose every layer ends in one batch gate of `rows` rows.
 * Drains the decode receive window, posts the first layer's receives,
 * crosses one control-byte barrier (so both sides are posted before any
 * send), and from then on each gate posts the next layer's receives and
 * sends its rows without any handshake.  Returns 1 also when RDMA is not
 * in use (the TCP fallback keeps its per-gate headers). */
int ds4_tp_batch_block_begin(ds4_tp *tp, uint32_t rows, uint32_t n_layers) {
#ifdef DS4_TP_HAVE_VERBS
    if (!tp->rdma_active || !tp_rdma_big_gate_capable(tp)) return 1;
    if (getenv("DS4_TP_DISABLE_VERIFY_WINDOW")) return 1;
    ds4_tp_rdma *r = &tp->rdma;
    if (rows == 0 || rows > DS4_TP_BATCH_MAX_ROWS || n_layers == 0 || r->block_active) return 0;
    if (!tp_rdma_drain_decode_window(tp)) return 0;
    pthread_mutex_lock(&r->post_lock);
    r->block_active = true;
    r->block_rows = rows;
    r->block_layers = n_layers;
    r->block_posted = 0;
    r->block_recv_done = 0;
    const int ok = tp_rdma_block_post_layer(tp, 0);
    pthread_mutex_unlock(&r->post_lock);
    if (!ok) { r->block_active = false; return 0; }
    if (!tp_rdma_window_barrier(tp, 0xB5u)) { r->block_active = false; return 0; }
    if (getenv("DS4_TP_BIG_GATE_DEBUG"))
        fprintf(stderr, "ds4-tp: verify window armed: %u rows x %u layers\n", rows, n_layers);
    return 1;
#else
    (void)tp; (void)rows; (void)n_layers;
    return 1;
#endif
}

/* End of the verify block: every gate must have been exchanged (all
 * posted receives consumed) and our signaled sends reaped. */
int ds4_tp_batch_block_end(ds4_tp *tp) {
#ifdef DS4_TP_HAVE_VERBS
    ds4_tp_rdma *r = &tp->rdma;
    if (!r->block_active) return 1;
    int ok = 1;
    const uint64_t want = (uint64_t)r->block_layers * r->block_rows;
    double deadline = tp_now_sec() + (double)tp->gate_timeout_ms / 1000.0;
    while (ok && (r->block_recv_done < want || r->send_outstanding > 0)) {
        ok = tp_rdma_drain_cq(tp);
        if (ok && tp_now_sec() > deadline) {
            fprintf(stderr, "ds4-tp: verify-block end: %llu/%llu rows received, %u sends pending\n",
                    (unsigned long long)r->block_recv_done, (unsigned long long)want,
                    r->send_outstanding);
            ok = 0;
        }
    }
    if (getenv("DS4_TP_BIG_GATE_DEBUG"))
        fprintf(stderr, "ds4-tp: verify window closed: %llu/%llu rows, ok=%d\n",
                (unsigned long long)r->block_recv_done, (unsigned long long)want, ok);
    r->block_active = false;
    return ok;
#else
    (void)tp;
    return 1;
#endif
}

int ds4_tp_batch_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t rows,
                               uint64_t seq) {
    if (tp->data_fd < 0 || rows == 0 || rows > DS4_TP_BATCH_MAX_ROWS) {
        /* Not reachable from the shipped callers -- they clamp rows against
         * DS4_TP_BATCH_MAX_ROWS before encoding the gate -- but a wrong row
         * count here fails the gate with the same signature as a dead wire,
         * so say which one it was. */
        fprintf(stderr,
                "ds4-tp: batch gate refused: layer %u rows %u (max %d), fd %d\n",
                layer, rows, (int)DS4_TP_BATCH_MAX_ROWS, tp->data_fd);
        return 0;
    }
    const uint64_t bytes = (uint64_t)rows * tp->vec_bytes;
    ds4_tp_gate_header h = { DS4_TP_BATCH_MAGIC, (uint16_t)layer,
                             (uint16_t)rows, seq };
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active && tp->rdma.block_active)
        return tp_rdma_block_gate_exchange(tp, layer, rows);
    if (tp->rdma_active && tp_rdma_big_gate_capable(tp)) {
        /* Header pair first: it is the arrival barrier, and only once it has
         * completed are both ranks known to be inside this gate.  The RDMA
         * rounds below can then run on the short in-gate budget. */
        const double meet_by = tp_meet_deadline(tp);
        ds4_tp_gate_header ph;
        tp_io_status io = tp_write_until(tp->data_fd, &h, sizeof(h), meet_by);
        if (io == TP_IO_OK)
            io = tp_read_until(tp->data_fd, &ph, sizeof(ph), meet_by);
        if (io != TP_IO_OK) {
            fprintf(stderr,
                    "ds4-tp: batch gate l=%u rows=%u seq=%llu header: %s\n",
                    layer, rows, (unsigned long long)seq, tp_io_why(io));
            return 0;
        }
        if (ph.magic != DS4_TP_BATCH_MAGIC || ph.layer != layer ||
            ph.gate != rows || ph.seq != seq) {
            fprintf(stderr,
                    "ds4-tp: batch gate desync: got l=%u rows=%u seq=%llu, "
                    "want l=%u rows=%u seq=%llu\n",
                    ph.layer, ph.gate, (unsigned long long)ph.seq,
                    layer, rows, (unsigned long long)seq);
            return 0;
        }
        if (!tp_rdma_drain_decode_window(tp)) return 0;
        return tp_rdma_big_gate_exchange(
                tp,
                tp->slab + ds4_tp_slab_batch_out_offset(tp, layer),
                tp->slab + ds4_tp_slab_batch_in_offset(tp, layer),
                bytes);
    }
#endif
    struct iovec iov[2] = {
        { &h, sizeof(h) },
        { tp->slab + ds4_tp_slab_batch_out_offset(tp, layer), bytes },
    };
    size_t want = sizeof(h) + bytes;
    ssize_t w = writev(tp->data_fd, iov, 2);
    if (w < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) w = 0;
    tp_io_status io = TP_IO_OK;
    if (w < 0) {
        io = TP_IO_ERROR;
    } else if ((size_t)w != want) {
        const double send_by = tp_gate_deadline(tp);
        size_t done = (size_t)w;
        if (done < sizeof(h)) {
            io = tp_write_until(tp->data_fd, (char *)&h + done,
                                sizeof(h) - done, send_by);
            done = sizeof(h);
        }
        if (io == TP_IO_OK) {
            const uint64_t payload_done = done - sizeof(h);
            io = tp_write_until(tp->data_fd,
                                tp->slab + ds4_tp_slab_batch_out_offset(tp, layer) +
                                    payload_done,
                                bytes - payload_done, send_by);
        }
    }
    if (io != TP_IO_OK) {
        fprintf(stderr, "ds4-tp: batch gate l=%u rows=%u seq=%llu send failed: %s\n",
                layer, rows, (unsigned long long)seq, tp_io_why(io));
        return 0;
    }
    ds4_tp_gate_header ph;
    io = tp_read_until(tp->data_fd, &ph, sizeof(ph), tp_meet_deadline(tp));
    if (io != TP_IO_OK) {
        fprintf(stderr, "ds4-tp: batch gate l=%u rows=%u seq=%llu header: %s\n",
                layer, rows, (unsigned long long)seq, tp_io_why(io));
        return 0;
    }
    if (ph.magic != DS4_TP_BATCH_MAGIC || ph.layer != layer ||
        ph.gate != rows || ph.seq != seq) {
        fprintf(stderr,
                "ds4-tp: batch gate desync: got l=%u rows=%u seq=%llu, "
                "want l=%u rows=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, rows, (unsigned long long)seq);
        return 0;
    }
    io = tp_read_until(tp->data_fd,
                       tp->slab + ds4_tp_slab_batch_in_offset(tp, layer),
                       bytes, tp_gate_deadline(tp));
    if (io != TP_IO_OK) {
        fprintf(stderr, "ds4-tp: batch gate l=%u rows=%u seq=%llu payload: %s\n",
                layer, rows, (unsigned long long)seq, tp_io_why(io));
        return 0;
    }
    return 1;
}

/* Prefill batch gate: RDMA uses the pipelined registered-slab path above.
 * The fallback alternates 2MB TCP write/read rounds in the same order, so
 * neither side can fill its send buffer while the peer is also only writing
 * (the 4MB socket buffers absorb one round). */
#define DS4_TP_BIG_CHUNK (2ull * 1024ull * 1024ull)

int ds4_tp_big_gate_exchange(ds4_tp *tp, uint32_t layer, uint64_t seq,
                             const void *out, void *in, uint64_t bytes) {
    if (tp->data_fd < 0 || !out || !in || bytes == 0) {
        fprintf(stderr,
                "ds4-tp: big gate refused: layer %u seq %llu bytes %llu, fd %d\n",
                layer, (unsigned long long)seq, (unsigned long long)bytes,
                tp->data_fd);
        return 0;
    }
#ifdef DS4_TP_HAVE_VERBS
    /* Per-gate timing belongs to the gate-line levels only.  In accounting
     * mode (level 2) this whole block is off: it runs on the service thread
     * inside the gate, and at 87 exchanges per chunk its fprintf perturbs the
     * very numbers the accounting is there to collect.  Level 3 asks for both
     * and accepts the perturbation. */
    const int dbg = tp_bulk_gateline_on();
    const double t_start = dbg ? tp_now_sec() : 0.0;
#endif
    ds4_tp_gate_header h = { DS4_TP_BATCH_MAGIC, (uint16_t)layer, 0xB16u, seq };
    /* This header pair is the pair's only rendezvous for a prefill gate, and on
     * the first gate of a session the peer can be a whole cold layer-0 encode
     * away, so it gets the arrival budget rather than the in-gate one.  Getting
     * that wrong fails the first request of every fresh pair -- see
     * DS4_TP_DEFAULT_MEET_TIMEOUT_MS. */
    const double meet_by = tp_meet_deadline(tp);
    ds4_tp_gate_header ph;
    tp_io_status io = tp_write_until(tp->data_fd, &h, sizeof(h), meet_by);
    if (io == TP_IO_OK)
        io = tp_read_until(tp->data_fd, &ph, sizeof(ph), meet_by);
    if (io != TP_IO_OK) {
        fprintf(stderr, "ds4-tp: big gate l=%u seq=%llu header: %s\n",
                layer, (unsigned long long)seq, tp_io_why(io));
        return 0;
    }
#ifdef DS4_TP_HAVE_VERBS
    const double t_hs = dbg ? tp_now_sec() : 0.0;
#endif
    if (ph.magic != DS4_TP_BATCH_MAGIC || ph.layer != layer ||
        ph.gate != 0xB16u || ph.seq != seq) {
        fprintf(stderr,
                "ds4-tp: big gate desync: got l=%u tag=%x seq=%llu, want l=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, (unsigned long long)seq);
        return 0;
    }
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active && tp_rdma_big_gate_capable(tp)) {
        if (!tp_rdma_drain_decode_window(tp)) return 0;
        const int ok = tp_rdma_big_gate_exchange(tp, out, in, bytes);
        if (dbg) {
            static unsigned n;
            static double hs_acc, tx_acc, last_end;
            const double t_end = tp_now_sec();
            hs_acc += t_hs - t_start; tx_acc += t_end - t_hs;
            if (tp_bulk_gateline_on())
                fprintf(stderr, "ds4-tp: big gate #%u layer %u: %llu bytes, since last release %.2f ms, handshake %.2f, transfer %.2f\n",
                        n + 1, layer, (unsigned long long)bytes,
                        last_end > 0.0 ? (t_start - last_end) * 1e3 : 0.0,
                        (t_hs - t_start) * 1e3, (t_end - t_hs) * 1e3);
            last_end = t_end;
            if ((++n % 32u) == 0u) {
                fprintf(stderr, "ds4-tp: big gate %u: %llu bytes, last: handshake wait %.2f ms, transfer %.2f ms; avg over 32: %.2f / %.2f ms\n",
                        n, (unsigned long long)bytes, (t_hs - t_start) * 1e3, (t_end - t_hs) * 1e3,
                        hs_acc / 32.0 * 1e3, tx_acc / 32.0 * 1e3);
                hs_acc = tx_acc = 0.0;
            }
        }
        return ok;
    }
#endif
    uint64_t off = 0;
    while (off < bytes) {
        const uint64_t n = bytes - off > DS4_TP_BIG_CHUNK ?
                           DS4_TP_BIG_CHUNK : bytes - off;
        const double round_by = tp_gate_deadline(tp);
        io = tp_write_until(tp->data_fd, (const char *)out + off, n, round_by);
        if (io == TP_IO_OK)
            io = tp_read_until(tp->data_fd, (char *)in + off, n, round_by);
        if (io != TP_IO_OK) {
            fprintf(stderr,
                    "ds4-tp: big gate l=%u seq=%llu round at %llu/%llu bytes: %s\n",
                    layer, (unsigned long long)seq, (unsigned long long)off,
                    (unsigned long long)bytes, tp_io_why(io));
            return 0;
        }
        off += n;
    }
    if (getenv("DS4_GLM_TP_DEBUG")) {
        const float *o = (const float *)out, *i = (const float *)in;
        fprintf(stderr,
                "ds4-tp: big gate l=%u seq=%llu out[0..3]=%g %g %g %g in[0..3]=%g %g %g %g\n",
                layer, (unsigned long long)seq,
                o[0], o[1], o[2], o[3], i[0], i[1], i[2], i[3]);
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Lockstep control plane.
 * --------------------------------------------------------------------- */

typedef struct {
    uint64_t session_id;
    uint32_t count;
    uint32_t reserved;
} ds4_tp_token_command_header;

typedef struct {
    uint64_t session_id;
    uint32_t token_count;
    uint32_t image_count;
} ds4_tp_multimodal_command_header;

typedef struct {
    uint32_t token_start;
    uint32_t token_count;
    uint32_t data_width;
    uint32_t width;
    uint32_t height;
    uint32_t content_width;
    uint32_t content_height;
    uint8_t fingerprint[32];
} ds4_tp_vision_wire;

typedef struct {
    uint64_t session_id;
    int32_t value;
    uint32_t reserved;
} ds4_tp_value_command;

typedef struct {
    uint64_t session_id;
    uint64_t seq;
    int32_t token;
    uint32_t flags;     /* was `reserved`, always sent as 0; same wire size */
} ds4_tp_eval_command;

typedef struct {
    uint32_t count;
    uint32_t reserved;
} ds4_tp_batch_command_header;

typedef struct {
    uint64_t prefill_session_id;
    uint32_t prompt_count;
    uint32_t item_count;
} ds4_tp_mixed_command_header;

typedef struct {
    uint64_t session_id;
    int32_t status;
    uint32_t reserved;
} ds4_tp_command_ack;

static int tp_send_token_command(ds4_tp *tp, uint32_t type,
                                 uint64_t session_id, const int *tokens,
                                 uint32_t count, uint32_t flags) {
    const uint64_t bytes64 = sizeof(ds4_tp_token_command_header) +
                             (uint64_t)count * sizeof(int32_t);
    if (!tp || (!tokens && count != 0) || bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = malloc(bytes ? bytes : 1u);
    if (!payload) return 0;
    ds4_tp_token_command_header h = { session_id, count, flags };
    memcpy(payload, &h, sizeof(h));
    int32_t *wire_tokens = (int32_t *)(payload + sizeof(h));
    for (uint32_t i = 0; i < count; i++) wire_tokens[i] = (int32_t)tokens[i];
    const int ok = tp_send_frame(tp->control_fd, type, payload, bytes);
    free(payload);
    return ok;
}

int ds4_tp_send_session_create(ds4_tp *tp, uint64_t session_id, int ctx_size) {
    ds4_tp_value_command msg = { session_id, (int32_t)ctx_size, 0 };
    pthread_mutex_lock(&tp->control_lock);
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_SESSION_CREATE,
                                 &msg, sizeof(msg));
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

int ds4_tp_send_session_destroy(ds4_tp *tp, uint64_t session_id) {
    pthread_mutex_lock(&tp->control_lock);
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_SESSION_DESTROY,
                                 &session_id, sizeof(session_id));
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

int ds4_tp_send_sync(ds4_tp *tp, uint64_t session_id,
                     const int *tokens, uint32_t n_tokens) {
    pthread_mutex_lock(&tp->control_lock);
    const int ok = tp_send_token_command(tp, DS4_TP_FRAME_SYNC, session_id,
                                         tokens, n_tokens, 0);
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

int ds4_tp_send_sync_multimodal(ds4_tp *tp, uint64_t session_id,
                                const int *tokens, uint32_t n_tokens,
                                const ds4_vision_span *images,
                                uint32_t image_count) {
    uint64_t bytes64 = sizeof(ds4_tp_multimodal_command_header) +
                       (uint64_t)n_tokens * sizeof(int32_t);
    if (!tp || (!tokens && n_tokens) || (!images && image_count) ||
        tp->n_embd == 0 || bytes64 > UINT32_MAX)
        return 0;
    for (uint32_t i = 0; i < image_count; i++) {
        const uint64_t values = (uint64_t)images[i].embedding.token_count *
                                tp->n_embd;
        if (values > UINT64_MAX / sizeof(float)) return 0;
        const uint64_t data_bytes = values * sizeof(float);
        const uint64_t add_bytes = sizeof(ds4_tp_vision_wire) + data_bytes;
        if (!images[i].embedding.data || add_bytes > UINT32_MAX ||
            bytes64 > UINT32_MAX - add_bytes)
            return 0;
        bytes64 += add_bytes;
    }
    uint8_t *payload = malloc((size_t)bytes64);
    if (!payload) return 0;
    ds4_tp_multimodal_command_header h = {
        session_id, n_tokens, image_count
    };
    memcpy(payload, &h, sizeof(h));
    uint8_t *p = payload + sizeof(h);
    int32_t *wire_tokens = (int32_t *)p;
    for (uint32_t i = 0; i < n_tokens; i++) wire_tokens[i] = tokens[i];
    p += (uint64_t)n_tokens * sizeof(int32_t);
    for (uint32_t i = 0; i < image_count; i++) {
        const ds4_vision_embedding *embedding = &images[i].embedding;
        ds4_tp_vision_wire wire = {
            images[i].token_start,
            embedding->token_count,
            tp->n_embd,
            embedding->width,
            embedding->height,
            embedding->content_width,
            embedding->content_height,
            {0},
        };
        memcpy(wire.fingerprint, embedding->fingerprint,
               sizeof(wire.fingerprint));
        memcpy(p, &wire, sizeof(wire));
        p += sizeof(wire);
        uint64_t data_bytes = (uint64_t)embedding->token_count *
                              tp->n_embd * sizeof(float);
        memcpy(p, embedding->data, (size_t)data_bytes);
        p += data_bytes;
    }
    int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_SYNC_MULTIMODAL,
                           payload, (uint32_t)bytes64);
    free(payload);
    return ok;
}

int ds4_tp_send_eval(ds4_tp *tp, uint64_t session_id,
                     uint64_t seq, int token, uint32_t flags) {
    ds4_tp_eval_command msg = { session_id, seq, (int32_t)token, flags };
    pthread_mutex_lock(&tp->control_lock);
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_EVAL, &msg, sizeof(msg));
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

int ds4_tp_send_rewind_mode(ds4_tp *tp, uint64_t session_id, int pos,
                            ds4_tp_rewind_mode mode) {
    /* The mode rides the previously-unused `reserved` word, so the frame keeps
     * its size and the decoder keeps its exact-length check.
     *
     * This subsumes upstream's ds4_tp_send_rewind(tp, sid, pos), which is this
     * with mode 0; its call sites are translated rather than kept, so there is
     * one rewind sender and the mode is always chosen explicitly. */
    ds4_tp_value_command msg = { session_id, (int32_t)pos, (uint32_t)mode };
    pthread_mutex_lock(&tp->control_lock);
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_REWIND,
                                 &msg, sizeof(msg));
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

int ds4_tp_rewind_ack_status(bool want_keep, int requested_pos,
                             int reusable_pos) {
    /* KEEP at or below zero is not a representable request -- there is nothing
     * to keep -- so refuse it rather than letting it read as a match against a
     * zero reusable position.  Fail closed: the leader sees the mismatch and
     * invalidates both ranks. */
    if (want_keep && requested_pos <= 0) return 1;
    /* Position equality, not "still reusable > 0": the rewind clamps to this
     * rank's own checkpoint length, so a worker holding less than the leader
     * lands lower and must not report that as a match. */
    const int expected = want_keep ? requested_pos : 0;
    return reusable_pos == expected ? 0 : 1;
}

int ds4_tp_send_glm_mtp(ds4_tp *tp, uint64_t session_id,
                        uint64_t seq, int token, int limit) {
    if (limit < 1 || limit > 2) return 0;
    /* The fourth word is `flags` on an EVAL frame and the shared-token limit on
     * this one; different frame types, so no ambiguity (see recv below). */
    ds4_tp_eval_command msg = { session_id, seq, (int32_t)token, (uint32_t)limit };
    pthread_mutex_lock(&tp->control_lock);
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_GLM_MTP,
                                 &msg, sizeof(msg));
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

int ds4_tp_send_rollback_capture(ds4_tp *tp, uint64_t session_id, int pos) {
    ds4_tp_value_command msg = { session_id, (int32_t)pos, 0 };
    pthread_mutex_lock(&tp->control_lock);
    const int ok = tp_send_frame(tp->control_fd,
                                 DS4_TP_FRAME_ROLLBACK_CAPTURE,
                                 &msg, sizeof(msg));
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

int ds4_tp_send_invalidate(ds4_tp *tp, uint64_t session_id) {
    pthread_mutex_lock(&tp->control_lock);
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_INVALIDATE,
                                 &session_id, sizeof(session_id));
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

int ds4_tp_send_cancel(ds4_tp *tp, uint64_t session_id) {
    pthread_mutex_lock(&tp->control_lock);
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_CANCEL,
                                 &session_id, sizeof(session_id));
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

static int ds4_tp_send_eval_batch_unlocked(
        ds4_tp *tp, const ds4_tp_batch_item *items, uint32_t count) {
    const uint64_t bytes64 = sizeof(ds4_tp_batch_command_header) +
                             (uint64_t)count * sizeof(*items);
    if (!tp || !items || count == 0 || bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = malloc(bytes);
    if (!payload) return 0;
    ds4_tp_batch_command_header h = { count, 0 };
    memcpy(payload, &h, sizeof(h));
    memcpy(payload + sizeof(h), items, (size_t)count * sizeof(*items));
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_EVAL_BATCH,
                                 payload, bytes);
    free(payload);
    return ok;
}

int ds4_tp_send_eval_batch(ds4_tp *tp, const ds4_tp_batch_item *items,
                           uint32_t count) {
    pthread_mutex_lock(&tp->control_lock);
    const int ok = ds4_tp_send_eval_batch_unlocked(tp, items, count);
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

static int ds4_tp_send_mixed_batch_unlocked(
        ds4_tp *tp, uint64_t prefill_session_id,
        const int *prompt, uint32_t prompt_count,
        const ds4_tp_batch_item *items, uint32_t count) {
    const uint64_t prompt_bytes = (uint64_t)prompt_count * sizeof(int32_t);
    const uint64_t item_bytes = (uint64_t)count * sizeof(*items);
    const uint64_t bytes64 = sizeof(ds4_tp_mixed_command_header) +
                             prompt_bytes + item_bytes;
    if (!tp || !prompt || prompt_count == 0 || !items || count == 0 ||
        bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = malloc(bytes);
    if (!payload) return 0;
    ds4_tp_mixed_command_header h = {
        prefill_session_id, prompt_count, count
    };
    memcpy(payload, &h, sizeof(h));
    int32_t *wire_tokens = (int32_t *)(payload + sizeof(h));
    for (uint32_t i = 0; i < prompt_count; i++) {
        wire_tokens[i] = (int32_t)prompt[i];
    }
    memcpy(payload + sizeof(h) + prompt_bytes, items, (size_t)item_bytes);
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_MIXED_BATCH,
                                 payload, bytes);
    free(payload);
    return ok;
}

int ds4_tp_send_mixed_batch(ds4_tp *tp, uint64_t prefill_session_id,
                            const int *prompt, uint32_t prompt_count,
                            const ds4_tp_batch_item *items,
                            uint32_t count) {
    pthread_mutex_lock(&tp->control_lock);
    const int ok = ds4_tp_send_mixed_batch_unlocked(
            tp, prefill_session_id, prompt, prompt_count, items, count);
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

int ds4_tp_send_command_ack(ds4_tp *tp, uint64_t session_id, int status) {
    ds4_tp_command_ack ack = { session_id, (int32_t)status, 0 };
    pthread_mutex_lock(&tp->control_lock);
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_COMMAND_ACK,
                                 &ack, sizeof(ack));
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

/* status_out, when non-NULL, receives the worker's reported status so callers
 * can tell a coordinated stop from a real failure.  It is left at -1 when no
 * well-formed ack for this session arrived, i.e. when the status is unknown
 * rather than merely non-zero. */
static int ds4_tp_wait_command_ack_unlocked(
        ds4_tp *tp, uint64_t session_id,
        const char *operation, int *status_out, char *err, size_t errlen) {
    if (status_out) *status_out = -1;
    uint32_t type = 0, bytes = 0;
    ds4_tp_command_ack ack;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_COMMAND_ACK || bytes != sizeof(ack) ||
        !tp_read_full(tp->control_fd, &ack, sizeof(ack))) {
        ds4_tp_mark_failed(tp);
        tp_set_err(err, errlen, "tp: worker failed during %s",
                   operation ? operation : "command");
        return 0;
    }
    if (ack.session_id != session_id || ack.status != 0) {
        if (status_out && ack.session_id == session_id) {
            *status_out = (int)ack.status;
        }
        tp_set_err(err, errlen,
                   "tp: worker %s failed (session %llu, status %d)",
                   operation ? operation : "command",
                   (unsigned long long)ack.session_id, (int)ack.status);
        return 0;
    }
    if (status_out) *status_out = 0;
    return 1;
}

int ds4_tp_wait_command_ack(ds4_tp *tp, uint64_t session_id,
                            const char *operation, char *err, size_t errlen) {
    return ds4_tp_wait_command_ack_status(tp, session_id, operation, NULL,
                                          err, errlen);
}

int ds4_tp_wait_command_ack_status(ds4_tp *tp, uint64_t session_id,
                                   const char *operation, int *status_out,
                                   char *err, size_t errlen) {
    pthread_mutex_lock(&tp->control_lock);
    const int ok = ds4_tp_wait_command_ack_unlocked(
            tp, session_id, operation, status_out, err, errlen);
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

int ds4_tp_send_stop(ds4_tp *tp) {
    pthread_mutex_lock(&tp->control_lock);
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_STOP, NULL, 0);
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

void ds4_tp_command_free(ds4_tp_command *command) {
    if (!command) return;
    free(command->tokens);
    free(command->items);
    for (uint32_t i = 0; i < command->n_images; i++)
        ds4_vision_embedding_free(&command->images[i].embedding);
    free(command->images);
    memset(command, 0, sizeof(*command));
    command->type = DS4_TP_FRAME_ERROR;
}

static int tp_command_decode_tokens(ds4_tp_command *command,
                                    const uint8_t *payload,
                                    uint32_t bytes,
                                    char *err, size_t errlen) {
    if (bytes < sizeof(ds4_tp_token_command_header)) return 0;
    ds4_tp_token_command_header h;
    memcpy(&h, payload, sizeof(h));
    const uint64_t want = sizeof(h) + (uint64_t)h.count * sizeof(int32_t);
    if (want != bytes) return 0;
    int *tokens = malloc(h.count ? (size_t)h.count * sizeof(*tokens) : 1u);
    if (!tokens) {
        tp_set_err(err, errlen, "tp: command token allocation failed");
        return -1;
    }
    const int32_t *wire_tokens = (const int32_t *)(payload + sizeof(h));
    for (uint32_t i = 0; i < h.count; i++) tokens[i] = wire_tokens[i];
    command->session_id = h.session_id;
    command->flags = h.reserved;
    command->tokens = tokens;
    command->n_tokens = h.count;
    return 1;
}

static int tp_command_decode_multimodal(ds4_tp *tp,
                                        ds4_tp_command *command,
                                        const uint8_t *payload,
                                        uint32_t bytes,
                                        char *err, size_t errlen) {
    if (bytes < sizeof(ds4_tp_multimodal_command_header)) return 0;
    ds4_tp_multimodal_command_header h;
    memcpy(&h, payload, sizeof(h));
    uint64_t pos = sizeof(h);
    uint64_t token_bytes = (uint64_t)h.token_count * sizeof(int32_t);
    if (pos + token_bytes > bytes) return 0;
    if (h.image_count > ((uint64_t)bytes - pos - token_bytes) /
                        sizeof(ds4_tp_vision_wire))
        return 0;
    int *tokens = malloc(h.token_count ?
                         (size_t)h.token_count * sizeof(tokens[0]) : 1u);
    ds4_vision_span *images = h.image_count ?
        calloc(h.image_count, sizeof(images[0])) : NULL;
    if (!tokens || (h.image_count && !images)) {
        free(tokens);
        free(images);
        tp_set_err(err, errlen, "tp: multimodal command allocation failed");
        return -1;
    }
    const int32_t *wire_tokens = (const int32_t *)(payload + pos);
    for (uint32_t i = 0; i < h.token_count; i++) tokens[i] = wire_tokens[i];
    pos += token_bytes;
    for (uint32_t i = 0; i < h.image_count; i++) {
        if (pos + sizeof(ds4_tp_vision_wire) > bytes) goto malformed;
        ds4_tp_vision_wire wire;
        memcpy(&wire, payload + pos, sizeof(wire));
        pos += sizeof(wire);
        uint64_t values = (uint64_t)wire.token_count * wire.data_width;
        if (wire.token_count == 0 || wire.data_width != tp->n_embd ||
            wire.width == 0 ||
            values > SIZE_MAX / sizeof(float))
            goto malformed;
        uint64_t data_bytes = values * sizeof(float);
        if (data_bytes > (uint64_t)bytes - pos) goto malformed;
        images[i].token_start = wire.token_start;
        images[i].embedding.data = malloc((size_t)data_bytes);
        if (!images[i].embedding.data) {
            tp_set_err(err, errlen, "tp: image embedding allocation failed");
            goto allocation_failed;
        }
        images[i].embedding.token_count = wire.token_count;
        images[i].embedding.width = wire.width;
        images[i].embedding.height = wire.height;
        images[i].embedding.content_width = wire.content_width;
        images[i].embedding.content_height = wire.content_height;
        memcpy(images[i].embedding.fingerprint, wire.fingerprint,
               sizeof(wire.fingerprint));
        memcpy(images[i].embedding.data, payload + pos, (size_t)data_bytes);
        pos += data_bytes;
    }
    if (pos != bytes) goto malformed;
    command->session_id = h.session_id;
    command->tokens = tokens;
    command->n_tokens = h.token_count;
    command->images = images;
    command->n_images = h.image_count;
    return 1;

malformed:
    tp_set_err(err, errlen, "tp: malformed multimodal sync command");
allocation_failed:
    for (uint32_t i = 0; i < h.image_count; i++)
        ds4_vision_embedding_free(&images[i].embedding);
    free(images);
    free(tokens);
    return 0;
}

static int ds4_tp_recv_command_unlocked(ds4_tp *tp, ds4_tp_command *command,
                                        char *err, size_t errlen) {
    memset(command, 0, sizeof(*command));
    command->type = DS4_TP_FRAME_ERROR;
    uint32_t ftype = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &ftype, &bytes)) {
        tp_set_err(err, errlen, "tp: control channel closed");
        return 0;
    }
    uint8_t *payload = NULL;
    if (bytes != 0) {
        payload = malloc(bytes);
        if (!payload || !tp_read_full(tp->control_fd, payload, bytes)) {
            free(payload);
            tp_set_err(err, errlen, "tp: truncated command frame");
            return 0;
        }
    }
    int ok = 1;
    switch (ftype) {
    case DS4_TP_FRAME_SYNC:
    case DS4_TP_FRAME_VERIFY:
        ok = tp_command_decode_tokens(command, payload, bytes, err, errlen);
        break;
    case DS4_TP_FRAME_SYNC_MULTIMODAL:
        ok = tp_command_decode_multimodal(tp, command, payload, bytes,
                                          err, errlen);
        break;
    case DS4_TP_FRAME_SESSION_CREATE:
    case DS4_TP_FRAME_ROLLBACK_CAPTURE:
    case DS4_TP_FRAME_REWIND: {
        ds4_tp_value_command msg;
        if (bytes != sizeof(msg)) { ok = 0; break; }
        memcpy(&msg, payload, sizeof(msg));
        command->session_id = msg.session_id;
        command->value = msg.value;
        command->flags = msg.reserved;
        break;
    }
    case DS4_TP_FRAME_SESSION_DESTROY:
    case DS4_TP_FRAME_INVALIDATE:
    /* A CANCEL is normally consumed by the prefill poll inside the SYNC it
     * aborts.  It can still surface here when it lost the race with the
     * worker's own completion -- the leader sent it just as the prefill
     * finished -- so it must parse as a valid frame rather than fall into the
     * unknown-type path, which is fatal. */
    case DS4_TP_FRAME_CANCEL:
        if (bytes != sizeof(command->session_id)) { ok = 0; break; }
        memcpy(&command->session_id, payload, sizeof(command->session_id));
        break;
    case DS4_TP_FRAME_EVAL:
    case DS4_TP_FRAME_GLM_MTP: {
        ds4_tp_eval_command msg;
        if (bytes != sizeof(msg)) { ok = 0; break; }
        memcpy(&msg, payload, sizeof(msg));
        command->session_id = msg.session_id;
        command->seq = msg.seq;
        command->value = msg.token;
        if (ftype == DS4_TP_FRAME_GLM_MTP) {
            if (msg.flags < 1u || msg.flags > 2u) { ok = 0; break; }
            command->limit = (int)msg.flags;
        } else {
            command->flags = msg.flags;
        }
        break;
    }
    case DS4_TP_FRAME_EVAL_BATCH: {
        ds4_tp_batch_command_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        const uint64_t want = sizeof(h) +
                              (uint64_t)h.count * sizeof(ds4_tp_batch_item);
        if (h.count == 0 || want != bytes) { ok = 0; break; }
        command->items = malloc((size_t)h.count * sizeof(*command->items));
        if (!command->items) { ok = -1; break; }
        memcpy(command->items, payload + sizeof(h),
               (size_t)h.count * sizeof(*command->items));
        command->n_items = h.count;
        break;
    }
    case DS4_TP_FRAME_MIXED_BATCH: {
        ds4_tp_mixed_command_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        const uint64_t token_bytes =
            (uint64_t)h.prompt_count * sizeof(int32_t);
        const uint64_t item_bytes =
            (uint64_t)h.item_count * sizeof(ds4_tp_batch_item);
        const uint64_t want = sizeof(h) + token_bytes + item_bytes;
        if (h.prompt_count == 0 || h.item_count == 0 || want != bytes) {
            ok = 0;
            break;
        }
        command->tokens = malloc((size_t)h.prompt_count *
                                 sizeof(*command->tokens));
        command->items = malloc((size_t)h.item_count *
                                sizeof(*command->items));
        if (!command->tokens || !command->items) { ok = -1; break; }
        const int32_t *wire_tokens =
            (const int32_t *)(payload + sizeof(h));
        for (uint32_t i = 0; i < h.prompt_count; i++) {
            command->tokens[i] = wire_tokens[i];
        }
        memcpy(command->items, payload + sizeof(h) + token_bytes,
               (size_t)item_bytes);
        command->session_id = h.prefill_session_id;
        command->n_tokens = h.prompt_count;
        command->n_items = h.item_count;
        break;
    }
    case DS4_TP_FRAME_STOP:
        if (bytes != 0) ok = 0;
        break;
    default:
        ok = 0;
        break;
    }
    free(payload);
    if (ok <= 0) {
        ds4_tp_command_free(command);
        if (ok == 0) {
            tp_set_err(err, errlen, "tp: invalid command frame type %u (%u bytes)",
                       ftype, bytes);
        } else if (!err || !err[0]) {
            tp_set_err(err, errlen, "tp: command allocation failed");
        }
        return 0;
    }
    command->type = (ds4_tp_frame_type)ftype;
    return 1;
}

int ds4_tp_recv_command(ds4_tp *tp, ds4_tp_command *command,
                        char *err, size_t errlen) {
    pthread_mutex_lock(&tp->control_lock);
    const int ok = ds4_tp_recv_command_unlocked(tp, command, err, errlen);
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

int ds4_tp_send_logits_half(ds4_tp *tp, const float *half, uint32_t count) {
    pthread_mutex_lock(&tp->control_lock);
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_LOGITS,
                                 half, count * sizeof(float));
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

/* Split rank 0's blocking logits recv into peer-wait and payload.
 *
 * Stage 0 measured the recv at 0.39 ms/token against rank 1's 0.047 ms send, and
 * the whole stage-1 decision turns on how much of that 0.35 ms gap is the 310 KB
 * crossing the socket (which a transport change could recover) versus rank 0
 * simply arriving first and waiting (which nothing can recover).
 *
 * The obvious probe -- an arm that sends 8 dummy bytes -- is confounded: with the
 * upper half of the logits missing, decode picks different tokens from the second
 * step on, therefore different experts, therefore different work, and the timing
 * difference is no longer attributable to the transport.  Teacher-forcing would
 * fix that but changes what is being measured and needs both arms rebuilt.
 *
 * This needs neither.  The header and the body are already two reads: the header
 * cannot arrive before rank 1 starts sending, so waiting for it IS the peer-wait
 * plus one frame latency, and the body read that follows is very nearly pure
 * payload.  Same code path, correct output, one arm. */
static int ds4_tp_recv_logits_half_unlocked(ds4_tp *tp, float *half, uint32_t count) {
    static int split_profile = -1;
    if (split_profile < 0) {
        split_profile = getenv("DS4_TP_LOGITS_PROFILE") != NULL ? 1 : 0;
    }
    const double t0 = split_profile ? tp_now_sec() : 0.0;
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_LOGITS || bytes != count * sizeof(float)) {
        fprintf(stderr, "ds4-tp: bad logits frame (type %u bytes %u)\n", type, bytes);
        return 0;
    }
    const double t1 = split_profile ? tp_now_sec() : 0.0;
    const int ok = tp_read_full(tp->control_fd, half, bytes);
    if (split_profile && ok) {
        static double wait_ms, body_ms;
        static uint64_t calls;
        const double t2 = tp_now_sec();
        wait_ms += (t1 - t0) * 1000.0;
        body_ms += (t2 - t1) * 1000.0;
        calls++;
        if ((calls % 64u) == 0u) {
            fprintf(stderr,
                    "ds4: tp logits split peer_wait %.3f ms + payload %.3f ms "
                    "= %.3f ms over %llu tokens (%u bytes).  peer_wait is not "
                    "recoverable by any transport change; payload is the stage-1 "
                    "ceiling.\n",
                    wait_ms / (double)calls, body_ms / (double)calls,
                    (wait_ms + body_ms) / (double)calls,
                    (unsigned long long)calls, bytes);
        }
    }
    return ok;
}

int ds4_tp_recv_logits_half(ds4_tp *tp, float *half, uint32_t count) {
    pthread_mutex_lock(&tp->control_lock);
    const int ok = ds4_tp_recv_logits_half_unlocked(tp, half, count);
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

int ds4_tp_send_verify(ds4_tp *tp, uint64_t session_id,
                       const int *drafts, uint32_t n, uint32_t flags) {
    pthread_mutex_lock(&tp->control_lock);
    const int ok = tp_send_token_command(tp, DS4_TP_FRAME_VERIFY, session_id,
                                         drafts, n, flags);
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

enum { DS4_TP_VERIFY_COMMIT_VERSION = 1 };

int ds4_tp_send_verify_commit(ds4_tp *tp, int32_t commit_n, int32_t replay_n) {
    /* Version this payload because the original two-word frame encoded the
     * first word as a boolean full_accept.  Silently mixing old and new ranks
     * would otherwise turn a partial-prefix decision into a full commit. */
    struct {
        uint32_t version;
        int32_t commit;
        int32_t replay;
    } msg = { DS4_TP_VERIFY_COMMIT_VERSION, commit_n, replay_n };
    pthread_mutex_lock(&tp->control_lock);
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_VERIFY_COMMIT,
                                 &msg, sizeof(msg));
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

static int ds4_tp_recv_verify_commit_unlocked(
        ds4_tp *tp, int32_t *commit_n, int32_t *replay_n) {
    uint32_t type = 0, bytes = 0;
    struct {
        uint32_t version;
        int32_t commit;
        int32_t replay;
    } msg;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_VERIFY_COMMIT || bytes != sizeof(msg) ||
        !tp_read_full(tp->control_fd, &msg, sizeof(msg)) ||
        msg.version != DS4_TP_VERIFY_COMMIT_VERSION) {
        fprintf(stderr, "ds4-tp: bad verify-commit frame (type %u bytes %u)\n",
                type, bytes);
        return 0;
    }
    *commit_n = msg.commit;
    *replay_n = msg.replay;
    return 1;
}

int ds4_tp_recv_verify_commit(ds4_tp *tp, int32_t *commit_n, int32_t *replay_n) {
    pthread_mutex_lock(&tp->control_lock);
    const int ok = ds4_tp_recv_verify_commit_unlocked(tp, commit_n, replay_n);
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

static int ds4_tp_hash_check_unlocked(ds4_tp *tp, uint64_t seq, uint64_t hash,
                                      char *err, size_t errlen) {
    struct { uint64_t seq; uint64_t hash; } mine = { seq, hash }, theirs;
    if (!tp_send_frame(tp->control_fd, DS4_TP_FRAME_HASH, &mine, sizeof(mine))) {
        tp_set_err(err, errlen, "tp: hash send failed");
        return 0;
    }
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_HASH || bytes != sizeof(theirs) ||
        !tp_read_full(tp->control_fd, &theirs, sizeof(theirs))) {
        tp_set_err(err, errlen, "tp: hash recv failed");
        return 0;
    }
    if (theirs.seq != seq || theirs.hash != hash) {
        tp_set_err(err, errlen,
                   "tp: LOCKSTEP DIVERGENCE at seq %llu: local %016llx peer %016llx",
                   (unsigned long long)seq,
                   (unsigned long long)hash, (unsigned long long)theirs.hash);
        return -1;
    }
    return 1;
}

int ds4_tp_hash_check(ds4_tp *tp, uint64_t seq, uint64_t hash, char *err, size_t errlen) {
    pthread_mutex_lock(&tp->control_lock);
    const int ok = ds4_tp_hash_check_unlocked(tp, seq, hash, err, errlen);
    pthread_mutex_unlock(&tp->control_lock);
    return ok;
}

/* ------------------------------------------------------------------------
 * Worker main loop.
 * --------------------------------------------------------------------- */

typedef struct {
    uint64_t id;
    ds4_session *session;
} ds4_tp_worker_session;

typedef struct {
    ds4_tp_worker_session *v;
    uint32_t len;
    uint32_t cap;
} ds4_tp_worker_sessions;

static int tp_worker_session_index(const ds4_tp_worker_sessions *sessions,
                                   uint64_t id) {
    if (!sessions || id == 0) return -1;
    for (uint32_t i = 0; i < sessions->len; i++) {
        if (sessions->v[i].id == id) return (int)i;
    }
    return -1;
}

static ds4_session *tp_worker_session_find(
        const ds4_tp_worker_sessions *sessions, uint64_t id) {
    const int index = tp_worker_session_index(sessions, id);
    return index >= 0 ? sessions->v[index].session : NULL;
}

static int tp_worker_session_add(ds4_tp_worker_sessions *sessions,
                                 uint64_t id, ds4_session *session) {
    if (!sessions || !session || id == 0 ||
        tp_worker_session_index(sessions, id) >= 0) return 0;
    if (sessions->len == sessions->cap) {
        uint32_t cap = sessions->cap ? sessions->cap * 2u : 8u;
        ds4_tp_worker_session *v =
            realloc(sessions->v, (size_t)cap * sizeof(*v));
        if (!v) return 0;
        sessions->v = v;
        sessions->cap = cap;
    }
    sessions->v[sessions->len++] = (ds4_tp_worker_session){ id, session };
    return 1;
}

static void tp_worker_session_remove(ds4_tp_worker_sessions *sessions,
                                     uint32_t index) {
    if (!sessions || index >= sessions->len) return;
    ds4_session_free(sessions->v[index].session);
    if (index + 1u < sessions->len) {
        memmove(&sessions->v[index], &sessions->v[index + 1u],
                (size_t)(sessions->len - index - 1u) * sizeof(sessions->v[0]));
    }
    sessions->len--;
}

static int tp_worker_send_logits(ds4_tp *tp, ds4_session *session,
                                 float *logits, int vocab) {
    if (!logits || vocab <= 0 || (vocab & 1) != 0) return 0;
    const uint32_t vhalf = (uint32_t)vocab / 2u;
    return ds4_session_copy_logits(session, logits, vocab) == vocab &&
           ds4_tp_send_logits_half(tp, logits + vhalf, vhalf);
}

int ds4_tp_leader_bind(ds4_tp **out, ds4_engine *engine,
                       const ds4_tp_options *opt, int ctx_size,
                       char *err, size_t errlen) {
    if (out) *out = NULL;
    if (!out || !engine || !opt) {
        tp_set_err(err, errlen, "invalid tensor-parallel leader bind");
        return 0;
    }
    ds4_tp_identity id = {
        .gguf_bytes = ds4_engine_model_bytes(engine),
        .model_id = (uint32_t)ds4_engine_model_id(engine),
        .n_layer = (uint32_t)ds4_engine_layer_count(engine),
        .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
        .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
        .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
        .ctx_size = (uint32_t)ctx_size,
    };
    ds4_engine_tp_gate_schedule(engine,
                                &id.gate_slot_start,
                                &id.gate_slot_step,
                                &id.gates_per_token,
                                id.gate_slot_mask,
                                &id.split_flags);
    ds4_tp *tp = NULL;
    if (!ds4_tp_create(&tp, opt, &id, err, errlen)) return 0;
    if (!ds4_engine_tp_bind(engine, tp, err, errlen)) {
        ds4_tp_free(tp);
        return 0;
    }
    *out = tp;
    return 1;
}

int ds4_tp_worker_run(ds4_engine *engine, const ds4_tp_options *opt) {
    char err[256] = "";
    ds4_tp_identity id = {
        .gguf_bytes = ds4_engine_model_bytes(engine),
        .model_id = (uint32_t)ds4_engine_model_id(engine),
        .n_layer = (uint32_t)ds4_engine_layer_count(engine),
        .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
        .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
        .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
        .ctx_size = 0, /* adopt the leader's */
    };
    ds4_engine_tp_gate_schedule(engine,
                                &id.gate_slot_start,
                                &id.gate_slot_step,
                                &id.gates_per_token,
                                id.gate_slot_mask,
                                &id.split_flags);

    ds4_tp *tp = NULL;
    if (!ds4_tp_create(&tp, opt, &id, err, sizeof(err))) {
        ds4_log(stderr, DS4_LOG_ERROR, "tp worker: %s", err);
        return 1;
    }
    if (!ds4_engine_tp_bind(engine, tp, err, sizeof(err))) {
        ds4_log(stderr, DS4_LOG_ERROR, "tp worker: %s", err);
        ds4_tp_free(tp);
        return 1;
    }
    ds4_tp_worker_sessions sessions = {0};
    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = ds4_engine_tp_vocab_split(engine) ?
        malloc((size_t)vocab * sizeof(*logits)) : NULL;
    if (ds4_engine_tp_vocab_split(engine) && !logits) {
        ds4_log(stderr, DS4_LOG_ERROR, "tp worker: logits buffer allocation failed");
        ds4_tp_free(tp);
        return 1;
    }
    ds4_log(stderr, DS4_LOG_OK, "tp worker ready for mirrored sessions");

    int rc = 0;
    ds4_tokens prompt = {0};
    while (1) {
        ds4_tp_command command;
        if (!ds4_tp_recv_command(tp, &command, err, sizeof(err))) {
            ds4_log(stderr, DS4_LOG_ERROR, "tp worker: %s", err);
            rc = 1;
            break;
        }
        if (command.type == DS4_TP_FRAME_STOP) {
            ds4_log(stderr, DS4_LOG_DEFAULT, "tp worker: leader finished");
            ds4_tp_command_free(&command);
            break;
        }

        if (command.type == DS4_TP_FRAME_SESSION_CREATE) {
            ds4_session *session = NULL;
            int status = 1;
            if (command.session_id != 0 && command.value > 0 &&
                tp_worker_session_index(&sessions, command.session_id) < 0 &&
                ds4_session_create(&session, engine, command.value) == 0 &&
                tp_worker_session_add(&sessions, command.session_id, session)) {
                /* Pay the first-submit cost before acknowledging creation so
                 * it cannot land in the leader's first timed prefill. */
                ds4_session_gpu_warmup(session);
                status = 0;
            } else if (session) {
                ds4_session_free(session);
            }
            if (!ds4_tp_send_command_ack(tp, command.session_id, status)) {
                rc = 1;
            }
            ds4_tp_command_free(&command);
            if (rc != 0) break;
            continue;
        }

        if (command.type == DS4_TP_FRAME_SESSION_DESTROY) {
            const int index = tp_worker_session_index(&sessions,
                                                      command.session_id);
            const int status = index >= 0 ? 0 : 1;
            if (index >= 0) tp_worker_session_remove(&sessions, (uint32_t)index);
            if (!ds4_tp_send_command_ack(tp, command.session_id, status)) rc = 1;
            ds4_tp_command_free(&command);
            if (rc != 0) break;
            continue;
        }

        if (command.type == DS4_TP_FRAME_CANCEL) {
            /* Lost the race with our own completion: the prefill it was meant
             * to abort has already finished and been acked, so there is nothing
             * to stop.  Dropping it is correct and must not be fatal -- and it
             * carries no ack, so the control stream stays aligned.  Handled
             * before the session lookup so a cancel for a session that has
             * since been destroyed is also harmless. */
            ds4_tp_command_free(&command);
            continue;
        }

        ds4_session *session =
            tp_worker_session_find(&sessions, command.session_id);
        if (command.type != DS4_TP_FRAME_EVAL_BATCH &&
            command.type != DS4_TP_FRAME_MIXED_BATCH && !session) {
            ds4_log(stderr, DS4_LOG_ERROR,
                    "tp worker: unknown session %llu for frame %d",
                    (unsigned long long)command.session_id,
                    (int)command.type);
            ds4_tp_command_free(&command);
            rc = 1;
            break;
        }

        if (command.type == DS4_TP_FRAME_SYNC ||
            command.type == DS4_TP_FRAME_SYNC_MULTIMODAL) {
            prompt.len = 0;
            for (uint32_t i = 0; i < command.n_tokens; i++) {
                ds4_tokens_push(&prompt, command.tokens[i]);
            }
            /* Watch for a leader cancel between chunks for the duration of this
             * sync only, so an aborted prefill stops here at the same boundary
             * instead of spinning on gates the leader will never send. */
            tp_worker_cancel_state cancel_state = {
                .tp = tp, .session_id = command.session_id,
                .cancelled = false, .broken = false,
            };
            ds4_session_set_cancel(session, tp_worker_prefill_cancelled,
                                   &cancel_state);
            int sync_rc = command.type == DS4_TP_FRAME_SYNC_MULTIMODAL ?
                ds4_session_sync_multimodal(session, &prompt,
                                            command.images, command.n_images,
                                            err, sizeof(err)) :
                ds4_session_sync(session, &prompt, err, sizeof(err));
            ds4_session_set_cancel(session, NULL, NULL);
            if (!ds4_tp_send_command_ack(tp, command.session_id, sync_rc)) {
                /* Transport is gone; there is nothing left to serve. */
                rc = 1;
            } else if (sync_rc == DS4_SESSION_SYNC_INTERRUPTED) {
                /* Clean, coordinated stop: an interrupted prefill leaves the
                 * checkpoint at the pre-sync length, which is exactly where the
                 * leader rewinds to, so both ranks remain at the same position.
                 * Do not invalidate -- that would drop to zero while the leader
                 * kept the prefix, and the conversation would re-prefill from
                 * scratch on the next request (or worse, diverge). */
                ds4_log(stderr, DS4_LOG_KVCACHE,
                        "tp worker sync: cancelled by leader; checkpoint kept at %d",
                        ds4_session_pos(session));
            } else if (ds4_backend_device_lost()) {
                /* Not survivable.  A command-buffer error means the GPU
                 * watchdog killed a submission, and nothing this process
                 * submits afterwards can be trusted -- so continuing produces a
                 * worker that answers the control protocol while computing
                 * nothing, which is strictly worse than exiting: the operator
                 * sees a live process and has to work out by hand that it needs
                 * restarting.  Stop loudly and let the supervisor restart us. */
                ds4_log(stderr, DS4_LOG_ERROR,
                        "tp worker sync: %s -- GPU reported a command buffer error, "
                        "so this worker cannot serve any further work; exiting for restart",
                        err);
                rc = 1;
            } else if (sync_rc != 0) {
                /* A failed prefill is a failed *session*, not a failed worker.
                 * The ack above already carried the failure, and the leader
                 * reads split logits only when that ack said success, so the
                 * control stream stays framed.  Drop this session's state and
                 * keep serving: the leader invalidates its own side on the same
                 * failed ack, so both ranks land at zero and the next request
                 * re-prefills from scratch in lockstep. */
                ds4_log(stderr, DS4_LOG_ERROR,
                        "tp worker sync: %s (session invalidated, worker continuing)",
                        err);
                ds4_session_invalidate(session);
            } else if (ds4_engine_tp_vocab_split(engine) &&
                       !tp_worker_send_logits(tp, session, logits, vocab)) {
                rc = 1;
            }
        } else if (command.type == DS4_TP_FRAME_EVAL) {
            /* The frame decides, not this rank's argv. */
            ds4_session_set_tp_eval_spec(
                    session, (command.flags & DS4_TP_EVAL_F_GLM_SPEC) != 0);
            if (ds4_session_eval(session, command.value, err, sizeof(err)) != 0) {
                ds4_log(stderr, DS4_LOG_ERROR, "tp worker eval: %s", err);
                rc = 1;
            }
        } else if (command.type == DS4_TP_FRAME_GLM_MTP) {
            int spec_rc = ds4_session_glm_tp_spec_cycle(
                session, command.value, command.limit, err, sizeof(err));
            if (!ds4_tp_send_command_ack(tp, command.session_id,
                                         spec_rc < 0 ? 1 : 0) || spec_rc < 0) {
                ds4_log(stderr, DS4_LOG_ERROR, "tp worker GLM MTP: %s", err);
                rc = 1;
            }
        } else if (command.type == DS4_TP_FRAME_VERIFY) {
            int replay_n = 0;
            int spec_rc = ds4_session_tp_spec_cycle(session, command.tokens,
                                                    (int)command.n_tokens,
                                                    command.flags, &replay_n,
                                                    err, sizeof(err));
            if (!ds4_tp_send_command_ack(tp, command.session_id, spec_rc)) {
                rc = 1;
            } else if (spec_rc != 0) {
                ds4_log(stderr, DS4_LOG_ERROR, "tp worker verify: %s", err);
                rc = 1;
            } else if (replay_n > 0 &&
                       ds4_engine_tp_vocab_split(engine) &&
                       !tp_worker_send_logits(tp, session, logits, vocab)) {
                rc = 1;
            }
        } else if (command.type == DS4_TP_FRAME_REWIND) {
            /* Obey the leader's mode rather than re-deriving one, then report
             * what was actually applied -- not that the frame was received.
             * The leader compares the two and invalidates both ranks if they
             * differ, so a restore this rank could not perform must come back
             * as INVALIDATE.  See ds4_tp_rewind_mode. */
            const bool want = command.flags == DS4_TP_REWIND_KEEP;
            ds4_session_rewind_mode(session, command.value, want);
            const int status = ds4_tp_rewind_ack_status(
                want, command.value, ds4_session_reusable_pos(session));
            if (!ds4_tp_send_command_ack(tp, command.session_id, status)) rc = 1;
        } else if (command.type == DS4_TP_FRAME_ROLLBACK_CAPTURE) {
            /* Capture only where the leader says, and only if we are where it
             * thinks we are.  A position mismatch means the two ranks are not
             * at the same frontier, which is the one thing the snapshot
             * protocol assumes; refuse rather than record a snapshot the leader
             * would later ask us to restore from the wrong place. */
            int status = 0;
            if (ds4_session_pos(session) != command.value) {
                fprintf(stderr,
                        "ds4: tp: rollback capture at %d but this rank is at "
                        "%d; refusing\n",
                        command.value, ds4_session_pos(session));
                status = 1;
            } else if (!ds4_session_glm53_rollback_capture(session)) {
                status = 1;
            }
            if (status != 0) ds4_session_glm53_rollback_drop(session);
            if (!ds4_tp_send_command_ack(tp, command.session_id, status)) rc = 1;
        } else if (command.type == DS4_TP_FRAME_INVALIDATE) {
            ds4_session_invalidate(session);
        } else if (command.type == DS4_TP_FRAME_EVAL_BATCH ||
                   command.type == DS4_TP_FRAME_MIXED_BATCH) {
            ds4_decode_item *items =
                calloc(command.n_items, sizeof(*items));
            bool mapped = items != NULL;
            for (uint32_t i = 0; mapped && i < command.n_items; i++) {
                items[i].session = tp_worker_session_find(
                    &sessions, command.items[i].session_id);
                items[i].token = command.items[i].token;
                mapped = items[i].session != NULL;
            }
            ds4_session *prefill = NULL;
            if (mapped && command.type == DS4_TP_FRAME_MIXED_BATCH) {
                prefill = tp_worker_session_find(&sessions,
                                                 command.session_id);
                mapped = prefill != NULL;
                prompt.len = 0;
                for (uint32_t i = 0; mapped && i < command.n_tokens; i++) {
                    ds4_tokens_push(&prompt, command.tokens[i]);
                }
            }
            int batch_rc = 1;
            if (mapped && command.type == DS4_TP_FRAME_EVAL_BATCH) {
                batch_rc = ds4_sessions_eval_batch(
                    items, (int)command.n_items, err, sizeof(err));
            } else if (mapped) {
                batch_rc = ds4_sessions_eval_batch_with_prefill(
                    items, (int)command.n_items, prefill, &prompt,
                    err, sizeof(err));
            }
            if (!ds4_tp_send_command_ack(tp, command.session_id, batch_rc)) {
                rc = 1;
            } else if (batch_rc != 0) {
                ds4_log(stderr, DS4_LOG_ERROR,
                        "tp worker batch: %s", err[0] ? err : "failed");
                rc = 1;
            } else if (ds4_engine_tp_vocab_split(engine)) {
                if (prefill &&
                    !tp_worker_send_logits(tp, prefill, logits, vocab)) {
                    rc = 1;
                }
                for (uint32_t i = 0; rc == 0 && i < command.n_items; i++) {
                    if (!tp_worker_send_logits(tp, items[i].session,
                                               logits, vocab)) rc = 1;
                }
            }
            free(items);
        } else {
            ds4_log(stderr, DS4_LOG_ERROR, "tp worker: unexpected frame %d",
                    (int)command.type);
            rc = 1;
        }
        ds4_tp_command_free(&command);
        if (rc != 0) break;
    }
    ds4_tokens_free(&prompt);
    while (sessions.len != 0) {
        tp_worker_session_remove(&sessions, sessions.len - 1u);
    }
    free(sessions.v);
    free(logits);
    ds4_tp_free(tp);
    return rc;
}
