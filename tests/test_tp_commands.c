#include "../ds4_tp.c"
#include <assert.h>

int main(void) {
    int fd[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    /* Every sender on this branch takes control_lock; a zeroed pthread mutex is
     * not a valid one on Apple's libpthread, so initialise both. */
    ds4_tp leader = { .control_fd = fd[0], .control_lock = PTHREAD_MUTEX_INITIALIZER };
    ds4_tp worker = { .control_fd = fd[1], .control_lock = PTHREAD_MUTEX_INITIALIZER };
    char err[256] = "";
    ds4_tp_command cmd;
    assert(DS4_TP_PROTOCOL_VERSION == 12);
    for (int i = 0; i < 4; i++) {
        assert(ds4_tp_send_eval(&leader, 42, 2*i, 100+i, 0u));
        assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
        assert(cmd.type == DS4_TP_FRAME_EVAL && cmd.value == 100+i);
        assert(cmd.limit == 0 && cmd.session_id == 42 && cmd.seq == (uint64_t)2*i);
        ds4_tp_command_free(&cmd);
        const int limit = 1 + i%2;
        assert(ds4_tp_send_glm_mtp(&leader, 42, 2*i+1, 200+i, limit));
        assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
        assert(cmd.type == DS4_TP_FRAME_GLM_MTP && cmd.value == 200+i);
        assert(cmd.limit == limit && cmd.session_id == 42 && cmd.seq == (uint64_t)2*i+1);
        ds4_tp_command_free(&cmd);
    }
    assert(!ds4_tp_send_glm_mtp(&leader, 42, 9, 1, 0));
    assert(!ds4_tp_send_glm_mtp(&leader, 42, 9, 1, 3));
    for (uint32_t bad = 0; bad <= 3; bad += 3) {
        ds4_tp_eval_command msg = {42, 9, 1, bad};
        assert(tp_send_frame(fd[0], DS4_TP_FRAME_GLM_MTP, &msg, sizeof(msg)));
        assert(!ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
        ds4_tp_command_free(&cmd);
    }
    /* On an EVAL the fourth word is the flags word (DS4_TP_EVAL_F_GLM_SPEC),
     * so it must round-trip rather than be rejected; only GLM_MTP, which reuses
     * the word as a 1..2 limit, range-checks it (asserted above). */
    assert(ds4_tp_send_eval(&leader, 42, 9, 1, DS4_TP_EVAL_F_GLM_SPEC));
    assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    assert(cmd.type == DS4_TP_FRAME_EVAL && cmd.value == 1 &&
           cmd.flags == DS4_TP_EVAL_F_GLM_SPEC && cmd.limit == 0);
    ds4_tp_command_free(&cmd);
    /* ds4_tp_send_rewind_mode subsumes upstream's ds4_tp_send_rewind; mode 0
     * (INVALIDATE) is exactly it, and the mode rides the reserved word. */
    assert(ds4_tp_send_rewind_mode(&leader, 42, 123, DS4_TP_REWIND_INVALIDATE));
    assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    assert(cmd.type == DS4_TP_FRAME_REWIND && cmd.value == 123 &&
           cmd.flags == DS4_TP_REWIND_INVALIDATE);
    ds4_tp_command_free(&cmd);
    assert(ds4_tp_send_rewind_mode(&leader, 42, 456, DS4_TP_REWIND_KEEP));
    assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    assert(cmd.type == DS4_TP_FRAME_REWIND && cmd.value == 456 &&
           cmd.flags == DS4_TP_REWIND_KEEP);
    ds4_tp_command_free(&cmd);
    /* Frames 19-23 must not collide: upstream's RDMA/MTP numbering won, ours
     * moved to 22/23.  A mixed pair is refused by the version bump, but the
     * numbers themselves are asserted here so a future edit cannot re-alias
     * them silently. */
    assert(DS4_TP_FRAME_RDMA_WARM == 19 && DS4_TP_FRAME_RDMA_POSTED == 20 &&
           DS4_TP_FRAME_GLM_MTP == 21 && DS4_TP_FRAME_CANCEL == 22 &&
           DS4_TP_FRAME_ROLLBACK_CAPTURE == 23);
    assert(ds4_tp_send_invalidate(&leader, 42));
    assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    assert(cmd.type == DS4_TP_FRAME_INVALIDATE && cmd.session_id == 42);
    ds4_tp_command_free(&cmd);
    const int tokens[] = {7, 19, 5};
    assert(ds4_tp_send_sync(&leader, 42, tokens, 3));
    assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    assert(cmd.type == DS4_TP_FRAME_SYNC && cmd.n_tokens == 3);
    assert(memcmp(cmd.tokens, tokens, sizeof(tokens)) == 0);
    ds4_tp_command_free(&cmd);
    assert(ds4_tp_send_command_ack(&worker, 42, 0));
    assert(ds4_tp_wait_command_ack(&leader, 42, "rebuild", err, sizeof(err)));
    assert(ds4_tp_send_command_ack(&worker, 42, 1));
    assert(!ds4_tp_wait_command_ack(&leader, 42, "GLM MTP", err, sizeof(err)));
    close(fd[0]);
    close(fd[1]);
    puts("TP command tests: ok");
    return 0;
}
