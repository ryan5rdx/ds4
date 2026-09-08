#include "../ds4_tp.c"
#include <assert.h>

int main(void) {
    int fd[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    ds4_tp leader = {
        .control_fd = fd[0], .control_lock = PTHREAD_MUTEX_INITIALIZER,
    };
    ds4_tp worker = {
        .control_fd = fd[1], .control_lock = PTHREAD_MUTEX_INITIALIZER,
    };
    char err[256] = "";
    ds4_tp_command cmd;
    assert(DS4_TP_PROTOCOL_VERSION == 12);
#ifdef DS4_TP_HAVE_VERBS
    assert(tp_rdma_cq_entries(4, 0) == 512);
    assert(tp_rdma_cq_entries(1024, 0) == 768);
    assert(tp_rdma_cq_entries(4095, 0) == 2302);
    assert(tp_rdma_cq_entries(4095, 2048) == 2048);
#endif
    for (int i = 0; i < 4; i++) {
        assert(ds4_tp_send_eval(&leader, 42, 2*i, 100+i));
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
    ds4_tp_eval_command bad_eval = {42, 9, 1, 2};
    assert(tp_send_frame(fd[0], DS4_TP_FRAME_EVAL, &bad_eval, sizeof(bad_eval)));
    assert(!ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    ds4_tp_command_free(&cmd);
    assert(ds4_tp_send_rewind_mode(&leader, 42, 123,
                                   DS4_TP_REWIND_INVALIDATE));
    assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    assert(cmd.type == DS4_TP_FRAME_REWIND && cmd.value == 123 &&
           cmd.flags == DS4_TP_REWIND_INVALIDATE);
    ds4_tp_command_free(&cmd);
    assert(ds4_tp_send_rewind_mode(&leader, 42, 456, DS4_TP_REWIND_KEEP));
    assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    assert(cmd.type == DS4_TP_FRAME_REWIND && cmd.value == 456 &&
           cmd.flags == DS4_TP_REWIND_KEEP);
    ds4_tp_command_free(&cmd);
    assert(DS4_TP_FRAME_RDMA_WARM == 19 &&
           DS4_TP_FRAME_RDMA_POSTED == 20 &&
           DS4_TP_FRAME_GLM_MTP == 21 &&
           DS4_TP_FRAME_CANCEL == 22 &&
           DS4_TP_FRAME_ROLLBACK_CAPTURE == 23);
    assert(ds4_tp_send_cancel(&leader, 42));
    assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    assert(cmd.type == DS4_TP_FRAME_CANCEL && cmd.session_id == 42);
    ds4_tp_command_free(&cmd);
    assert(ds4_tp_send_rollback_capture(&leader, 42, 456));
    assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    assert(cmd.type == DS4_TP_FRAME_ROLLBACK_CAPTURE &&
           cmd.session_id == 42 && cmd.value == 456);
    ds4_tp_command_free(&cmd);
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
    int status = -1;
    assert(!ds4_tp_wait_command_ack_status(&leader, 42, "GLM MTP", &status,
                                           err, sizeof(err)));
    assert(status == 1);
    pthread_mutex_destroy(&leader.control_lock);
    pthread_mutex_destroy(&worker.control_lock);
    close(fd[0]);
    close(fd[1]);
    puts("TP command tests: ok");
    return 0;
}
