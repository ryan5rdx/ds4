#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define main deepseek4_quantize_cli_main
#include "../deepseek4-quantize.c"
#undef main

static void require(bool condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "test_dspark_mxfp4_plan: %s\n", message);
        exit(1);
    }
}

static dspark_tensor_plan make_plan(dspark_plan_kind kind,
                                    const char *name,
                                    int64_t width) {
    dspark_tensor_plan tp = {0};
    tp.kind = kind;
    tp.meta.name = (char *)name;
    tp.meta.type = DS4Q_TYPE_MXFP4;
    tp.meta.n_dims = kind == DSPARK_PLAN_EXPERT ? 3 : 2;
    tp.meta.ne[0] = width;
    tp.meta.ne[1] = 2;
    tp.meta.ne[2] = kind == DSPARK_PLAN_EXPERT ? 1 : 0;
    return tp;
}

static bool plan_set_size_fails(dspark_tensor_plan tp) {
    pid_t pid = fork();
    require(pid >= 0, "fork failed");
    if (pid == 0) {
        (void)freopen("/dev/null", "w", stderr);
        dspark_plan_set_size(&tp);
        _exit(0);
    }

    int status = 0;
    require(waitpid(pid, &status, 0) == pid, "waitpid failed");
    return WIFEXITED(status) && WEXITSTATUS(status) != 0;
}

int main(void) {
    dspark_tensor_plan expert = make_plan(
        DSPARK_PLAN_EXPERT,
        "mtp.0.ffn_gate_exps.weight",
        64);
    dspark_plan_set_size(&expert);
    require(expert.meta.size == 68, "expert MXFP4 plan has the wrong size");

    dspark_tensor_plan dense = make_plan(
        DSPARK_PLAN_REGULAR,
        "mtp.0.main_proj.weight",
        64);
    require(plan_set_size_fails(dense), "regular MXFP4 plan was accepted");

    dspark_tensor_plan misnamed = make_plan(
        DSPARK_PLAN_EXPERT,
        "mtp.0.main_proj.weight",
        64);
    require(plan_set_size_fails(misnamed), "misnamed expert MXFP4 plan was accepted");

    dspark_tensor_plan unaligned = make_plan(
        DSPARK_PLAN_EXPERT,
        "mtp.0.ffn_gate_exps.weight",
        48);
    require(plan_set_size_fails(unaligned), "unaligned expert MXFP4 plan was accepted");

    puts("test_dspark_mxfp4_plan: OK");
    return 0;
}
