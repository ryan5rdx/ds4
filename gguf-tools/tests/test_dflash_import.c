#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define main deepseek4_quantize_cli_main
#include "../deepseek4-quantize.c"
#undef main

static void require(bool condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "test_dflash_import: %s\n", message);
        exit(1);
    }
}

static char *format_name(const char *format, int stage, const char *suffix) {
    int n = snprintf(NULL, 0, format, stage, suffix);
    require(n > 0, "snprintf sizing failed");
    char *name = xmalloc((size_t)n + 1);
    snprintf(name, (size_t)n + 1, format, stage, suffix);
    return name;
}

static void add_source_tensor(gguf_file *source,
                              int *index,
                              size_t *offset,
                              char *name,
                              ds4q_type type,
                              int n_dims,
                              const int64_t ne[3]) {
    tensor_meta *t = &source->tensors[(*index)++];
    t->name = name;
    t->type = type;
    t->n_dims = n_dims;
    for (int i = 0; i < n_dims; i++) t->ne[i] = ne[i];
    t->size = tensor_nbytes(type, t->ne, t->n_dims);
    t->old_offset = *offset;
    *offset += ds4q_pad(t->size, source->alignment);
}

static gguf_file make_source(void) {
    gguf_file source = {0};
    source.version = 3;
    source.architecture = xstrdup("dflash");
    source.n_tensors = 81;
    source.alignment = 32;
    source.has_dflash_block_count = true;
    source.dflash_block_count = 3;
    source.has_dflash_block_size = true;
    source.dflash_block_size = 5;
    source.has_dflash_noise_token_id = true;
    source.dflash_noise_token_id = 128799;
    source.dflash_target_layer_count = 3;
    source.dflash_target_layers[0] = 41;
    source.dflash_target_layers[1] = 42;
    source.dflash_target_layers[2] = 43;
    source.tensors = xcalloc((size_t)source.n_tensors, sizeof(source.tensors[0]));

    int index = 0;
    size_t offset = 0;
    for (int stage = 0; stage < 3; stage++) {
        for (size_t i = 0; i < sizeof(dflash_block_tensors) / sizeof(dflash_block_tensors[0]); i++) {
            const dflash_block_schema *s = &dflash_block_tensors[i];
            add_source_tensor(&source,
                              &index,
                              &offset,
                              format_name("blk.%d.%s", stage, s->source_suffix),
                              s->source_type,
                              s->n_dims,
                              s->ne);
        }
    }
    for (size_t i = 0; i < sizeof(dflash_special_tensors) / sizeof(dflash_special_tensors[0]); i++) {
        const dflash_special_schema *s = &dflash_special_tensors[i];
        add_source_tensor(&source,
                          &index,
                          &offset,
                          xstrdup(s->source_name),
                          s->source_type,
                          s->n_dims,
                          s->ne);
    }
    require(index == 81, "fixture did not create 81 tensors");

    char path[] = "/tmp/ds4-dflash-import-XXXXXX";
    int fd = mkstemp(path);
    require(fd >= 0, "mkstemp failed");
    source.path = xstrdup(path);

    FILE *fp = fdopen(fd, "wb");
    require(fp != NULL, "fdopen failed");
    require(fwrite("GGUF", 1, 4, fp) == 4, "source magic write failed");
    write_u32(fp, 3);
    write_u64(fp, source.n_tensors);
    write_u64(fp, 5);
    write_gguf_kv_string(fp, "general.architecture", "dflash");
    write_gguf_kv_u32(fp, "dflash.block_count", 3);
    write_gguf_kv_u32(fp, "dflash.block_size", 5);
    write_gguf_string(fp, "dflash.target_layers");
    write_u32(fp, GGUF_TYPE_ARRAY);
    write_u32(fp, GGUF_TYPE_INT32);
    write_u64(fp, 3);
    write_u32(fp, 41);
    write_u32(fp, 42);
    write_u32(fp, 43);
    write_gguf_kv_u32(fp, "tokenizer.ggml.mask_token_id", 128799);
    for (uint64_t i = 0; i < source.n_tensors; i++) {
        const tensor_meta *t = &source.tensors[i];
        write_gguf_string(fp, t->name);
        write_u32(fp, (uint32_t)t->n_dims);
        for (int j = 0; j < t->n_dims; j++) write_u64(fp, (uint64_t)t->ne[j]);
        write_u32(fp, (uint32_t)t->type);
        write_u64(fp, t->old_offset);
    }
    off_t meta_end = ftello(fp);
    require(meta_end >= 0, "source ftello failed");
    source.data_offset = ds4q_pad((size_t)meta_end, source.alignment);
    write_padding(fp, source.data_offset - (size_t)meta_end);
    require(ftruncate(fd, (off_t)(source.data_offset + offset)) == 0,
            "sparse source ftruncate failed");
    require(fclose(fp) == 0, "source close failed");

    char *source_path = xstrdup(source.path);
    free_gguf_file(&source);
    gguf_file parsed = load_gguf_metadata(source_path);
    free(source_path);
    return parsed;
}

static void destroy_source(gguf_file *source) {
    unlink(source->path);
    free_gguf_file(source);
}

static bool build_plan_fails(gguf_file *source) {
    pid_t pid = fork();
    require(pid >= 0, "fork failed");
    if (pid == 0) {
        (void)freopen("/dev/null", "w", stderr);
        dflash_import_plan plan = build_dflash_import_plan(source);
        free_dflash_import_plan(&plan);
        _exit(0);
    }
    int status = 0;
    require(waitpid(pid, &status, 0) == pid, "waitpid failed");
    return WIFEXITED(status) && WEXITSTATUS(status) != 0;
}

static void test_atomic_output(void) {
    char dir[] = "/tmp/ds4-dflash-atomic-XXXXXX";
    require(mkdtemp(dir) != NULL, "mkdtemp failed");
    char output[512];
    snprintf(output, sizeof(output), "%s/model.gguf", dir);

    FILE *initial = fopen(output, "wb");
    require(initial != NULL, "initial output open failed");
    require(fwrite("OLD", 1, 3, initial) == 3, "initial output write failed");
    require(fclose(initial) == 0, "initial output close failed");

    FILE *replacement = open_atomic_output(output);
    require(fwrite("NEW!", 1, 4, replacement) == 4, "replacement write failed");
    size_t len = 0;
    char *contents = read_file(output, &len);
    require(len == 3 && memcmp(contents, "OLD", 3) == 0,
            "old output changed before atomic commit");
    free(contents);
    commit_atomic_output(replacement, output);
    contents = read_file(output, &len);
    require(len == 4 && memcmp(contents, "NEW!", 4) == 0,
            "atomic commit did not publish the replacement");
    free(contents);

    pid_t pid = fork();
    require(pid >= 0, "atomic failure fork failed");
    if (pid == 0) {
        FILE *failed = open_atomic_output(output);
        (void)fwrite("BAD", 1, 3, failed);
        exit(23);
    }
    int status = 0;
    require(waitpid(pid, &status, 0) == pid, "atomic failure waitpid failed");
    require(WIFEXITED(status) && WEXITSTATUS(status) == 23,
            "atomic failure child had unexpected status");
    contents = read_file(output, &len);
    require(len == 4 && memcmp(contents, "NEW!", 4) == 0,
            "failed atomic write damaged the old output");
    free(contents);

    DIR *dp = opendir(dir);
    require(dp != NULL, "opendir failed");
    int files = 0;
    for (struct dirent *de = readdir(dp); de; de = readdir(dp)) {
        if (strcmp(de->d_name, ".") != 0 && strcmp(de->d_name, "..") != 0) files++;
    }
    closedir(dp);
    require(files == 1, "failed atomic write left a temporary file behind");
    require(unlink(output) == 0 && rmdir(dir) == 0, "atomic test cleanup failed");
}

int main(void) {
    test_atomic_output();
    gguf_file source = make_source();
    require(source.architecture && strcmp(source.architecture, "dflash") == 0,
            "source architecture did not survive GGUF readback");
    require(source.has_dflash_block_count && source.dflash_block_count == 3 &&
            source.has_dflash_block_size && source.dflash_block_size == 5 &&
            source.has_dflash_noise_token_id && source.dflash_noise_token_id == 128799,
            "source dflash scalar metadata did not survive GGUF readback");
    require(source.dflash_target_layer_count == 3 &&
            source.dflash_target_layers[0] == 41 &&
            source.dflash_target_layers[1] == 42 &&
            source.dflash_target_layers[2] == 43,
            "source target layers did not survive GGUF readback");
    dflash_import_plan plan = build_dflash_import_plan(&source);
    require(plan.len == 81, "output plan did not contain 81 tensors");
    require(plan.data_offset + plan.tensor_bytes == UINT64_C(10835055488),
            "unexpected full-fat output size");
    require(plan.dspark.target_layer_count == 3, "wrong target-layer count");
    require(plan.dspark.target_layers[0] == 40 &&
            plan.dspark.target_layers[1] == 41 &&
            plan.dspark.target_layers[2] == 42,
            "one-based target layers were not converted to zero-based");

    int type_f32 = 0;
    int type_q8 = 0;
    int type_mxfp4 = 0;
    int copy = 0;
    int to_q8 = 0;
    int to_f32 = 0;
    for (int i = 0; i < plan.len; i++) {
        const dflash_import_tensor *t = &plan.tensors[i];
        type_f32 += t->meta.type == DS4Q_TYPE_F32;
        type_q8 += t->meta.type == DS4Q_TYPE_Q8_0;
        type_mxfp4 += t->meta.type == DS4Q_TYPE_MXFP4;
        copy += t->conversion == DFLASH_COPY;
        to_q8 += t->conversion == DFLASH_BF16_TO_Q8_0;
        to_f32 += t->conversion == DFLASH_BF16_TO_F32;
    }
    require(type_f32 == 45 && type_q8 == 27 && type_mxfp4 == 9,
            "unexpected imported type inventory");
    require(copy == 75 && to_q8 == 2 && to_f32 == 4,
            "unexpected imported conversion inventory");

    int idx = dflash_import_plan_find(&plan, "mtp.0.ffn_gate_inp.weight");
    require(idx >= 0 && plan.tensors[idx].meta.type == DS4Q_TYPE_F32,
            "router gate was not expanded to F32");
    idx = dflash_import_plan_find(&plan, "mtp.2.markov_head.markov_w1.weight");
    require(idx >= 0 && plan.tensors[idx].meta.type == DS4Q_TYPE_Q8_0,
            "Markov W1 was not converted to Q8_0");
    idx = dflash_import_plan_find(&plan, "mtp.2.confidence_head.proj.weight");
    require(idx >= 0 && plan.tensors[idx].meta.type == DS4Q_TYPE_F32,
            "confidence projection was not expanded to F32");
    idx = dflash_import_plan_find(&plan, "mtp.1.ffn_gate_exps.weight");
    require(idx >= 0 && plan.tensors[idx].meta.type == DS4Q_TYPE_MXFP4,
            "MXFP4 routed expert was not preserved");

    {
        char path[] = "/tmp/ds4-dflash-output-XXXXXX";
        int fd = mkstemp(path);
        require(fd >= 0, "output mkstemp failed");
        FILE *fp = fdopen(fd, "wb");
        require(fp != NULL, "output fdopen failed");
        write_dflash_import_header(fp, &plan);
        require(ftruncate(fd, (off_t)(plan.data_offset + plan.tensor_bytes)) == 0,
                "sparse output ftruncate failed");
        require(fclose(fp) == 0, "output close failed");
        struct stat st;
        require(stat(path, &st) == 0 &&
                (uint64_t)st.st_size == plan.data_offset + plan.tensor_bytes,
                "output file has the wrong final length");
        FILE *raw = fopen(path, "rb");
        require(raw != NULL && fseeko(raw, (off_t)plan.meta_size, SEEK_SET) == 0,
                "output padding seek failed");
        for (size_t i = plan.meta_size; i < plan.data_offset; i++) {
            require(fgetc(raw) == 0, "output metadata padding was not zero-filled");
        }
        fclose(raw);
        gguf_file output = load_gguf_metadata(path);
        require(output.architecture && strcmp(output.architecture, "deepseek4-dspark") == 0,
                "output architecture did not survive GGUF readback");
        require(output.n_tensors == 81 && output.data_offset == plan.data_offset,
                "output tensor table did not survive GGUF readback");
        int output_idx = hmap_get(&output.tensor_map, "mtp.0.ffn_gate_inp.weight");
        require(output_idx >= 0 && output.tensors[output_idx].type == DS4Q_TYPE_F32,
                "output F32 router metadata did not survive GGUF readback");
        output_idx = hmap_get(&output.tensor_map, "mtp.2.markov_head.markov_w1.weight");
        require(output_idx >= 0 && output.tensors[output_idx].type == DS4Q_TYPE_Q8_0,
                "output Q8 Markov metadata did not survive GGUF readback");
        free_gguf_file(&output);
        unlink(path);
    }

    free_dflash_import_plan(&plan);

    source.dflash_target_layers[0] = 42;
    source.dflash_target_layers[1] = 43;
    source.dflash_target_layers[2] = 44;
    require(build_plan_fails(&source), "wrong consecutive target layers were accepted");
    source.dflash_target_layers[0] = 41;
    source.dflash_target_layers[1] = 42;
    source.dflash_target_layers[2] = 43;

    source.tensors[0].type = DS4Q_TYPE_Q8_0;
    require(build_plan_fails(&source), "wrong tensor type was accepted");
    source.tensors[0].type = dflash_block_tensors[0].source_type;

    uint64_t saved_offset = source.tensors[0].old_offset;
    source.tensors[0].old_offset = UINT64_MAX - (uint64_t)source.data_offset + 1;
    require(build_plan_fails(&source), "overflowing tensor offset was accepted");
    source.tensors[0].old_offset = saved_offset;

    {
        float input[] = {1.0f, -2.0f};
        uint16_t bf16[2];
        ds4q_f32_to_bf16_row(input, bf16, 2);
        byte_buf source_data = {
            .data = (uint8_t *)bf16,
            .size = sizeof(bf16),
        };
        tensor_meta source_meta = {
            .type = DS4Q_TYPE_BF16,
            .n_dims = 2,
            .ne = {2, 1},
            .size = sizeof(bf16),
        };
        tensor_meta target_meta = source_meta;
        target_meta.type = DS4Q_TYPE_F32;
        target_meta.size = 2 * sizeof(float);
        byte_buf converted = convert_dflash_bf16_tensor(&source_data,
                                                        &source_meta,
                                                        &target_meta,
                                                        DFLASH_BF16_TO_F32);
        float actual[2];
        memcpy(actual, converted.data, sizeof(actual));
        require(converted.size == sizeof(actual) &&
                actual[0] == 1.0f && actual[1] == -2.0f,
                "BF16-to-F32 conversion did not preserve values exactly");
        free(converted.data);
    }

    {
        float input[32];
        uint16_t bf16[32];
        for (int i = 0; i < 32; i++) input[i] = (float)(i - 16) / 8.0f;
        ds4q_f32_to_bf16_row(input, bf16, 32);
        byte_buf source_data = {
            .data = (uint8_t *)bf16,
            .size = sizeof(bf16),
        };
        tensor_meta source_meta = {
            .type = DS4Q_TYPE_BF16,
            .n_dims = 2,
            .ne = {32, 1},
            .size = sizeof(bf16),
        };
        tensor_meta target_meta = source_meta;
        target_meta.type = DS4Q_TYPE_Q8_0;
        target_meta.size = ds4q_row_size(DS4Q_TYPE_Q8_0, 32);
        byte_buf converted = convert_dflash_bf16_tensor(&source_data,
                                                        &source_meta,
                                                        &target_meta,
                                                        DFLASH_BF16_TO_Q8_0);
        require(converted.size == 34, "BF16-to-Q8_0 conversion has wrong row size");
        free(converted.data);
    }

    destroy_source(&source);
    puts("test_dflash_import: OK");
    return 0;
}
