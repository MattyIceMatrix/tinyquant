/* Cross-validation: the C engine must reproduce the Python reference output
 * byte for byte, for every test sample. Not "close" -- identical. An INT8
 * pipeline where the trainer-side simulation and the deployed kernel disagree
 * is a pipeline whose accuracy numbers mean nothing, so this is the test the
 * whole repo rests on. */
#include "tinyq.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SAMPLES 4096

static int8_t xq[MAX_SAMPLES][TQ_MAX_WIDTH];
static float xf[MAX_SAMPLES][TQ_MAX_WIDTH];
static int32_t labels[MAX_SAMPLES];
static uint32_t n_samples, n_feat;

static int load_vectors(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TQV1", 4) != 0) { fclose(f); return -2; }
    if (fread(&n_samples, 4, 1, f) != 1 || fread(&n_feat, 4, 1, f) != 1) { fclose(f); return -3; }
    if (n_samples > MAX_SAMPLES || n_feat > TQ_MAX_WIDTH) { fclose(f); return -4; }
    for (uint32_t i = 0; i < n_samples; i++)
        if (fread(xq[i], 1, n_feat, f) != n_feat) { fclose(f); return -5; }
    for (uint32_t i = 0; i < n_samples; i++)
        if (fread(xf[i], 4, n_feat, f) != n_feat) { fclose(f); return -5; }
    if (fread(labels, 4, n_samples, f) != n_samples) { fclose(f); return -6; }
    fclose(f);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.tq8> <vectors.tqv> <reference_out.bin>\n", argv[0]);
        return 2;
    }

    tq_model *m = malloc(sizeof(tq_model));
    if (!m) return 1;
    int8_t *ref = NULL;
    int status = 1;

    int rc = tq_load(m, argv[1]);
    if (rc != 0) { fprintf(stderr, "tq_load failed: %d\n", rc); goto cleanup; }
    if ((rc = load_vectors(argv[2])) != 0) {
        fprintf(stderr, "vectors failed: %d\n", rc);
        goto cleanup;
    }

    const uint32_t n_out = m->layers[m->n_layers - 1].n_out;

    FILE *rf = fopen(argv[3], "rb");
    if (!rf) { fprintf(stderr, "cannot open reference\n"); goto cleanup; }
    ref = malloc((size_t)n_samples * n_out);
    if (!ref) { fclose(rf); goto cleanup; }
    if (fread(ref, 1, (size_t)n_samples * n_out, rf) != (size_t)n_samples * n_out) {
        fprintf(stderr, "reference size mismatch\n");
        fclose(rf);
        goto cleanup;
    }
    fclose(rf);

    int8_t out[TQ_MAX_WIDTH];
    uint32_t mismatched_samples = 0, mismatched_values = 0, argmax_diff = 0, correct = 0;
    int32_t worst_abs_diff = 0;

    for (uint32_t i = 0; i < n_samples; i++) {
        tq_forward(m, xq[i], out);
        int bad = 0;
        for (uint32_t j = 0; j < n_out; j++) {
            int32_t d = (int32_t)out[j] - (int32_t)ref[(size_t)i * n_out + j];
            if (d != 0) {
                bad = 1;
                mismatched_values++;
                int32_t ad = d < 0 ? -d : d;
                if (ad > worst_abs_diff) worst_abs_diff = ad;
                if (mismatched_values <= 5)
                    fprintf(stderr, "  sample %u class %u: C=%d python=%d\n",
                            i, j, out[j], ref[(size_t)i * n_out + j]);
            }
        }
        mismatched_samples += (uint32_t)bad;
        int pc = tq_argmax_i8(out, n_out);
        int pr = tq_argmax_i8(&ref[(size_t)i * n_out], n_out);
        argmax_diff += (pc != pr);
        correct += (pc == labels[i]);
    }

    printf("{\n");
    printf("  \"samples\": %u,\n  \"classes\": %u,\n", n_samples, n_out);
    printf("  \"mismatched_samples\": %u,\n", mismatched_samples);
    printf("  \"mismatched_values\": %u,\n", mismatched_values);
    printf("  \"worst_abs_diff\": %d,\n", worst_abs_diff);
    printf("  \"argmax_disagreements\": %u,\n", argmax_diff);
    printf("  \"c_engine_accuracy\": %.6f,\n", (double)correct / n_samples);
    printf("  \"bit_exact\": %s\n", mismatched_values == 0 ? "true" : "false");
    printf("}\n");

    status = (mismatched_values == 0) ? 0 : 3;

cleanup:
    /* Nothing here is strictly necessary before exit, but the audit in
     * scripts/resource_audit.py runs valgrind with leaks treated as errors,
     * and a harness that leaks makes it harder to trust the engine it is
     * testing. So the harness holds itself to the same standard. */
    free(ref);
    free(m);
    return status;
}
