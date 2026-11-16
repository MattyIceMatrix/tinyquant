/* Bit-exactness for the conv engine: the C graph kernel must reproduce the
 * Python reference byte for byte, on every test image. Same standard as the
 * dense engine -- identical, not close. */
#include "tinyq_graph.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_IMAGES 4096

static int8_t *images;
static int32_t *labels;
static uint32_t n_img, img_h, img_w, img_c;

static int load_images(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TQW1", 4) != 0) { fclose(f); return -2; }
    uint32_t hdr[4];
    if (fread(hdr, 4, 4, f) != 4) { fclose(f); return -3; }
    n_img = hdr[0]; img_h = hdr[1]; img_w = hdr[2]; img_c = hdr[3];
    if (n_img == 0 || n_img > MAX_IMAGES) { fclose(f); return -4; }

    const size_t per = (size_t)img_h * img_w * img_c;
    images = malloc(per * n_img);
    labels = malloc(sizeof(int32_t) * n_img);
    if (!images || !labels) { fclose(f); return -5; }
    if (fread(images, 1, per * n_img, f) != per * n_img) { fclose(f); return -6; }
    if (fread(labels, 4, n_img, f) != n_img) { fclose(f); return -7; }
    fclose(f);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.tqg> <images.tqw> <reference_out.bin>\n", argv[0]);
        return 2;
    }

    tqg_model *m = malloc(sizeof(tqg_model));
    int8_t *ref = NULL, *out = NULL;
    int status = 1;
    if (!m) return 1;

    int rc = tqg_load(m, argv[1]);
    if (rc != 0) { fprintf(stderr, "tqg_load failed: %d\n", rc); goto cleanup; }
    if ((rc = load_images(argv[2])) != 0) { fprintf(stderr, "images failed: %d\n", rc); goto cleanup; }

    const uint32_t n_out = tqg_output_size(m);
    FILE *rf = fopen(argv[3], "rb");
    if (!rf) { fprintf(stderr, "cannot open reference\n"); goto cleanup; }
    ref = malloc((size_t)n_img * n_out);
    out = malloc(n_out);
    if (!ref || !out) { fclose(rf); goto cleanup; }
    if (fread(ref, 1, (size_t)n_img * n_out, rf) != (size_t)n_img * n_out) {
        fprintf(stderr, "reference size mismatch\n");
        fclose(rf);
        goto cleanup;
    }
    fclose(rf);

    uint32_t mismatched_values = 0, argmax_diff = 0, correct = 0;
    int32_t worst = 0;
    const size_t per = (size_t)img_h * img_w * img_c;

    for (uint32_t i = 0; i < n_img; i++) {
        tqg_forward(m, images + i * per, out);
        int best_c = 0, best_r = 0;
        for (uint32_t j = 0; j < n_out; j++) {
            const int8_t rv = ref[(size_t)i * n_out + j];
            int32_t d = (int32_t)out[j] - (int32_t)rv;
            if (d != 0) {
                mismatched_values++;
                if (d < 0) d = -d;
                if (d > worst) worst = d;
                if (mismatched_values <= 5)
                    fprintf(stderr, "  image %u class %u: C=%d python=%d\n", i, j, out[j], rv);
            }
            if (out[j] > out[best_c]) best_c = (int)j;
            if (rv > ref[(size_t)i * n_out + best_r]) best_r = (int)j;
        }
        argmax_diff += (best_c != best_r);
        correct += (best_c == labels[i]);
    }

    printf("{\n");
    printf("  \"images\": %u,\n  \"classes\": %u,\n", n_img, n_out);
    printf("  \"mismatched_values\": %u,\n", mismatched_values);
    printf("  \"worst_abs_diff\": %d,\n", worst);
    printf("  \"argmax_disagreements\": %u,\n", argmax_diff);
    printf("  \"c_engine_accuracy\": %.6f,\n", (double)correct / n_img);
    printf("  \"bit_exact\": %s\n", mismatched_values == 0 ? "true" : "false");
    printf("}\n");
    status = (mismatched_values == 0) ? 0 : 3;

cleanup:
    free(out);
    free(ref);
    free(images);
    free(labels);
    free(m);
    return status;
}
