/* clock_gettime is POSIX, not C99; -std=c99 hides it without this. */
#define _POSIX_C_SOURCE 200809L
/* Latency benchmark. Reports percentiles rather than a mean: on a shared or
 * frequency-scaling machine the mean hides the tail, and the tail is what an
 * edge deployment has to budget for. */
#include "tinyq.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SAMPLES 4096

static int8_t xq[MAX_SAMPLES][TQ_MAX_WIDTH];
static float xf[MAX_SAMPLES][TQ_MAX_WIDTH];
static int32_t labels[MAX_SAMPLES];
static uint32_t n_samples, n_feat;

static double *timings;

static int load_vectors(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TQV1", 4) != 0) { fclose(f); return -2; }
    if (fread(&n_samples, 4, 1, f) != 1) { fclose(f); return -3; }
    if (fread(&n_feat, 4, 1, f) != 1) { fclose(f); return -3; }
    if (n_samples > MAX_SAMPLES || n_feat > TQ_MAX_WIDTH) { fclose(f); return -4; }
    for (uint32_t i = 0; i < n_samples; i++)
        if (fread(xq[i], 1, n_feat, f) != n_feat) { fclose(f); return -5; }
    for (uint32_t i = 0; i < n_samples; i++)
        if (fread(xf[i], 4, n_feat, f) != n_feat) { fclose(f); return -5; }
    if (fread(labels, 4, n_samples, f) != n_samples) { fclose(f); return -6; }
    fclose(f);
    return 0;
}

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double pct(double *sorted, size_t n, double p)
{
    double idx = (p / 100.0) * (double)(n - 1);
    size_t lo = (size_t)idx;
    size_t hi = lo + 1 < n ? lo + 1 : lo;
    double frac = idx - (double)lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.tq8> <model.tqf> <vectors.tqv> [reps]\n", argv[0]);
        return 2;
    }
    const int reps = (argc > 4) ? atoi(argv[4]) : 200;

    tq_model *mi = malloc(sizeof(tq_model));   /* big; allocated once, before timing */
    tqf_model *mf = malloc(sizeof(tqf_model));
    if (!mi || !mf) { fprintf(stderr, "alloc failed\n"); return 1; }

    int rc;
    if ((rc = tq_load(mi, argv[1])) != 0) {
        fprintf(stderr, "tq_load: %d\n", rc);
        free(mf); free(mi);
        return 1;
    }
    if ((rc = tqf_load(mf, argv[2])) != 0) {
        fprintf(stderr, "tqf_load: %d\n", rc);
        free(mf); free(mi);
        return 1;
    }
    if ((rc = load_vectors(argv[3])) != 0) {
        fprintf(stderr, "vectors: %d\n", rc);
        free(mf); free(mi);
        return 1;
    }

    const size_t n_timings = (size_t)reps * n_samples;
    timings = malloc(sizeof(double) * n_timings);
    if (!timings) { fprintf(stderr, "alloc failed\n"); free(mf); free(mi); return 1; }

    int8_t out_i8[TQ_MAX_WIDTH];
    float out_f32[TQ_MAX_WIDTH];
    volatile int sink = 0;

    /* warm up caches and let the CPU reach a steady clock */
    for (int r = 0; r < 20; r++)
        for (uint32_t i = 0; i < n_samples; i++) {
            tq_forward(mi, xq[i], out_i8);
            tqf_forward(mf, xf[i], out_f32);
            sink += out_i8[0] + (int)out_f32[0];
        }

    /* ---- INT8 ---- */
    size_t k = 0;
    int correct_i8 = 0;
    for (int r = 0; r < reps; r++)
        for (uint32_t i = 0; i < n_samples; i++) {
            double t0 = now_us();
            tq_forward(mi, xq[i], out_i8);
            timings[k++] = now_us() - t0;
            if (r == 0) {
                int pred = tq_argmax_i8(out_i8, mi->layers[mi->n_layers - 1].n_out);
                correct_i8 += (pred == labels[i]);
            }
        }
    qsort(timings, n_timings, sizeof(double), cmp_double);
    double i8_p50 = pct(timings, n_timings, 50), i8_p95 = pct(timings, n_timings, 95),
           i8_p99 = pct(timings, n_timings, 99), i8_min = timings[0];

    /* ---- FP32 ---- */
    k = 0;
    int correct_f32 = 0;
    for (int r = 0; r < reps; r++)
        for (uint32_t i = 0; i < n_samples; i++) {
            double t0 = now_us();
            tqf_forward(mf, xf[i], out_f32);
            timings[k++] = now_us() - t0;
            if (r == 0) correct_f32 += (tqf_predict(mf, xf[i]) == labels[i]);
        }
    qsort(timings, n_timings, sizeof(double), cmp_double);
    double f_p50 = pct(timings, n_timings, 50), f_p95 = pct(timings, n_timings, 95),
           f_p99 = pct(timings, n_timings, 99), f_min = timings[0];

    printf("{\n");
    printf("  \"samples\": %u,\n  \"reps\": %d,\n", n_samples, reps);
    printf("  \"skip_zeros_build\": %d,\n", TQ_SKIP_ZEROS);
    printf("  \"int8\":  {\"min_us\": %.4f, \"p50_us\": %.4f, \"p95_us\": %.4f, \"p99_us\": %.4f, \"accuracy\": %.6f},\n",
           i8_min, i8_p50, i8_p95, i8_p99, (double)correct_i8 / n_samples);
    printf("  \"fp32\":  {\"min_us\": %.4f, \"p50_us\": %.4f, \"p95_us\": %.4f, \"p99_us\": %.4f, \"accuracy\": %.6f},\n",
           f_min, f_p50, f_p95, f_p99, (double)correct_f32 / n_samples);
    printf("  \"speedup_p50\": %.4f,\n  \"speedup_min\": %.4f,\n", f_p50 / i8_p50, f_min / i8_min);
    printf("  \"sink\": %d\n}\n", sink);

    free(timings);
    free(mf);
    free(mi);
    return 0;
}
