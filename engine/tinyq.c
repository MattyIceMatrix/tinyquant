#include "tinyq.h"

#include <stdio.h>
#include <string.h>

/* ----------------------------------------------------------------------- *
 * fixed-point requantization -- mirrors tinyquant/fixedpoint.py exactly
 * ----------------------------------------------------------------------- */
static int32_t sat_round_dbl_high_mul(int32_t a, int32_t b)
{
    if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;
    int64_t ab = (int64_t)a * (int64_t)b;
    int64_t nudge = (ab >= 0) ? (1 << 30) : (1 - (1 << 30));
    int64_t out = (ab + nudge) >> 31;
    if (out > INT32_MAX) return INT32_MAX;
    if (out < INT32_MIN) return INT32_MIN;
    return (int32_t)out;
}

static int32_t rounding_divide_by_pot(int32_t x, int exponent)
{
    if (exponent == 0) return x;
    const int32_t mask = (int32_t)((1u << exponent) - 1u);
    const int32_t remainder = x & mask;
    const int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

int32_t tq_requantize(int32_t acc, int32_t q_mult, int lshift, int rshift)
{
    int64_t shifted = (int64_t)acc << lshift;
    if (shifted > INT32_MAX) shifted = INT32_MAX;
    if (shifted < INT32_MIN) shifted = INT32_MIN;
    return rounding_divide_by_pot(
        sat_round_dbl_high_mul((int32_t)shifted, q_mult), rshift);
}

/* ----------------------------------------------------------------------- *
 * loading
 * ----------------------------------------------------------------------- */
static int read_exact(FILE *f, void *dst, size_t n)
{
    return fread(dst, 1, n, f) == n ? 0 : -1;
}

/* Copy n bytes from the file into the model's static arena, returning a
 * pointer into it. Returns NULL if the arena would overflow -- the only
 * allocation policy this engine has. */
static const void *arena_read(FILE *f, uint8_t *blob, size_t cap, size_t *used, size_t n)
{
    if (*used + n > cap) return NULL;
    uint8_t *p = blob + *used;
    if (read_exact(f, p, n) != 0) return NULL;
    *used += n;
    /* keep subsequent reads 4-byte aligned for the int32 arrays */
    while (*used % 4) {
        if (*used + 1 > cap) return NULL;
        (*used)++;
    }
    return p;
}

int tq_load(tq_model *m, const char *path)
{
    memset(m, 0, sizeof(*m));
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[4];
    if (read_exact(f, magic, 4) != 0 || memcmp(magic, "TQI1", 4) != 0) { fclose(f); return -2; }
    if (read_exact(f, &m->n_layers, 4) != 0) { fclose(f); return -3; }
    if (read_exact(f, &m->n_in, 4) != 0) { fclose(f); return -3; }
    if (read_exact(f, &m->in_scale, 4) != 0) { fclose(f); return -3; }
    if (read_exact(f, &m->weight_bits, 4) != 0) { fclose(f); return -3; }
    if (m->n_layers == 0 || m->n_layers > TQ_MAX_LAYERS) { fclose(f); return -4; }

    for (uint32_t i = 0; i < m->n_layers; i++) {
        uint32_t hdr[4];
        if (read_exact(f, hdr, sizeof(hdr)) != 0) { fclose(f); return -5; }
        tq_layer *l = &m->layers[i];
        l->n_in = hdr[0];
        l->n_out = hdr[1];
        l->relu = (uint8_t)hdr[2];
        if (l->n_out > TQ_MAX_WIDTH || l->n_in > TQ_MAX_WIDTH) { fclose(f); return -6; }

        l->w = arena_read(f, m->blob, TQ_MAX_WEIGHTS, &m->blob_used,
                          (size_t)l->n_in * l->n_out);
        l->bias = arena_read(f, m->blob, TQ_MAX_WEIGHTS, &m->blob_used, 4u * l->n_out);
        l->q_mult = arena_read(f, m->blob, TQ_MAX_WEIGHTS, &m->blob_used, 4u * l->n_out);
        l->lshift = arena_read(f, m->blob, TQ_MAX_WEIGHTS, &m->blob_used, l->n_out);
        l->rshift = arena_read(f, m->blob, TQ_MAX_WEIGHTS, &m->blob_used, l->n_out);
        if (!l->w || !l->bias || !l->q_mult || !l->lshift || !l->rshift) {
            fclose(f);
            return -7;
        }
    }
    fclose(f);
    return 0;
}

int tqf_load(tqf_model *m, const char *path)
{
    memset(m, 0, sizeof(*m));
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[4];
    if (read_exact(f, magic, 4) != 0 || memcmp(magic, "TQF1", 4) != 0) { fclose(f); return -2; }
    if (read_exact(f, &m->n_layers, 4) != 0) { fclose(f); return -3; }
    if (read_exact(f, &m->n_in, 4) != 0) { fclose(f); return -3; }
    if (m->n_layers == 0 || m->n_layers > TQ_MAX_LAYERS) { fclose(f); return -4; }

    for (uint32_t i = 0; i < m->n_layers; i++) {
        uint32_t hdr[4];
        if (read_exact(f, hdr, sizeof(hdr)) != 0) { fclose(f); return -5; }
        m->layers[i].n_in = hdr[0];
        m->layers[i].n_out = hdr[1];
        m->layers[i].relu = (uint8_t)hdr[2];
        if (hdr[1] > TQ_MAX_WIDTH || hdr[0] > TQ_MAX_WIDTH) { fclose(f); return -6; }
        m->layers[i].w = (const float *)arena_read(
            f, m->blob, TQ_MAX_WEIGHTS * 4, &m->blob_used, 4u * hdr[0] * hdr[1]);
        m->layers[i].bias = (const float *)arena_read(
            f, m->blob, TQ_MAX_WEIGHTS * 4, &m->blob_used, 4u * hdr[1]);
        if (!m->layers[i].w || !m->layers[i].bias) { fclose(f); return -7; }
    }
    fclose(f);
    return 0;
}

/* ----------------------------------------------------------------------- *
 * inference
 * ----------------------------------------------------------------------- */
void tq_forward(const tq_model *m, const int8_t *x, int8_t *out)
{
    /* Two ping-pong buffers on the stack: no heap, bounded by TQ_MAX_WIDTH. */
    int8_t buf_a[TQ_MAX_WIDTH], buf_b[TQ_MAX_WIDTH];
    int32_t acc[TQ_MAX_WIDTH];

    const int8_t *in = x;
    int8_t *dst = buf_a;

    for (uint32_t li = 0; li < m->n_layers; li++) {
        const tq_layer *l = &m->layers[li];

        for (uint32_t j = 0; j < l->n_out; j++) acc[j] = l->bias[j];

        /* Weights are (n_in, n_out) row-major, so the inner loop walks memory
         * contiguously and the accumulator stays hot. This is the layout that
         * matters more than any arithmetic trick at this size. */
        for (uint32_t i = 0; i < l->n_in; i++) {
            const int32_t xi = (int32_t)in[i];
#if TQ_SKIP_ZEROS
            /* Post-ReLU activations are frequently exactly zero, so the row can
             * be skipped entirely. Off by default: it makes latency depend on
             * the input, and leaving it off keeps the INT8-vs-FP32 comparison
             * in benchmarks/ honest, since the FP32 path has no such shortcut.
             * Build with -DTQ_SKIP_ZEROS=1 to measure it (bench reports both). */
            if (xi == 0) continue;
#endif
            const int8_t *wrow = l->w + (size_t)i * l->n_out;
            for (uint32_t j = 0; j < l->n_out; j++) {
                acc[j] += xi * (int32_t)wrow[j];
            }
        }

        const int32_t lo = l->relu ? 0 : -128;
        for (uint32_t j = 0; j < l->n_out; j++) {
            int32_t v = tq_requantize(acc[j], l->q_mult[j], l->lshift[j], l->rshift[j]);
            if (v < lo) v = lo;
            if (v > 127) v = 127;
            dst[j] = (int8_t)v;
        }

        in = dst;
        dst = (dst == buf_a) ? buf_b : buf_a;
    }

    const uint32_t n_last = m->layers[m->n_layers - 1].n_out;
    memcpy(out, in, n_last);
}

int tq_argmax_i8(const int8_t *v, uint32_t n)
{
    int best = 0;
    for (uint32_t i = 1; i < n; i++)
        if (v[i] > v[best]) best = (int)i;
    return best;
}

int tq_predict(const tq_model *m, const int8_t *x)
{
    int8_t out[TQ_MAX_WIDTH];
    tq_forward(m, x, out);
    return tq_argmax_i8(out, m->layers[m->n_layers - 1].n_out);
}

void tqf_forward(const tqf_model *m, const float *x, float *out)
{
    float buf_a[TQ_MAX_WIDTH], buf_b[TQ_MAX_WIDTH];
    const float *in = x;
    float *dst = buf_a;

    for (uint32_t li = 0; li < m->n_layers; li++) {
        const uint32_t n_in = m->layers[li].n_in, n_out = m->layers[li].n_out;
        for (uint32_t j = 0; j < n_out; j++) dst[j] = m->layers[li].bias[j];
        for (uint32_t i = 0; i < n_in; i++) {
            const float xi = in[i];
            const float *wrow = m->layers[li].w + (size_t)i * n_out;
            for (uint32_t j = 0; j < n_out; j++) dst[j] += xi * wrow[j];
        }
        if (m->layers[li].relu)
            for (uint32_t j = 0; j < n_out; j++)
                if (dst[j] < 0.0f) dst[j] = 0.0f;
        in = dst;
        dst = (dst == buf_a) ? buf_b : buf_a;
    }
    memcpy(out, in, sizeof(float) * m->layers[m->n_layers - 1].n_out);
}

int tqf_predict(const tqf_model *m, const float *x)
{
    float out[TQ_MAX_WIDTH];
    tqf_forward(m, x, out);
    const uint32_t n = m->layers[m->n_layers - 1].n_out;
    int best = 0;
    for (uint32_t i = 1; i < n; i++)
        if (out[i] > out[best]) best = (int)i;
    return best;
}
