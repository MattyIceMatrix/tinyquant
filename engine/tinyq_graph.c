#include "tinyq_graph.h"

#include <stdio.h>
#include <string.h>

static int read_exact(FILE *f, void *dst, size_t n)
{
    return fread(dst, 1, n, f) == n ? 0 : -1;
}

static const void *arena_read(FILE *f, uint8_t *blob, size_t cap, size_t *used, size_t n)
{
    if (*used + n > cap) return NULL;
    uint8_t *p = blob + *used;
    if (read_exact(f, p, n) != 0) return NULL;
    *used += n;
    while (*used % 4) {
        if (*used + 1 > cap) return NULL;
        (*used)++;
    }
    return p;
}

int tqg_load(tqg_model *m, const char *path)
{
    memset(m, 0, sizeof(*m));
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[4];
    if (read_exact(f, magic, 4) != 0 || memcmp(magic, "TQG1", 4) != 0) { fclose(f); return -2; }
    uint32_t hdr[5];
    if (read_exact(f, hdr, sizeof(hdr)) != 0) { fclose(f); return -3; }
    m->n_layers = hdr[0];
    m->in_h = hdr[1];
    m->in_w = hdr[2];
    m->in_c = hdr[3];
    m->weight_bits = hdr[4];
    if (read_exact(f, &m->in_scale, 4) != 0) { fclose(f); return -3; }
    if (m->n_layers == 0 || m->n_layers > TQG_MAX_LAYERS) { fclose(f); return -4; }

    for (uint32_t i = 0; i < m->n_layers; i++) {
        uint32_t lh[6];
        if (read_exact(f, lh, sizeof(lh)) != 0) { fclose(f); return -5; }
        tqg_layer *l = &m->layers[i];
        l->kind = lh[0];
        l->relu = lh[1];
        l->kh = lh[2];
        l->kw = lh[3];
        l->c_in = lh[4];
        l->c_out = lh[5];
        if (l->kind != TQG_KIND_CONV && l->kind != TQG_KIND_DENSE) { fclose(f); return -6; }
        if (l->c_out == 0 || l->c_out > TQG_MAX_TENSOR) { fclose(f); return -6; }

        const size_t wcount = (size_t)l->kh * l->kw * l->c_in * l->c_out;
        l->w = arena_read(f, m->blob, TQG_MAX_BLOB, &m->blob_used, wcount);
        l->bias = arena_read(f, m->blob, TQG_MAX_BLOB, &m->blob_used, 4u * l->c_out);
        l->q_mult = arena_read(f, m->blob, TQG_MAX_BLOB, &m->blob_used, 4u * l->c_out);
        l->lshift = arena_read(f, m->blob, TQG_MAX_BLOB, &m->blob_used, l->c_out);
        l->rshift = arena_read(f, m->blob, TQG_MAX_BLOB, &m->blob_used, l->c_out);
        if (!l->w || !l->bias || !l->q_mult || !l->lshift || !l->rshift) { fclose(f); return -7; }
    }
    fclose(f);
    return 0;
}

uint32_t tqg_output_size(const tqg_model *m)
{
    return m->layers[m->n_layers - 1].c_out;
}

void tqg_forward(const tqg_model *m, const int8_t *x, int8_t *out)
{
    /* Two fixed activation buffers, ping-ponged. No heap, no VLAs. */
    static int8_t buf_a[TQG_MAX_TENSOR];
    static int8_t buf_b[TQG_MAX_TENSOR];
    int32_t acc[TQG_MAX_TENSOR];

    const int8_t *in = x;
    int8_t *dst = buf_a;
    uint32_t h = m->in_h, w = m->in_w, c = m->in_c;

    for (uint32_t li = 0; li < m->n_layers; li++) {
        const tqg_layer *l = &m->layers[li];
        uint32_t oh, ow, oc = l->c_out;

        if (l->kind == TQG_KIND_CONV) {
            oh = h - l->kh + 1;
            ow = w - l->kw + 1;

            for (uint32_t oy = 0; oy < oh; oy++) {
                for (uint32_t ox = 0; ox < ow; ox++) {
                    int32_t *a = acc + (size_t)(oy * ow + ox) * oc;
                    for (uint32_t j = 0; j < oc; j++) a[j] = l->bias[j];

                    for (uint32_t ky = 0; ky < l->kh; ky++) {
                        for (uint32_t kx = 0; kx < l->kw; kx++) {
                            const int8_t *xrow =
                                in + (((size_t)(oy + ky) * w) + (ox + kx)) * c;
                            /* weights are (kh, kw, c_in, c_out) row-major */
                            const int8_t *wbase =
                                l->w + ((((size_t)ky * l->kw) + kx) * l->c_in) * oc;
                            for (uint32_t ci = 0; ci < c; ci++) {
                                const int32_t xv = (int32_t)xrow[ci];
                                const int8_t *wrow = wbase + (size_t)ci * oc;
                                for (uint32_t j = 0; j < oc; j++)
                                    a[j] += xv * (int32_t)wrow[j];
                            }
                        }
                    }
                }
            }
        } else { /* dense over the flattened activation, which is already HWC */
            oh = 1;
            ow = 1;
            const uint32_t n_in = l->c_in;
            for (uint32_t j = 0; j < oc; j++) acc[j] = l->bias[j];
            for (uint32_t i = 0; i < n_in; i++) {
                const int32_t xv = (int32_t)in[i];
                const int8_t *wrow = l->w + (size_t)i * oc;
                for (uint32_t j = 0; j < oc; j++) acc[j] += xv * (int32_t)wrow[j];
            }
        }

        const int32_t lo = l->relu ? 0 : -128;
        const size_t n_out_total = (size_t)oh * ow * oc;
        for (size_t idx = 0; idx < n_out_total; idx++) {
            const uint32_t j = (uint32_t)(idx % oc);
            int32_t v = tq_requantize(acc[idx], l->q_mult[j], l->lshift[j], l->rshift[j]);
            if (v < lo) v = lo;
            if (v > 127) v = 127;
            dst[idx] = (int8_t)v;
        }

        in = dst;
        dst = (dst == buf_a) ? buf_b : buf_a;
        h = oh;
        w = ow;
        c = oc;
    }

    memcpy(out, in, tqg_output_size(m));
}

int tqg_predict(const tqg_model *m, const int8_t *x)
{
    int8_t out[TQG_MAX_TENSOR];
    tqg_forward(m, x, out);
    const uint32_t n = tqg_output_size(m);
    int best = 0;
    for (uint32_t i = 1; i < n; i++)
        if (out[i] > out[best]) best = (int)i;
    return best;
}
