/* tinyq_graph -- integer-only INT8 inference for small conv stacks.
 *
 * Same arithmetic as tinyq.h (symmetric INT8, int32 accumulation, Q0.31
 * fixed-point requantization, no floating point, no heap) extended to layers
 * that are convolutions. Activations are NHWC; conv weights are
 * (kh, kw, c_in, c_out), so the innermost loop walks output channels
 * contiguously -- the same locality argument as the dense kernel.
 *
 * Padding is 'valid', stride 1.
 */
#ifndef TINYQ_GRAPH_H
#define TINYQ_GRAPH_H

#include <stdint.h>
#include <stddef.h>

#include "tinyq.h" /* for tq_requantize and the TQ_MAX_* knobs */

#ifndef TQG_MAX_LAYERS
#define TQG_MAX_LAYERS 16
#endif
#ifndef TQG_MAX_TENSOR
#define TQG_MAX_TENSOR 8192 /* int8 elements in any one activation tensor */
#endif
#ifndef TQG_MAX_BLOB
#define TQG_MAX_BLOB 1048576
#endif

#define TQG_KIND_CONV 0
#define TQG_KIND_DENSE 1

typedef struct {
    uint32_t kind, relu;
    uint32_t kh, kw, c_in, c_out; /* dense uses kh=kw=1, c_in=n_in, c_out=n_out */
    const int8_t *w;
    const int32_t *bias;
    const int32_t *q_mult;
    const int8_t *lshift;
    const int8_t *rshift;
} tqg_layer;

typedef struct {
    uint32_t n_layers, in_h, in_w, in_c, weight_bits;
    float in_scale; /* input quantization only; never used during inference */
    tqg_layer layers[TQG_MAX_LAYERS];
    uint8_t blob[TQG_MAX_BLOB];
    size_t blob_used;
} tqg_model;

int tqg_load(tqg_model *m, const char *path);

/* Runs the whole graph. `out` receives the final layer's c_out int8 values. */
void tqg_forward(const tqg_model *m, const int8_t *x, int8_t *out);
int tqg_predict(const tqg_model *m, const int8_t *x);
uint32_t tqg_output_size(const tqg_model *m);

#endif /* TINYQ_GRAPH_H */
