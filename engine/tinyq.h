/* tinyq -- integer-only INT8 MLP inference for constrained targets.
 *
 * Properties this header promises and the test suite checks:
 *   - C99, no dependencies beyond libc stdio/string for model loading
 *   - no heap allocation anywhere, including load (fixed static arenas)
 *   - no floating point on the INT8 inference path
 *   - bit-identical outputs to the Python reference in tinyquant/
 */
#ifndef TINYQ_H
#define TINYQ_H

#include <stdint.h>
#include <stddef.h>

/* Arena sizes are compile-time knobs so the same engine can be built for a
 * microcontroller (tiny arenas, nothing dynamic) or for the scaling study in
 * benchmarks/, which needs multi-megabyte models. Override with -D. */
#ifndef TQ_MAX_LAYERS
#define TQ_MAX_LAYERS 16
#endif
#ifndef TQ_MAX_WIDTH
#define TQ_MAX_WIDTH 512
#endif
#ifndef TQ_MAX_WEIGHTS
#define TQ_MAX_WEIGHTS 262144 /* bytes of weight storage across all layers */
#endif

typedef struct {
    uint32_t n_in, n_out;
    uint8_t relu;
    const int8_t *w;        /* (n_in, n_out) row-major */
    const int32_t *bias;    /* n_out */
    const int32_t *q_mult;  /* n_out, Q0.31 */
    const int8_t *lshift;   /* n_out */
    const int8_t *rshift;   /* n_out */
} tq_layer;

typedef struct {
    uint32_t n_layers, n_in, weight_bits;
    float in_scale;         /* only used to quantize inputs, never during inference */
    tq_layer layers[TQ_MAX_LAYERS];
    uint8_t blob[TQ_MAX_WEIGHTS];
    size_t blob_used;
} tq_model;

typedef struct {
    uint32_t n_layers, n_in;
    struct {
        uint32_t n_in, n_out;
        uint8_t relu;
        const float *w, *bias;
    } layers[TQ_MAX_LAYERS];
    uint8_t blob[TQ_MAX_WEIGHTS * 4];
    size_t blob_used;
} tqf_model;

/* 0 on success, negative on error. */
int tq_load(tq_model *m, const char *path);
int tqf_load(tqf_model *m, const char *path);

/* Integer-only forward pass. out must hold n_out int8 values of the last layer. */
void tq_forward(const tq_model *m, const int8_t *x, int8_t *out);
int tq_argmax_i8(const int8_t *v, uint32_t n);
int tq_predict(const tq_model *m, const int8_t *x);

void tqf_forward(const tqf_model *m, const float *x, float *out);
int tqf_predict(const tqf_model *m, const float *x);

/* Exposed for the cross-validation test. */
int32_t tq_requantize(int32_t acc, int32_t q_mult, int lshift, int rshift);

#endif /* TINYQ_H */
