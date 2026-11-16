# Findings — including the ones that made me wrong

A record of what I expected, what the measurements said, and what I changed as a
result. It exists because the useful part of an engineering project is usually
the part where the first idea failed, and that part is normally deleted before
publication.

Every number here is reproducible with the command given beneath it.

---

## 1. I expected INT8 to be faster. It was 29% slower.

**Hypothesis.** INT8 moves a quarter of the bytes of FP32 and uses cheaper
arithmetic, so the INT8 kernel should beat the FP32 kernel.

**Measurement.** Same model, same machine, both kernels written in the same style
to keep the comparison fair:

```
FP32  p50 0.778 µs
INT8  p50 1.103 µs      →  0.71× (INT8 slower)
```

**Status: refuted.** Reproduce with `make bench`.

---

## 2. "It must be unoptimised code." It was not.

**Hypothesis.** The slowdown is an artefact of compiling at a low optimisation
level; turning the optimiser up will close the gap.

**Measurement.** INT8 was slower at every optimisation level I tried, and
`-march=native` made it *worse*, not better:

| flags | INT8 p50 | FP32 p50 | speedup |
|---|---|---|---|
| `-O0` | 17.909 µs | 17.624 µs | 0.98× |
| `-O2` | 3.442 µs | 3.014 µs | 0.88× |
| `-O3` | 1.703 µs | 1.566 µs | 0.92× |
| `-Os` | 4.867 µs | 3.200 µs | 0.66× |
| `-O3 -march=native` | 1.094 µs | 0.757 µs | 0.69× |
| `-O3 -fno-tree-vectorize` | 3.058 µs | 3.000 µs | 0.98× |

**Status: refuted — and it pointed at the real cause.** The last row is the
informative one. With vectorisation *off*, the two kernels are within 2% of each
other. Turning vectorisation *on* is what hands FP32 the lead. So the FP32
advantage is not that float arithmetic is cheap; it is that the compiler packs 8
floats into an AVX register, while my INT8 path sign-extends each byte to int32
before accumulating and therefore moves the same number of lanes while also
paying for a fixed-point requantisation per output. The theoretical 4× INT8
advantage needs instructions that multiply-accumulate int8 natively (x86 VNNI,
ARM `SDOT`); the compiler will not invent those.

Reproduce with `make opt-levels`.

---

## 3. "Then INT8 is just worse." No — the win is memory-bound, and it shows up on cue.

**Hypothesis.** If the INT8 advantage is really about data movement, it should
be absent while the weights fit in cache and appear once they do not.

**Measurement.** Models of increasing width, median of three trials each:

| FP32 weight footprint | INT8 speedup | note |
|---|---|---|
| 35 KiB | 0.67× | fits in L1 (96 KiB) |
| 133 KiB | 0.98× | past L1 |
| 522 KiB | 1.03× | |
| 2.0 MiB | 1.14× | inside L2 (4 MiB) |
| 8.0 MiB | 1.47× | past L2 |
| 32 MiB | 1.43× | far past L2 |

**Status: supported.** The crossover lands where the cache hierarchy says it
should. This is the result I would defend in an interview: INT8 is a memory
optimisation, and on a model small enough to be cache-resident its benefits are
size and FPU-independence rather than latency.

Caveat I am not hiding: the mid-range points (133 KiB–522 KiB) showed up to 18%
spread across trials on a shared cloud vCPU, so I do not lean on them. The
endpoints were stable to under 1%, and the conclusion rests on those.

Reproduce with `make scaling`.

---

## 4. I built per-channel weight scaling before I could show it was needed.

**Hypothesis.** Per-channel weight scales will improve accuracy over a single
per-tensor scale.

**Measurement.** At INT8, they are *identical* — 98.61% either way. A reviewer
would be right to call that premature complexity. The feature only earns its
place under pressure:

| weight bits | per-tensor | per-channel | difference |
|---|---|---|---|
| 8 | 98.61% | 98.61% | none |
| 5 | 98.06% | 98.33% | +0.28pp |
| 4 | 98.06% | 98.61% | +0.56pp (lossless) |
| 2 | 58.61% | 70.00% | **+11.39pp** |

**Status: refuted at 8 bits, supported below 5.** I kept the feature and changed
the claim: it is not an accuracy win, it is insurance that pays out when you push
the bit-width down. Reproduce with `make sweep`.

---

## 5. I expected percentile calibration to beat min-max. It did not.

**Hypothesis.** Clipping activation ranges at the 99.9th percentile instead of
the absolute maximum will tighten the scales and improve accuracy.

**Measurement.** Across the whole bit-width sweep the two methods landed within a
fraction of a point of each other, trading places depending on the cell.

**Status: not supported on this model.** Plausible reason: this network's
activations have no long tail to clip — there is no outlier batch for min-max to
over-react to. I left both estimators implemented and both measured rather than
declaring a winner this model cannot demonstrate. On a network with activation
outliers I would expect a different answer, which is exactly why it is still in
the code.

---

## 6. Fuzzing found a real limitation: biases that do not fit in int32.

**Finding.** Of 500 randomly generated configurations, 84 were refused by the
quantizer, all for the same reason: `bias does not fit in int32`.

A bias is stored in accumulator units, scaled by `s_x · s_w`. When weights are
tiny (≈1e-4) and activations are large (≈50), that product is small enough that
an ordinary bias needs more than 31 bits to represent.

**What I changed.** Nothing about the refusal: the quantizer raises instead of
silently wrapping, and the fuzz report counts and names every refusal rather than
skipping it. A silent wrap here would produce a model that is subtly wrong in a
way accuracy testing on an easy task might never catch.

**What I did not do.** No trained model in this repo hits it — the weight and
activation scales that trigger it are pathological combinations the fuzzer
constructs on purpose. Handling it properly (rescaling the layer, or widening the
bias) is future work, and I would rather state that than ship a workaround I have
not measured. Reproduce with `make fuzz`.

---

## 7. Valgrind found two memory leaks. They were mine.

**Finding.** `valgrind --leak-check=full --errors-for-leak-kinds=all` reported
two definite leaks totalling 266,664 bytes — the model struct and the reference
buffer in my own test harnesses, never freed before exit.

**Status: fixed.** Both harnesses now release everything on every exit path,
including error paths, and the audit treats leaks as errors so a regression fails
the build. The engine itself allocates nothing, so it was never implicated — but
a harness that leaks makes the engine it tests harder to trust.

Reproduce with `make audit`.

---

## 8. My own claim-checker caught my benchmark protocol being unsound.

**Finding.** After an unrelated change I re-ran `make bench`, and
`tests/test_claims.py` failed on the claim *"INT8 is slower on the real model"* —
that run had reported INT8 **1.71× faster**.

The code had not become faster. Re-running the identical binary on identical
input produced these INT8/FP32 ratios:

```
0.479   0.698   0.702   ...   1.705
```

A single timing run on a shared 2-vCPU machine is not a measurement; the noise
was wide enough to invert the headline conclusion of the repository.

**What I changed.** Not the claim — the protocol. `make bench` now runs the
benchmark 7 times and reports the median, keeping every trial's ratio in
`artifacts/latency.json` so the spread stays visible. The median across 7 trials
(0.705×) agrees with the original figure to three decimal places; the difference
is that it is now reproducible rather than lucky.

**Why this is in the findings rather than quietly fixed.** It is the clearest
evidence I have that the claim-checking in this repo does real work. A README
with hand-copied numbers would have shipped the 1.71× figure as a win, and it
would have been wrong.

Reproduce with `make bench`.

---

## 9. The bias fix was wrong the first time, and inference caught it.

**Finding (§6 continued).** Fuzzing refused 84 of 500 configurations because
biases did not fit in int32. The fix: instead of refusing, coarsen the weight
scale for the affected output channels just enough that the bias fits — trading
weight resolution, which costs nothing on a trained model, for a representable
bias.

**It did not work.** Quantization then succeeded, and *inference* overflowed.

The reason is worth stating plainly, because it is the kind of thing that is
obvious only afterwards: the accumulator holds `bias + sum(x_i * w_i)`. Fitting
the bias at exactly `INT32_MAX` leaves nothing for the sum, so the layer
overflows the moment a real input arrives. The bias budget has to reserve the
worst-case dot product — `n_in * 127 * wmax`, every activation and weight
saturated with the same sign — before the bias is allowed to use what is left.

On the trained model that reservation is about 0.05% of the int32 range and
changes nothing: zero channels are rescaled and accuracy is identical to the
digit.

**Result.** Refusals across 500 random configurations:

```
before the fix            84   (all: bias does not fit in int32)
after the naive fix        3   (all: int32 accumulator overflow)
after reserving headroom   0
```

500 of 500 now quantize, and all 500 are bit-exact against the C engine.

**Why it is in the findings.** My first fix turned one overflow into another one
further down the pipeline. The only reason that did not ship is that the
reference implementation asserts its accumulator bounds instead of trusting
them. Reproduce with `make fuzz`.

---

## 10. Per-channel scaling matters roughly three times more for convolutions.

**Hypothesis.** The dense sweep showed per-channel scaling buying nothing until
5 bits and +11.4pp at 2 bits. Convolutions should behave similarly.

**Measurement.** They do not. The same sweep on the conv model:

| weight bits | dense advantage | conv advantage |
|---|---|---|
| 8 | 0.00pp | −0.28pp |
| 4 | +0.56pp | +1.11pp |
| 2 | +11.39pp | **+39.17pp** |

At 2 bits per-tensor scaling collapses the CNN to 32.78% — barely better than
guessing three classes out of ten — while per-channel holds 71.94%.

**Status: refuted, in the useful direction.** It follows from what a conv kernel
is. Each output channel is a filter with its own dynamic range, and per-tensor
scaling forces every filter in the stack to share one scale, so the quiet
filters quantize to nearly nothing. A dense layer's columns are less dissimilar.
I would now expect the effect to be larger still for depthwise-separable
convolutions, which is the obvious next measurement rather than an assertion.

**Caveat I am keeping attached to this.** The test set is 360 samples, so one
sample is 0.28pp. The 8-bit and 5-bit rows are inside noise and should be read
as "no detectable difference". Only the 4-bit and 2-bit rows carry the claim.

Reproduce with `make sweep-cnn`.

---

## What this adds up to

The headline claim of this repo is not "I made inference faster." It is: I built
the pipeline, measured it properly, got a result I did not want, found the actual
mechanism, and showed the conditions under which the optimisation does and does
not pay. Both engines are bit-exact across four instruction sets, two compilers and five
optimisation levels, allocation-free on the inference path, and clean under
valgrind — which is what makes the negative result worth anything. A slow kernel
that is also wrong teaches nobody anything.

Four of the ten findings here are mistakes of mine that measurement caught: a
benchmark protocol that reported a 1.71x speedup from noise, two memory leaks, a
bias fix that moved the overflow instead of removing it, and a feature I built
before I could show it was needed. That ratio is not an accident of this project.
It is what the work looks like when the checking is real.
