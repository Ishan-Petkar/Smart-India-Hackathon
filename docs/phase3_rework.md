# Phase 3 Review: Synthetic Cloud Augmentation Engine — TA-MMLD

**Role:** Senior ML/research architect review. No code, no repo edits — design only, ready for a coding agent to implement without further architectural decisions.

---

## 1. VERDICT

The "SCL-Guided Stochastic Hybrid" direction is **fundamentally correct and an improvement over the original Method C**, but as specified it is **not implementation-ready**. It correctly demotes Perlin noise from primary geometry generator to perturbation, which fixes the single biggest risk in the original plan (a model that keys on "Perlin texture = cloud" rather than actual cloud statistics). However:

- The **train/val/test template-leakage question is unresolved** and, as written, would very likely leak.
- The **coverage distribution is asserted, not derived** — no justification tying it to real Himalayan cloud climatology or to what Phase 5/6 need.
- The **shadow model is underspecified physically** and needs an explicit, honestly-labeled approximation rather than an implied physical model.
- The **radiometric model is not yet fixed** — "spectrally flat" is rejected but nothing replaces it.
- **Phase 5 difficulty calibration is not addressed** — nothing in the design currently prevents trivially easy segmentation.

Net: adopt the hybrid direction, but tighten it into one locked specification (given at the end) before any code is written.

---

## 2. MAJOR PROBLEMS

1. **Geographic/template leakage (critical).** SCL cloud shapes are extracted from the *same 14 scenes* that get split into train/val/test. If a cloud template extracted from a val-scene's SCL band can be pasted onto a train-scene's LISS-IV patch (or vice versa), the "held-out" scenes are not independent — the network can be evaluated on cloud shapes it has effectively seen. This must be fixed at the template-library level, not just the patch level.
2. **Coverage distribution has no derivation.** The proposed 20/30/30/20 bucketing is plausible-looking but arbitrary. It should be justified against (a) what fraction of the *deployment* scenes will actually be cloudy at what density, if any prior estimate exists, and (b) curriculum needs of Phase 6 (which already defines its own curriculum in Phase 6.6 — 3.9 and 6.6 must not silently disagree).
3. **Double-counting of curriculum control.** Phase 6.6 already specifies a training curriculum by epoch (10–30% → 20–60% → 40–80%). If Phase 3's generator also imposes its own fixed coverage distribution, the two schedules will conflict unless Phase 3 exposes coverage as a *parameter* the training loop controls, not a distribution baked into the generator's default sampling.
4. **Binary vs. continuous mask ambiguity.** The plan says "generate continuous opacity mask" but Phase 5 target (5.2) is described as "known synthetic cloud mask" for a segmentation model, which conventionally wants a binary or thresholded target. These two must be reconciled: keep one continuous ground-truth field, derive a binary Phase-5 label from it via a defined threshold, don't maintain two independent representations.
5. **Shadow physics oversold.** LISS-IV alone will not reliably provide cloud-base height, and true shadow projection requires cloud height, sun geometry, and terrain elevation at the shadow-landing point. Sun elevation/azimuth *can* be computed exactly (scene time + geolocation), but cloud height cannot be inferred from a single VNIR image. The design must explicitly state this is a **stochastic geometric approximation**, not a physical shadow model, or the eventual paper/report will overclaim.
6. **Cross-sensor template transfer is asserted without a validity boundary.** SCL is 20m Sentinel-2, LISS-IV is 5m. A cloud "shape" carried across a 4x resolution gap, a different sensor PSF, and typically a different acquisition date is a *shape prior*, not ground truth — fine, but the spec needs explicit language forbidding any claim that SCL-derived masks equal true cloud presence in the LISS-IV scene.
7. **Class 10 (thin cirrus) is spectrally and geometrically distinct from 8/9 (cloud)** and should not be pooled into one template library with the same opacity model — cirrus templates map naturally to the "thin" thickness regime, 8/9 map to moderate/thick.
8. **Small/fragmented SCL connected components are a known SCL failure mode** (salt-and-pepper misclassification, especially near cloud edges and over bright/snow terrain). Naively extracting all connected components will pollute the template library with non-cloud artifacts. This needs explicit size/shape filtering.
9. **Snow/cloud confusion is a live risk for this specific project** (Himalayan terrain, explicitly called out in Phase 5.5 "false positive < 5%: don't mask snow as cloud"). If synthetic cloud texture is spectrally close to snow/ice reflectance and templates aren't checked against known snow-covered scene regions, Phase 5 training data will actively reinforce the snow/cloud confusion it's supposed to avoid.
10. **Phase 5 could become trivially easy** if synthetic edges are too clean, textures too uniform, or the same finite template library is reused per-epoch without enough stochastic augmentation — the segmentation model would learn "detect this specific synthetic signature," not "detect clouds."
11. **Phase 6 adequacy is not automatically true.** A visually convincing cloudy image is not the same as an information-destruction process matched to what real clouds do to a diffusion model's job. Thick-cloud interiors must destroy essentially all signal (opacity → ~1, texture variance → ~0); thin cirrus must leave a recoverable, attenuated, *not* zeroed-out surface signal; if the generator makes "thick" and "thin" too similar, Phase 6 will not learn a useful dependence on the opacity field.
12. **Reproducibility of "failed samples" is mentioned but not defined** — the spec needs a hard boundary between what counts as a rejected sample (and how it's logged / retried with a new seed) vs. an accepted one.
13. **Batched/runtime performance is a real constraint** the design doesn't budget for. If template retrieval, morphological ops, and Perlin field generation aren't vectorizable/cacheable, this becomes a training bottleneck at Phase 6 scale (batch 4–8, hundreds of epochs).

---

## 3. FINAL RECOMMENDED ARCHITECTURE

```
Offline (once, before training):
  SCL rasters (all scenes) 
      → per-scene cloud-class extraction (classes 8, 9 separate from 10)
      → connected-component extraction + size/shape filtering
      → per-template metadata (source scene, split assignment, class, area, elongation)
      → Template Library, partitioned by split (train/val/test) at scene level

Runtime (per training step, given a clear LISS-IV patch + its scene's split):
  1. Sample cloud regime (thin | moderate | thick | dense) — see §5
  2. Sample coverage target within that regime's range
  3. Draw N candidate templates ONLY from the template library partition
     matching the patch's split
  4. Apply stochastic geometric transform (rotate/flip/scale/translate/
     dilate-erode/local warp) to get base binary geometry
  5. Composite candidates (union/blend) until target coverage reached
  6. Perlin/simplex perturbation of the boundary + interior density
     (secondary role only — see §8) → continuous opacity mask M_cloud ∈ [0,1]
  7. Generate cloud texture conditioned on regime (see §5)
  8. Compute sun elevation/azimuth from scene metadata; generate shadow
     mask as a stochastic offset of M_cloud (see §6)
  9. Composite: clear patch → cloudy patch (see formula in §5)
  10. Return cloudy_patch, M_cloud, M_shadow, opacity stats, metadata dict
```

Key structural change from the original proposal: **the template library is split-partitioned before any sampling happens**, and coverage/regime sampling is a **caller-provided parameter**, not an internal fixed distribution — Phase 6's curriculum scheduler drives it.

---

## 4. CLOUD MASK REPRESENTATION

Use a **single continuous opacity mask** `M_cloud ∈ [0,1]` as the canonical ground truth. Do not maintain a separate binary mask as a distinct object.

- Phase 3 / Phase 6 (diffusion conditioning): use `M_cloud` directly (continuous) — diffusion benefits from graded corruption strength as a conditioning signal.
- Phase 5 (segmentation target): derive a binary label via a **documented, fixed threshold** (e.g., `M_cloud > 0.15` = cloud-present, chosen to include thin cirrus edges) applied at data-loading time, not at generation time. This keeps one source of truth.
- Also emit **effective coverage** as `mean(M_cloud)` over the patch for logging/curriculum control (not `mean(binary mask)` — a continuous measure is more informative and matches the "coverage measured from opacity mask" option raised in the prompt).

---

## 5. CLOUD TEXTURE / RADIOMETRIC MODEL

**Reject pure spectral flatness.** Real clouds over VNIR are close to spectrally flat *for thick cloud tops*, but:
- Thin cirrus is genuinely semi-transparent — the underlying surface signal must partially survive, and per-band transmittance is not identical (NIR tends to be attenuated somewhat differently than visible under thin cirrus, though the effect is modest at LISS-IV band widths).
- Cloud base/side texture under moderate thickness often has faint band-to-band structure from illumination geometry, not just a flat white value.

Recommended per-regime formulation, using per-pixel opacity `a = A_opacity * M_cloud`:

```
thin (a ~ 0.05–0.35):
    X = Y_clear * (1 - k_thin * a) + cloud_color * (k_thin * a)
    # k_thin < 1: attenuation, NOT full replacement — surface stays partially visible
    # cloud_color: near-flat but with small per-band jitter (~±3-5% reflectance)

moderate (a ~ 0.35–0.65):
    X = Y_clear * (1 - a) + cloud_color * a
    # standard linear mix, cloud_color flat with small jitter

thick/dense (a ~ 0.65–1.0):
    X = cloud_color  (with a→1 asymptotically; residual surface signal → ~0)
    # texture itself gets slight spatial noise so thick-cloud regions
    # are not a single constant value (that would be an easy-to-detect
    # "obviously fake" flat patch and a shortcut for Phase 5)
```

`cloud_color` should be sampled from a plausible bright-reflectance range for each band with small correlated jitter (not independent per-band noise, which would look unrealistic) — this can be calibrated later against real bright/cloud pixels already observed in the S2 SCL-flagged regions of the same scenes, but do not hardcode this to a single constant.

---

## 6. SHADOW MODEL

Recommendation: an explicit **stochastic geometric approximation**, documented as such — not a physically exact shadow model.

- **Use real sun elevation/azimuth** (computable exactly from scene acquisition time + lat/lon — cheap and correct, no reason not to use it).
- **Do not attempt real cloud-height-based projection.** LISS-IV/S2/S1 do not give per-cloud height; any height value would be fabricated. Instead:
  - Compute displacement **direction** from real sun azimuth (this part is physically grounded).
  - Sample displacement **magnitude** stochastically from a bounded range calibrated to typical patch scale (e.g., a few percent to ~20% of patch width), not from a fabricated height/elevation-angle formula presented as physical.
  - Optionally weight magnitude by cloud regime (thicker/taller-looking clouds → slightly larger stochastic range), but label this as a heuristic, not physics.
- Shadow opacity: partial, not equal to cloud opacity — shadows are gentler attenuation (e.g., 15–40% darkening of clear reflectance), and should decay at shadow edges (soft mask, not hard-edged).
- Terrain interaction: optional refinement (steep slopes distort real shadow shape) — flag as a "nice to have," not required for v1; do not silently claim terrain-correct shadows if not implemented.
- **Explicitly do not claim** cloud-height accuracy, physically exact projection, or terrain-corrected shadow casting unless actually implemented — this matters for §16 (defensibility).

---

## 7. SCL TEMPLATE STRATEGY

- **Classes used:** 8 (cloud medium) and 9 (cloud high) go into one "cloud" template pool; **10 (thin cirrus) is a separate pool**, mapped preferentially to the "thin" radiometric regime. Do not merge 8/9/10 into one undifferentiated pool.
- **Filtering:** discard connected components below a minimum pixel-area threshold (removes SCL salt-and-pepper noise) and reject extreme-aspect-ratio slivers unless explicitly kept as a "wispy cirrus" sub-category.
- **Cleanup:** apply light morphological closing to fill small holes from SCL misclassification before storing a template, but cap how much augmentation-time dilation/erosion is applied later so shapes don't degrade into blobs.
- **Snow safeguard:** cross-check candidate cloud templates against any available snow/ice indicator for that scene (if SCL or elevation/known-snowline data lets you flag high-altitude bright regions) and down-weight or exclude templates whose source pixels are suspicious in this regard — directly protects Phase 5's stated snow/cloud false-positive requirement.
- **Train/val/test isolation (hard requirement):** partition the **source scenes** into train/val/test *before* any template extraction, then build three independent template libraries, one per split, from only that split's scenes. A validation-scene's SCL cloud shape must never be composited onto a training patch, and vice versa. This is the single most important correctness fix relative to the original proposal.
- Cross-sensor transfer is treated explicitly as **shape-prior only** — resample templates to the 5m grid at use-time; never claim positional/radiometric equivalence to actual LISS-IV cloud pixels.

---

## 8. PERLIN/SIMPLEX ROLE

Keep it, but **strictly as secondary perturbation**, confirming the proposal's instinct:
- Boundary irregularity / edge softening (blur + noise-modulated alpha at the mask boundary, avoiding hard cut-paste edges).
- Interior opacity variation within an already-SCL-shaped mask (breaks up flat-looking interiors, especially in the thick regime where a constant `cloud_color` would otherwise be a Phase-5 shortcut).
- Small-scale texture modulation of `cloud_color` (not primary color source, just jitter).

**Do not** use Perlin/simplex to generate cloud geometry, ever, even as a fallback — this reintroduces exactly the shortcut risk the review is meant to eliminate. If a scene has too few usable templates (e.g., very few SCL-flagged pixels in that split), the correct fallback is to reuse/re-augment existing templates more aggressively (more geometric transform variety), not to fall back to procedural noise-only geometry.

---

## 9. COVERAGE DISTRIBUTION

Do not bake a fixed distribution into the generator. Instead:

- **Expose coverage as a required caller parameter** (single value or range) — the generator samples within whatever bounds it's given.
- Phase 6's existing curriculum (6.6: warmup 10–30%, core 20–60%, hardening 40–80%) becomes the *only* place coverage scheduling logic lives. Phase 3 should not duplicate or silently contradict it.
- For Phase 5 (segmentation) training, which isn't epoch-curriculum-scheduled in the same way, use a distribution roughly uniform over 10–80% with mild oversampling of the low end (say 15–20% weight on 10–25% coverage) since near-clear scenes with small cloud fragments are the hardest real-world detection case and most valuable for reducing false negatives — this is a defensible, stated choice, not the original 20/30/30/20 guess, and should be revisited once real validation metrics are available.
- Coverage should be measured from the **continuous opacity mask's mean**, not binary geometry — this is more meaningful and avoids ambiguity between "geometrically covered" and "effectively obscured."

---

## 10. RANDOMIZATION / REPRODUCIBILITY

- Single top-level `seed` argument to `generate()`; internally derive independent sub-seeds (e.g., via `numpy.random.SeedSequence`) for template selection, geometric transform, Perlin perturbation, texture jitter, and shadow displacement, so changing one sub-process's random draws doesn't silently change another's.
- `seed=None` → nondeterministic (for actual training throughput); explicit `seed=<int>` → fully reproducible, including template choice.
- Every call returns/logs a metadata dict (see §14) sufficient to exactly reproduce the sample given the same template library version.
- "Reproducibility of failed samples": define a **rejection criterion** up front (e.g., resulting coverage more than X% off target after N compositing attempts, or emptied mask due to over-aggressive erosion) — on rejection, log the attempted parameters and seed, then resample with a derived seed; never silently retry with an unlogged seed.

---

## 11. VALIDATION TEST SUITE

Expand the proposed 7 categories with:

8. **Split-isolation test (new, critical):** automated check that no template in the val or test library shares a source-scene ID with any train-library template; run this as a CI-style assertion, not just a manual QA step.
9. **Regime-consistency test:** confirm sampled `A_opacity` values fall within the intended thin/moderate/thick/dense bounds for the requested regime, and that thin-regime samples retain measurable correlation with the underlying clear patch (e.g., SSIM between cloudy and clear pixels within thin-mask regions stays above a floor) while thick-regime samples do not.
10. **Snow-confusion spot check:** run the generator on scenes/patches known or suspected to contain snow and visually/statistically confirm cloud placement isn't concentrated on snow-bright regions in a way that would teach false associations.
11. **Determinism test:** same seed + same inputs → bit-identical output, across two separate calls/processes.
12. **Throughput/perf test:** measure wall-clock per-patch generation time at target batch size; assert against a training-loop budget (define the budget once GPU pipeline timing is known).
13. **Anti-shortcut test:** train a trivial classifier (or just check simple statistics) that distinguishes "is this a synthetic cloud edge" purely from low-level cues (e.g., exact opacity-field zero-crossings); if trivially separable, edge/texture generation needs more stochastic diversity.

---

## 12. PHASE 5 COMPATIBILITY

Concrete changes to reduce the "too easy" risk:
- Never use a single deterministic `cloud_color` constant across a whole dataset epoch — jitter it per-sample.
- Always apply boundary softening (§8) — no hard binary paste edges in the *default* generation path (reserve fully hard edges, if ever used, for a small minority of "opaque, sharp-edge" cases that mimic genuinely hard-edged real cumulus, not as the default).
- Ensure template diversity is large relative to epoch count seen by Phase 5 (reuse the same base template with materially different geometric transforms + Perlin perturbation each time, rather than a small fixed set of finished masks).
- Include the snow-safeguard from §7 specifically because Phase 5's own target metric calls out snow/cloud false positives.
- Periodically evaluate Phase 5 on any real (non-synthetic) cloud examples available, even a small hand-labeled set, to catch synthetic-to-real gap early rather than trusting synthetic validation alone.

---

## 13. PHASE 6 COMPATIBILITY

The corruption process needs to actually destroy information proportionally to what Phase 6 must reconstruct:
- Thick/dense regions must approach **true information loss** (opacity → ~1, negligible correlation with `Y_clear`) — otherwise the diffusion model can cheat by learning a shortcut inverse of the corruption function rather than genuine reconstruction/hallucination from multi-modal context (SAR, S2, terrain).
- Thin cirrus must leave **partial, learnable signal** — this is the case that validates that S1/S2/DEM conditioning is even necessary; if thin cases are too easy (or too hard) relative to real cirrus attenuation, the loss landscape in Phase 6.6's "warmup" stage will not transfer to real thin-cloud cases.
- Boundary transition zones (mixed cloud/terrain) are exactly where per-pixel opacity blending matters most — a hard binary mask there would train the model on unrealistic "always fully clear or fully occluded" statistics, most damaging to the SSIM/perceptual losses in Phase 6.5.
- Spatial-frequency realism: real cloud opacity fields have specific spatial correlation (structured, multi-scale) — pure high-frequency Perlin noise as the dominant boundary/interior signal would push the reconstruction model toward denoising a "noise" pattern rather than an "occlusion" pattern; the SCL-shape-first + Perlin-as-perturbation-only design directly addresses this, but should be spot-checked (§11.7 visual QA, §11.9 anti-shortcut test) rather than assumed correct by design.
- Do not assume "visually convincing to a human" implies "useful gradient signal for the diffusion loss" — this is exactly why §11's statistical/anti-shortcut tests matter more than visual QA alone.

---

## 14. FINAL IMPLEMENTATION SPECIFICATION

```python
# scripts/synthetic_clouds.py

class CloudTemplateLibrary:
    """Built once offline, partitioned by split."""
    def __init__(self, split: Literal["train", "val", "test"]): ...
    @classmethod
    def build_from_scl(cls, scl_paths_by_scene: dict[str, str],
                        scene_split_map: dict[str, str],
                        min_component_area_px: int,
                        max_aspect_ratio: float) -> dict[str, "CloudTemplateLibrary"]:
        """Returns {'train': lib, 'val': lib, 'test': lib}, each built
        ONLY from that split's scenes. Separately stores class-10
        (cirrus) templates from class-8/9 (cloud) templates."""

class CloudGenerator:
    def __init__(self, template_library: CloudTemplateLibrary,
                 config: CloudGenConfig): ...

    def generate(
        self,
        clear_patch: np.ndarray,        # (3, H, W) Green/Red/NIR, [0,1]
        coverage: float | tuple[float, float],  # required, caller-set
        regime: Literal["thin", "moderate", "thick", "dense", "mixed"] = "mixed",
        sun_azimuth_deg: float | None = None,   # from scene metadata
        sun_elevation_deg: float | None = None,
        seed: int | None = None,
    ) -> CloudSample:
        ...

@dataclass
class CloudSample:
    cloudy_patch: np.ndarray     # (3, H, W)
    cloud_mask: np.ndarray       # (H, W) continuous opacity M_cloud in [0,1]
    shadow_mask: np.ndarray      # (H, W) continuous in [0,1]
    effective_coverage: float    # mean(M_cloud)
    metadata: CloudSampleMetadata

@dataclass
class CloudSampleMetadata:
    seed: int
    template_ids: list[str]
    regime: str
    target_coverage: float
    effective_coverage: float
    opacity_params: dict          # per-regime a_min/a_max used
    transform_params: dict        # rotation, scale, flip, dilation/erosion, warp params per template
    perlin_params: dict           # octaves, frequency, blend weight
    shadow_params: dict           # azimuth used, displacement magnitude, opacity
    rejected_attempts: int        # count of resampling due to rejection criteria
```

Config parameters to expose (not hardcode): `min_component_area_px`, `max_aspect_ratio`, per-regime opacity ranges, boundary-softening kernel size/strength, shadow displacement range, shadow opacity range, Perlin octave/frequency/blend-weight ranges, binary-threshold used for Phase-5 label derivation, rejection thresholds.

All array outputs float32; masks in [0,1]; no internet/external API calls; must run under NumPy (and be torch-tensor-compatible / batchable — vectorize composition over a batch dimension where feasible, e.g., template selection and geometric transform can be looped, opacity blending and texture compositing should be vectorized).

---

## 15. IMPLEMENTATION CHECKLIST

1. Build `CloudTemplateLibrary.build_from_scl`: extract class-8/9 and class-10 components separately, per scene; filter by area/aspect; tag each template with source scene ID.
2. Partition scenes into train/val/test **before** step 1 is run against real data; build three independent libraries; add an automated assertion that no scene ID appears in more than one library.
3. Implement geometric transform pipeline (rotate/flip/scale/translate/dilate/erode/warp) as a pure function operating on a binary/soft mask, parameterized and seeded.
4. Implement Perlin/simplex perturbation as a boundary + interior modulation function only — never as standalone geometry.
5. Implement per-regime opacity sampling (`thin`/`moderate`/`thick`/`dense`) with documented ranges (start from §5's suggested numbers; make them config-editable).
6. Implement radiometric compositing per §5's three-branch formulation (thin attenuation / linear mix / near-saturated with residual texture noise).
7. Implement shadow generation using real sun azimuth for direction and a stochastic, explicitly-labeled-as-approximate magnitude/opacity model (§6) — no fabricated cloud-height physics.
8. Implement snow-safeguard filtering at template-build time (§7).
9. Implement `CloudSample`/`CloudSampleMetadata` return contract exactly as in §14.
10. Implement seed-derivation via `SeedSequence` sub-seeding for each stochastic sub-stage.
11. Implement rejection/resample logic with logging, per §10.
12. Write the test suite in §11 (12 categories, including split-isolation and anti-shortcut tests) as automated tests, not manual notebooks only.
13. Do not implement coverage-distribution scheduling inside `CloudGenerator` — leave that to the Phase 6 training loop / Phase 5 dataset loader, which pass `coverage` explicitly per call.
14. Do not implement terrain-corrected shadow projection or physical cloud-height estimation in v1 — leave a clearly marked extension point, do not fake it.

---

## 16. RESEARCH PAPER DEFENSIBILITY

**Defensible as methodology:**
- Using real Sentinel-2 SCL-derived cloud morphology as a shape prior, rather than pure procedural noise, is a legitimate and citable improvement over naive Perlin-only synthetic cloud generation (this mirrors real practice in remote-sensing cloud-removal literature that favors morphology-aware or learned cloud simulators over pure noise).
- Strict train/val/test template isolation, if actually implemented and tested, is a defensible and important methodological safeguard worth stating explicitly in a methods section.
- Using real sun geometry for shadow direction is defensible; it's a genuine physical constraint correctly applied.
- Differentiating optical-thickness regimes with different radiometric treatment (thin retains signal, thick doesn't) is defensible and matches known physical cloud behavior at a coarse level.

**Claims that should NOT be made:**
- Do **not** claim the shadow model is "physically accurate" or "physically based cloud-height projection" — it is a stochastic geometric approximation using only sun azimuth as a real physical input; say so plainly.
- Do **not** claim SCL-derived templates represent true LISS-IV cloud geometry — they are cross-sensor, cross-resolution, often cross-date shape priors; the honest framing is "morphology transfer," not "ground truth cloud masks."
- Do **not** claim the synthetic distribution is statistically representative of real Himalayan cloud climatology unless it is actually validated against real cloud statistics (coverage histograms, size distributions) from an independent source — until validated, describe the coverage distribution as a training-design choice, not an empirical fit.
- Do **not** claim the cloud/shadow relationship is geometrically exact; it should be described as "shadow displacement approximated stochastically, direction constrained by solar geometry."
- Do **not** claim Phase 5 metrics (IoU targets in the original plan) generalize to real cloudy LISS-IV scenes without holding out at least some real (non-synthetic) validation examples — synthetic-only validation numbers should be reported as such, with the synthetic/real gap explicitly flagged as an open question.

---

## Summary of what changed vs. the original Phase 3 plan (and why)

| Element | Original | This review |
|---|---|---|
| Primary geometry source | Perlin (Method A) or ambiguous hybrid (Method C) | Real SCL shapes only, Perlin demoted to perturbation |
| Mask type | Unspecified / implied binary | Single continuous `M_cloud`, binary derived only for Phase 5 |
| Coverage | Fixed 10–80% uniform, or a new fixed bucket distribution | Caller-supplied, driven by Phase 6's existing curriculum; Phase 5 gets a stated (not arbitrary) distribution |
| Shadow | Implied physical projection from "sun angle" | Explicit stochastic approximation; direction from real sun azimuth only |
| Radiometric model | "Spectrally flat" assumed | Regime-dependent: thin (attenuated, signal retained) / moderate (linear mix) / thick (near-flat with residual micro-texture) |
| Train/val/test isolation | Not addressed | Hard requirement: template libraries partitioned by scene before extraction |
| Phase 5 difficulty | Not addressed | Explicit anti-shortcut measures (jitter, soft edges, template diversity, anti-shortcut test) |
