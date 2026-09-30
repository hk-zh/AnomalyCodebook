# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Independent reimplementation of the MVTec-LOCO AU-sPRO metric (saturated
# Per-Region Overlap, integrated against the false-positive rate). It follows
# the specification carried in each object's `defects_config.json`:
#
#   * every ground-truth defect region is stored as its own PNG whose non-zero
#     pixels all take one `pixel_value`;
#   * that pixel_value maps to a `saturation_threshold`. With
#     `relative_saturation` the saturation area is threshold * region_area,
#     otherwise it is the threshold read as an absolute pixel count;
#   * a region's saturated overlap at score threshold t is
#         min( |region pixels with score >= t| / saturation_area , 1 ).
#
# The (FPR, mean-sPRO) curve is then integrated up to a set of FPR limits and
# normalised by the limit, exactly as the official evaluator reports AU-sPRO.
#
# This is NOT MVTec's code; it is an independent implementation used to compare
# our method and the DAPO baseline under one identical metric. Absolute
# saturation thresholds are defined at full ground-truth resolution, so callers
# must pass predictions already upsampled to the GT size.

import numpy as np


def build_saturation_lookup(defects_config):
    """pixel_value -> (saturation_threshold, relative_saturation) dict."""
    lut = {}
    for entry in defects_config:
        lut[int(entry["pixel_value"])] = (
            float(entry["saturation_threshold"]),
            bool(entry["relative_saturation"]),
        )
    return lut


def region_saturation_area(region_size, sat_threshold, relative):
    """Number of correctly-predicted pixels at which a region's PRO saturates."""
    if relative:
        area = sat_threshold * region_size
    else:
        area = sat_threshold
    # A region can never require more overlap than it has pixels, and never
    # fewer than one, so the ratio stays a valid fraction.
    return float(min(max(area, 1.0), region_size))


def collect_regions(gt_masks, sat_lut):
    """Turn per-image GT PNGs into a flat list of region descriptors.

    gt_masks: list over images; each element is a list of HxW uint8 arrays
              (one per region PNG, non-zero pixels share one pixel_value).
    Returns list of dicts: {sorted_scores placeholder via 'pixels' index, 'sat'}
    Here we only precompute the saturation area and keep the boolean mask, since
    scores are attached later per prediction.
    """
    regions = []
    for img_idx, masks in enumerate(gt_masks):
        for m in masks:
            vals = np.unique(m[m > 0])
            if vals.size == 0:
                continue
            # Each region PNG holds a single pixel_value; guard anyway.
            for v in vals:
                pv = int(v)
                if pv not in sat_lut:
                    raise KeyError(f"pixel_value {pv} absent from defects_config")
                sel = m == pv
                size = int(sel.sum())
                if size == 0:
                    continue
                sat_thr, rel = sat_lut[pv]
                regions.append({
                    "img_idx": img_idx,
                    "mask": sel,
                    "sat_area": region_saturation_area(size, sat_thr, rel),
                })
    return regions


def au_spro_from_scores(scores, gt_masks, sat_lut, fpr_limits=(0.05, 0.3),
                        num_fpr=400, subsample_bg=4_000_000, seed=0):
    """Compute normalised AU-sPRO at each FPR limit.

    scores:   list over images of HxW float arrays (predictions at GT res).
    gt_masks: list over images of lists of HxW uint8 region masks; an image
              with an empty list is a normal ('good') image.
    Every non-region pixel across every image (good or anomalous) counts toward
    the false-positive rate, matching the official protocol.
    """
    rng = np.random.default_rng(seed)

    # --- background scores drive the FPR axis --------------------------------
    # Union of all region masks per image marks the true-positive pixels; the
    # complement is background. Background is subsampled per image so the
    # quantile thresholds stay memory-bounded on full-resolution maps.
    bg_chunks = []
    per_img_union = []
    for i, sc in enumerate(scores):
        masks = gt_masks[i]
        if masks:
            union = np.zeros(sc.shape, dtype=bool)
            for m in masks:
                union |= m > 0
        else:
            union = np.zeros(sc.shape, dtype=bool)
        per_img_union.append(union)
        bg = sc[~union].ravel()
        bg_chunks.append(bg)

    all_bg = np.concatenate(bg_chunks)
    total_bg = all_bg.size
    if all_bg.size > subsample_bg:
        idx = rng.choice(all_bg.size, size=subsample_bg, replace=False)
        bg_sample = all_bg[idx]
    else:
        bg_sample = all_bg
    del all_bg, bg_chunks          # release the full background immediately
    bg_sample.sort()

    # Thresholds chosen as quantiles of the background scores so the sampled
    # FPR values are (approximately) uniform on [0, max_limit].
    max_limit = max(fpr_limits)
    fpr_targets = np.linspace(0.0, max_limit, num_fpr)
    # quantile at (1 - f) of background -> score threshold with FPR ~= f
    thresholds = np.quantile(bg_sample, np.clip(1.0 - fpr_targets, 0.0, 1.0))

    # FPR estimated from the same (large) background subsample. At the FPR
    # levels of interest (>=0.01) a multi-million-pixel sample is more than
    # accurate enough, and it avoids holding billions of full-res pixels.
    n = bg_sample.size
    ge_counts = n - np.searchsorted(bg_sample, thresholds, side="left")
    fpr = ge_counts / max(n, 1)

    regions = collect_regions(gt_masks, sat_lut)

    # --- mean saturated PRO at each threshold --------------------------------
    # For each region, sort its pixel scores once; overlap at t is the count of
    # region pixels with score >= t, obtained by one searchsorted per region.
    if len(regions) == 0:
        raise ValueError("no defect regions found; cannot compute sPRO")

    spro_sum = np.zeros_like(thresholds)
    for reg in regions:
        rs = scores[reg["img_idx"]][reg["mask"]].ravel()
        rs.sort()
        overlap = rs.size - np.searchsorted(rs, thresholds, side="left")
        spro_sum += np.minimum(overlap / reg["sat_area"], 1.0)
    mean_spro = spro_sum / len(regions)

    # --- integrate AU-sPRO up to each limit ----------------------------------
    # Sort by FPR (ascending) for trapezoidal integration; interpolate a point
    # exactly at the limit, integrate, then normalise by the limit.
    order = np.argsort(fpr)
    f_sorted = fpr[order]
    s_sorted = mean_spro[order]

    out = {}
    for lim in fpr_limits:
        f, s = _clip_curve(f_sorted, s_sorted, lim)
        au = np.trapz(s, f) / lim
        out[lim] = float(au)
    return out, {"num_regions": len(regions), "total_bg": int(total_bg)}


def _clip_curve(f_sorted, s_sorted, limit):
    """Restrict a monotonically-sorted (fpr, spro) curve to [0, limit],
    interpolating the endpoint at exactly `limit`."""
    keep = f_sorted <= limit
    f = f_sorted[keep]
    s = s_sorted[keep]
    if f.size == 0 or f[0] > 0:
        # Prepend (0, spro-at-0) using the smallest available point.
        f = np.concatenate(([0.0], f))
        s0 = s[0] if s.size else s_sorted[0]
        s = np.concatenate(([s0], s))
    if f[-1] < limit:
        # Interpolate spro at the limit from the first point beyond it.
        beyond = np.where(f_sorted > limit)[0]
        if beyond.size and beyond[0] >= 1:
            j = beyond[0]
            f0, f1 = f_sorted[j - 1], f_sorted[j]
            s_interp = s_sorted[j - 1] if f1 == f0 else (
                s_sorted[j - 1]
                + (s_sorted[j] - s_sorted[j - 1]) * (limit - f0) / (f1 - f0)
            )
        else:
            # Either no point exceeds the limit (extend flat from the last
            # kept value) or even the lowest-FPR point already exceeds it
            # (curve never reaches low FPR; hold the first sPRO value flat).
            s_interp = s[-1]
        f = np.concatenate((f, [limit]))
        s = np.concatenate((s, [s_interp]))
    return f, s


# ---------------------------------------------------------------------------
# Self-test: a hand-checkable synthetic case.
# ---------------------------------------------------------------------------
def _self_test():
    # Two 200x200 images. Background carries continuous low-level noise in
    # [0, 0.1) so the FPR axis is well-defined; every region pixel we "detect"
    # is scored at 1.0, cleanly above the noise floor.
    #   img0: 20x20 structural region (pixel 236, relative 1.0 -> saturation
    #         area = 400 px, i.e. the whole region must be covered).
    #   img1: 100x100 "missing" region (pixel 250, absolute 2000 -> only 2000
    #         of the 10000 px need to be covered for full credit).
    rng = np.random.default_rng(0)
    H = W = 200
    sat_lut = {236: (1.0, True), 250: (2000.0, False)}

    m0 = np.zeros((H, W), np.uint8); m0[0:20, 0:20] = 236
    m1 = np.zeros((H, W), np.uint8); m1[0:100, 0:100] = 250
    gt = [[m0], [m1]]

    def noisy():
        return (rng.random((H, W)) * 0.1).astype(np.float32)

    # Perfect predictor: 1.0 on every region pixel, noise elsewhere.
    s0 = noisy(); s0[m0 > 0] = 1.0
    s1 = noisy(); s1[m1 > 0] = 1.0
    out, info = au_spro_from_scores([s0, s1], gt, sat_lut,
                                    fpr_limits=(0.05, 0.3), num_fpr=800)
    # No background pixel exceeds the region score, so at low FPR both regions
    # saturate to 1.0 and mean sPRO == 1.0 across the whole [0, limit] range.
    assert info["num_regions"] == 2, info
    assert abs(out[0.05] - 1.0) < 1e-3, out
    assert abs(out[0.3] - 1.0) < 1e-3, out

    # Degrade img0 so only half its region (200 of 400 px) is detected:
    # structural sPRO -> 0.5; the missing region still has 10000 >> 2000 px
    # detected so it stays saturated at 1.0. Mean -> 0.75.
    s0b = noisy(); s0b[m0 > 0] = 0.0            # undetected region px -> 0
    s0b[0:10, 0:20] = 1.0                        # detect 200 of the 400 px
    out2, _ = au_spro_from_scores([s0b, s1], gt, sat_lut,
                                  fpr_limits=(0.05,), num_fpr=800)
    assert abs(out2[0.05] - 0.75) < 5e-3, out2

    print("spro self-test OK:", {k: round(v, 4) for k, v in out.items()},
          "| degraded:", round(out2[0.05], 4))


if __name__ == "__main__":
    _self_test()
