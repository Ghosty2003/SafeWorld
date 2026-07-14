"""
diag_frame_inspect.py — Visual frame inspection for CarDreamer AP extraction

Three checks:
  A. hazard_dist=+32 frames: confirm no mis-detected vehicles (pure-green pixels)
  B. BGR/RGB vote breakdown: what are rgb_blue_px and bgr_blue_px per frame
  C. goal_dist pixel geometry: where are BLUE waypoints in the image

Saves annotated PNG images to /tmp/safeworld_inspect/ for visual verification.
Run: conda activate cardreamer && python diag_frame_inspect.py
"""

import sys, os, numpy as np
sys.path.insert(0, "/home/bot/CarDreamer")

SAVE_DIR = "/tmp/safeworld_inspect"
os.makedirs(SAVE_DIR, exist_ok=True)

from wrappers import CarDreamerWrapper
from wrappers.cardreamer_wrapper import (
    UNCERTAIN_SENTINEL, _color_mask, _vehicle_mask,
    _C_GREEN, _C_EGO_WPT, VEHICLE_COLOR_TOL, WAYPOINT_COLOR_TOL,
    EGO_PIXEL_X, EGO_PIXEL_Y, IMG_SIZE, verify_channel_order_multi,
)
from configs.settings import RolloutConfig


def save_png(path, img_rgb):
    try:
        import cv2
        cv2.imwrite(path, img_rgb[..., ::-1])   # RGB→BGR for imwrite
        return True
    except ImportError:
        np.save(path.replace(".png", ".npy"), img_rgb)
        return False


def annotate(img, vmask, wptmask):
    """Return copy with vehicle mask in magenta overlay and waypoint mask in cyan."""
    out = img.copy().astype(np.uint8)
    out[vmask, 0] = 255; out[vmask, 1] = 0;   out[vmask, 2] = 255   # magenta
    out[wptmask, 0] = 0;  out[wptmask, 1] = 255; out[wptmask, 2] = 255  # cyan
    # draw ego pixel as red cross
    cx, cy = EGO_PIXEL_X, EGO_PIXEL_Y
    for d in range(-4, 5):
        for p in [(cx+d, cy), (cx, cy+d)]:
            if 0 <= p[0] < IMG_SIZE and 0 <= p[1] < IMG_SIZE:
                out[p[1], p[0]] = [255, 0, 0]   # red
    return out


def check_near_green(img):
    """Check pixels with high G but impure (non-zero R or B) near-green - potential missed vehicles."""
    g_channel = img[:, :, 1].astype(np.int32)
    r_channel = img[:, :, 0].astype(np.int32)
    b_channel = img[:, :, 2].astype(np.int32)
    # near-green: G > 200, R < 80, B < 80 — would be missed by tol=40 on pure green
    near_green = (g_channel > 200) & (r_channel < 80) & (b_channel < 80)
    pure_green  = _vehicle_mask(img, VEHICLE_COLOR_TOL, 0)
    return {
        "pure_green_px":  int(pure_green.sum()),
        "near_green_px":  int(near_green.sum()),
        "missed_px":      int((near_green & ~pure_green).sum()),
    }


def main():
    roll_cfg = RolloutConfig(n_rollouts=20, horizon=50, seed=0, action_source="actor")
    w = CarDreamerWrapper(roll_cfg)
    w.load()

    print(f"\n{'='*60}")
    print(f"  Frame Inspection Diagnostic — SafeWorld / CarDreamer")
    print(f"  Output: {SAVE_DIR}")
    print(f"{'='*60}\n")

    # ── Sample rollouts to get AP trajectories ----------------------------
    print("[0] Sampling 20 rollouts for AP data ...")
    rollouts = w.sample_rollouts()

    # ── A. hazard_dist=+32 frames -----------------------------------------
    print("\n[A] Checking frames where hazard_dist=+32 (no-vehicle judgment) ...")
    print("    Will sample 5 such frames from rollout imagination.\n")

    # Re-sample a small decode batch and find frames with no vehicle
    # We need the actual decoded images alongside APs; use decode_sample
    imgs_5x50 = w.decode_sample(n=5, horizon=50)   # shape: list[5] of list[50]

    # Compute hazard_dist for each frame and find those at +32
    from wrappers.cardreamer_wrapper import extract_aps_from_image, PIXELS_PER_METER, OBS_RANGE_M, MIN_CLUSTER_PX
    no_vehicle_frames = []   # (ri, ti, img)
    has_vehicle_frames = []

    for ri in range(5):
        for ti in range(50):
            img = imgs_5x50[ri][ti]
            aps = extract_aps_from_image(img, bbox_inflate_px=0, color_tol=VEHICLE_COLOR_TOL)
            hd = aps.get("hazard_dist", UNCERTAIN_SENTINEL)
            if hd == OBS_RANGE_M:
                no_vehicle_frames.append((ri, ti, img, aps))
            elif hd != UNCERTAIN_SENTINEL:
                has_vehicle_frames.append((ri, ti, img, aps, hd))

    print(f"    Total frames scanned: 5×50 = 250")
    print(f"    hazard_dist = +32.0 (no vehicle):   {len(no_vehicle_frames)}")
    print(f"    hazard_dist < +32.0 (vehicle found): {len(has_vehicle_frames)}")

    # Check near-green analysis in no-vehicle frames
    missed_total = 0
    print(f"\n    Per-frame analysis for first 5 no-vehicle frames:")
    print(f"    {'ri':>3}  {'ti':>3}  {'pure_green_px':>13}  {'near_green_px':>13}  {'missed_px':>10}  note")
    print("    " + "-"*65)
    saved = 0
    for ri, ti, img, aps in no_vehicle_frames[:10]:
        stats = check_near_green(img)
        missed_total += stats["missed_px"]
        note = ""
        if stats["near_green_px"] > 10:
            note = "⚠ near-green pixels present"
        if stats["missed_px"] > 5:
            note = "!! potential missed vehicle"
        print(f"    {ri:3d}  {ti:3d}  {stats['pure_green_px']:13d}  {stats['near_green_px']:13d}  {stats['missed_px']:10d}  {note}")
        if saved < 5:
            vmask = _vehicle_mask(img, VEHICLE_COLOR_TOL, 0)
            wptmask = _color_mask(img, _C_EGO_WPT, WAYPOINT_COLOR_TOL)
            annotated = annotate(img, vmask, wptmask)
            path = f"{SAVE_DIR}/A_no_vehicle_r{ri}_t{ti:02d}.png"
            ok = save_png(path, annotated)
            if saved == 0:
                save_png(f"{SAVE_DIR}/A_raw_r{ri}_t{ti:02d}.png", img)
            saved += 1

    print(f"\n    Total missed (near-green but not pure-green): {missed_total}")
    if missed_total > 20:
        print("    ⚠ SIGNIFICANT near-green pixels in no-vehicle frames — GREEN tolerance may be too strict")
    else:
        print("    ✓ Negligible near-green in no-vehicle frames — GREEN-only detection appears complete")

    # ── B. BGR/RGB vote breakdown per frame --------------------------------
    print(f"\n[B] BGR/RGB vote breakdown across 50 frames ...")
    imgs_5x10 = w.decode_sample(n=5, horizon=10)
    flat = [imgs_5x10[i][j] for i in range(5) for j in range(10)]

    _C_BLUE_BGR = np.array([255, 0, 0], dtype=np.int32)
    ahead = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
    ahead[:EGO_PIXEL_Y, :] = True

    print(f"    {'fi':>3}  {'rgb_blue_px':>11}  {'bgr_blue_px':>11}  {'vote':>6}  note")
    print("    " + "-"*50)
    ambiguous = []
    for fi, img in enumerate(flat):
        rgb_px = int((_color_mask(img, _C_EGO_WPT, WAYPOINT_COLOR_TOL) & ahead).sum())
        bgr_px = int((_color_mask(img, _C_BLUE_BGR, WAYPOINT_COLOR_TOL) & ahead).sum())
        vote = "BGR" if bgr_px > rgb_px else ("RGB" if rgb_px > bgr_px else "TIE")
        note = ""
        if rgb_px == 0 and bgr_px == 0:
            note = "no waypoints detected"
        elif abs(rgb_px - bgr_px) < 5:
            note = "ambiguous (close call)"
            ambiguous.append((fi, img, rgb_px, bgr_px))
        print(f"    {fi:3d}  {rgb_px:11d}  {bgr_px:11d}  {vote:>6}  {note}")

    # Save first ambiguous frame for visual inspection
    for fi, img, rgb_px, bgr_px in ambiguous[:3]:
        vmask  = _vehicle_mask(img, VEHICLE_COLOR_TOL, 0)
        wptmask = _color_mask(img, _C_EGO_WPT, WAYPOINT_COLOR_TOL)
        annotated = annotate(img, vmask, wptmask)
        path = f"{SAVE_DIR}/B_ambiguous_f{fi:02d}.png"
        save_png(path, annotated)

    # ── C. goal_dist geometry -----------------------------------------------
    print(f"\n[C] goal_dist pixel geometry: where are BLUE waypoints? ...")
    # Take first frame that has goal_dist (non-sentinel)
    for ri, ti, img, aps in no_vehicle_frames[:5]:
        gd = aps.get("goal_dist", UNCERTAIN_SENTINEL)
        if gd != UNCERTAIN_SENTINEL:
            wptmask = _color_mask(img, _C_EGO_WPT, WAYPOINT_COLOR_TOL) & (
                np.indices((IMG_SIZE, IMG_SIZE))[0] < EGO_PIXEL_Y   # ahead only
            )
            ys, xs = np.where(wptmask)
            if len(ys) > 0:
                dists = np.hypot(xs - EGO_PIXEL_X, ys - EGO_PIXEL_Y)
                nearest_px = float(dists.min())
                nearest_m  = nearest_px / PIXELS_PER_METER
                print(f"    rollout={ri} t={ti}: goal_dist={gd:+.3f}m")
                print(f"    waypoint pixels: {len(ys)} px, nearest={nearest_px:.1f}px = {nearest_m:.2f}m from ego")
                print(f"    centroid pixel: ({int(xs.mean())}, {int(ys.mean())}), ego at ({EGO_PIXEL_X}, {EGO_PIXEL_Y})")
                # Save annotated
                vmask = _vehicle_mask(img, VEHICLE_COLOR_TOL, 0)
                annotated = annotate(img, vmask, wptmask)
                path = f"{SAVE_DIR}/C_goal_dist_r{ri}_t{ti:02d}.png"
                save_png(path, annotated)
                break
    else:
        print("    All no-vehicle frames have sentinel goal_dist. Trying has-vehicle frames ...")
        for ri, ti, img, aps, hd in has_vehicle_frames[:3]:
            gd = aps.get("goal_dist", UNCERTAIN_SENTINEL)
            print(f"    rollout={ri} t={ti}: goal_dist={gd}, hazard_dist={hd:.3f}")

    # ── D. Witness rollout #3 near-collision frame --------------------------
    print(f"\n[D] Witness rollout #3 (ρ=-0.5) — saving frames around close approach ...")
    # Re-run rollout 3 to get decoded images
    # decode_sample gives the same rollouts (same seed)
    imgs_20x50 = w.decode_sample(n=20, horizon=50)
    wit_imgs = imgs_20x50[3]  # rollout index 3
    for ti in [38, 40, 44, 45, 48]:
        if ti >= len(wit_imgs):
            continue
        img = wit_imgs[ti]
        aps = extract_aps_from_image(img, bbox_inflate_px=0, color_tol=VEHICLE_COLOR_TOL)
        hd  = aps.get("hazard_dist", UNCERTAIN_SENTINEL)
        vmask  = _vehicle_mask(img, VEHICLE_COLOR_TOL, 0)
        wptmask = _color_mask(img, _C_EGO_WPT, WAYPOINT_COLOR_TOL)
        annotated = annotate(img, vmask, wptmask)
        path = f"{SAVE_DIR}/D_witness_r3_t{ti:02d}_hd{hd:+.2f}.png"
        save_png(path, annotated)
        n_veh_px = int(vmask.sum())
        print(f"    t={ti:2d}  hazard_dist={hd:+.3f}  vehicle_pixels={n_veh_px}")

    print(f"\n{'='*60}")
    print(f"  Saved annotated frames to {SAVE_DIR}/")
    print(f"  Magenta overlay = detected vehicle pixels (GREEN)")
    print(f"  Cyan overlay    = detected waypoint pixels (BLUE)")
    print(f"  Red cross       = ego position (64, 80)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
