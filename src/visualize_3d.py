"""
Animated interactive 3D "digital twin" of the operator in World Space.

Head, body (24 joints) and both hands (26 joints each) are plotted as-is in
absolute World Space (OpenXR: ``x`` right, ``y`` up, ``z`` backwards, metres),
connected into a skeleton and animated over the whole log. The result is a
self-contained offline HTML player with Play/Pause buttons and a time slider.
No camera model is involved, so this view is free of the projection /
calibration effects visible on the video overlay.

Downsampling
------------
The log has ~20k samples (~90 Hz); embedding all of them would make the HTML
huge and the browser unusable. ``--frames`` samples (default 250) are picked
evenly in *time* (not by line index, so gaps in the log do not skew coverage)
from the first to the last timestamp -- roughly every ~90th sample.

Every animation frame carries the same number of points per trace; missing
joints and inactive hands are written as ``null`` so Plotly simply hides them.
Axis ranges are fixed to the extent of all sampled frames, otherwise the scene
would rescale (and visually jump) on every frame.
"""

import argparse
import os
import time

import numpy as np
import plotly.graph_objects as go

import parser as tracking_parser
from visualize import BODY_BONES, FINGERS, PALM_INDEX, pose_to_matrix

DEFAULT_OUTPUT = os.path.join(tracking_parser.DEFAULT_DATA_DIR, "skeleton_3d_animated.html")
DEFAULT_FRAMES = 250
DEFAULT_FRAME_MS = 100   # playback time per animation frame

GAZE_LENGTH_M = 0.3      # length of the head forward-direction line
AXIS_PAD_M = 0.1         # margin around the data in the fixed axis ranges
DECIMALS = 4             # 0.1 mm precision keeps the HTML small

BODY_NAMES = ["pelvis", "L hip", "R hip", "spine1", "L knee", "R knee", "spine2",
              "L ankle", "R ankle", "spine3", "L foot", "R foot", "neck", "L collar",
              "R collar", "head", "L shoulder", "R shoulder", "L elbow", "R elbow",
              "L wrist", "R wrist", "L hand", "R hand"]
HAND_NAMES = ["palm", "wrist"] + [
    f"{finger} {segment}"
    for finger, segments in [("thumb", ["metacarpal", "proximal", "distal", "tip"])] + [
        (f, ["metacarpal", "proximal", "intermediate", "distal", "tip"])
        for f in ("index", "middle", "ring", "little")]
    for segment in segments]
HAND_BONES = [(PALM_INDEX, 1)] + [   # palm -> wrist, then every finger chain
    bone for chain in FINGERS.values() for bone in zip(chain[:-1], chain[1:])]

# Hands and head carry the categorical identity (first three slots of the
# reference palette, which stay CVD-distinguishable as a set); the body is the
# neutral scaffold they hang on.
STYLE = {
    "leftHand": {"label": "Left hand", "color": "#2a78d6"},
    "rightHand": {"label": "Right hand", "color": "#eb6834"},
    "head": {"label": "Head (Head.pose)", "color": "#1baf7a"},
    "body": {"label": "Body", "color": "#8a8983"},
}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"


# --------------------------------------------------------------------------- #
# Downsampling & per-sample geometry
# --------------------------------------------------------------------------- #
def sample_indices(timestamps, count):
    """Indices of ``count`` samples spread evenly in time over the whole log."""
    targets = np.linspace(timestamps[0], timestamps[-1], count)
    idx = np.searchsorted(timestamps, targets).clip(0, len(timestamps) - 1)
    return np.unique(idx)


def visible_hands(data, i):
    """(2, 26, 3) hand joints of sample ``i`` with inactive hands blanked to NaN."""
    hands = data.hands_joints[i].copy()
    hands[~data.hands_active[i]] = np.nan
    return hands


def head_geometry(head_pose):
    """Head position and the tip of its forward direction (-z in the head frame)."""
    head_to_world = pose_to_matrix(head_pose)
    origin = head_to_world[:3, 3]
    forward = head_to_world[:3, :3] @ np.array([0.0, 0.0, -1.0])
    return origin, origin + GAZE_LENGTH_M * forward


def to_list(values):
    """Round to :data:`DECIMALS` and turn NaN into ``None`` (JSON ``null`` = hidden point)."""
    return [None if np.isnan(v) else v for v in np.round(values, DECIMALS).tolist()]


def joint_xyz(points):
    return [to_list(points[:, axis]) for axis in range(3)]


def bone_xyz(points, bones):
    """Bone segments as one polyline: ``a, b, NaN`` per bone (fixed length every frame)."""
    seg = np.full((len(bones), 3, 3), np.nan)
    for k, (a, b) in enumerate(bones):
        seg[k, 0], seg[k, 1] = points[a], points[b]
    seg = seg.reshape(-1, 3)
    return [to_list(seg[:, axis]) for axis in range(3)]


def sample_geometry(data, i):
    """x/y/z lists for every trace of sample ``i``, in :func:`build_traces` order."""
    hands = visible_hands(data, i)
    origin, tip = head_geometry(data.head_poses[i])
    geometry = [bone_xyz(data.body_joints[i], BODY_BONES), joint_xyz(data.body_joints[i])]
    for h in range(len(tracking_parser.HANDS)):
        geometry += [bone_xyz(hands[h], HAND_BONES), joint_xyz(hands[h])]
    geometry += [joint_xyz(origin[None]), joint_xyz(np.stack([origin, tip])),
                 joint_xyz(tip[None])]
    return geometry


# --------------------------------------------------------------------------- #
# Plotly figure
# --------------------------------------------------------------------------- #
def skeleton_traces(style, group, names, marker_size):
    """Line trace for the bones + marker trace for the joints, sharing one legend entry."""
    lines = go.Scatter3d(
        mode="lines", name=style["label"], legendgroup=group,
        line=dict(color=style["color"], width=5), hoverinfo="skip")
    joints = go.Scatter3d(
        mode="markers", name=style["label"], legendgroup=group, showlegend=False,
        marker=dict(size=marker_size, color=style["color"], line=dict(color=SURFACE, width=1)),
        text=[f"{style['label']}: {i} {n}" for i, n in enumerate(names)],
        hovertemplate="%{text}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f} m<extra></extra>")
    return [lines, joints]


def head_traces():
    """Head marker, forward-direction line and a small dot marking the line's tip."""
    style = STYLE["head"]
    marker = go.Scatter3d(
        mode="markers", name=style["label"], legendgroup="head",
        marker=dict(size=9, symbol="diamond", color=style["color"],
                    line=dict(color=SURFACE, width=2)),
        hovertemplate="Head<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f} m<extra></extra>")
    gaze = go.Scatter3d(
        mode="lines", name="Head forward", legendgroup="head", showlegend=False,
        line=dict(color=style["color"], width=4, dash="dash"), hoverinfo="skip")
    tip = go.Scatter3d(
        mode="markers", name="Head forward", legendgroup="head", showlegend=False,
        marker=dict(size=3, color=style["color"]), hoverinfo="skip")
    return [marker, gaze, tip]


def build_traces():
    """Empty, styled traces; coordinates are filled from :func:`sample_geometry`."""
    traces = skeleton_traces(STYLE["body"], "body", BODY_NAMES, marker_size=5)
    for name in tracking_parser.HANDS:
        traces += skeleton_traces(STYLE[name], name, HAND_NAMES, marker_size=3)
    return traces + head_traces()


def axis_ranges(geometries):
    """Fixed [min, max] per axis over all frames, padded by :data:`AXIS_PAD_M`."""
    ranges = []
    for axis in range(3):
        values = np.array([v for frame in geometries for trace in frame
                           for v in trace[axis] if v is not None])
        ranges.append([values.min() - AXIS_PAD_M, values.max() + AXIS_PAD_M])
    return ranges


def frame_title(data, i):
    t_sec = (data.timestamps[i] - data.timestamps[0]) / 1e9
    lost = [STYLE[name]["label"].lower() for h, name in enumerate(tracking_parser.HANDS)
            if not data.hands_active[i, h]]
    status = f" - not tracked: {', '.join(lost)}" if lost else ""
    return (f"Operator digital twin - World Space<br>"
            f"<sup>t = {t_sec:6.2f} s, log sample {i} / {len(data)}{status}</sup>")


def player_controls(frame_names, labels, frame_ms):
    """Play/Pause ``updatemenus`` and a time slider over ``frame_names``."""
    # Scatter3d does not tween between frames, so transitions are off and every
    # frame is a full redraw.
    play = dict(frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0),
                fromcurrent=True, mode="immediate")
    pause = dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0),
                 mode="immediate")
    seek = dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0),
                mode="immediate")
    menus = [dict(
        type="buttons", direction="left", showactive=False,
        x=0.0, y=0.0, xanchor="left", yanchor="top", pad=dict(t=45, r=10),
        bgcolor=SURFACE, bordercolor=GRID, font=dict(color=TEXT),
        buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                 dict(label="⏸ Pause", method="animate", args=[[None], pause])])]
    sliders = [dict(
        active=0, x=0.14, y=0.0, len=0.86, xanchor="left", yanchor="top", pad=dict(t=30),
        # One tick label per step would be an unreadable smear with ~250 steps:
        # hide them and show the current time in the header instead.
        font=dict(color="rgba(0,0,0,0)"), ticklen=0, minorticklen=0,
        bgcolor=GRID, bordercolor=GRID, activebgcolor=STYLE["head"]["color"],
        currentvalue=dict(prefix="t = ", suffix=" s", font=dict(color=TEXT_MUTED, size=13)),
        steps=[dict(label=label, method="animate", args=[[name], seek])
               for name, label in zip(frame_names, labels)])]
    return menus, sliders


def figure_layout(title, ranges):
    """Shared layout: fixed World Space axes, ``aspectmode='data'``, Y-up camera.

    Also used by ``dashboard.py`` so the web app and the HTML export look the same.
    """
    x_range, y_range, z_range = ranges
    axis = dict(backgroundcolor=SURFACE, gridcolor=GRID, zerolinecolor=GRID,
                color=TEXT_MUTED, showspikes=False, autorange=False)
    return dict(
        title=dict(text=title, font=dict(color=TEXT)),
        paper_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=TEXT),
        legend=dict(itemsizing="constant", bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=70, b=0),
        # Keep the user's camera rotation/zoom while frames change.
        uirevision="skeleton",
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X, m (right)", range=x_range, **axis),
            yaxis=dict(title="Y, m (up)", range=y_range, **axis),
            zaxis=dict(title="Z, m (back)", range=z_range, **axis),
            # OpenXR is Y-up: keep the raw world values on the axes and only tell
            # the viewer camera which way is up.
            camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=1.4, y=0.5, z=-2.2)),
            uirevision="skeleton",
        ),
    )


def build_figure(data, indices, frame_ms):
    geometries = [sample_geometry(data, i) for i in indices]
    traces = build_traces()
    for trace, (x, y, z) in zip(traces, geometries[0]):
        trace.update(x=x, y=y, z=z)

    names = [str(i) for i in indices]
    frames = [go.Frame(name=name,
                       data=[go.Scatter3d(x=x, y=y, z=z) for x, y, z in geometry],
                       layout=dict(title_text=frame_title(data, i)))
              for name, i, geometry in zip(names, indices, geometries)]
    labels = [f"{(data.timestamps[i] - data.timestamps[0]) / 1e9:.1f}" for i in indices]
    menus, sliders = player_controls(names, labels, frame_ms)

    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(**figure_layout(frame_title(data, indices[0]), axis_ranges(geometries)))
    fig.update_layout(updatemenus=menus, sliders=sliders, margin_b=90)
    return fig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Export an animated interactive 3D skeleton (World Space) to HTML.")
    p.add_argument("--data-dir", default=tracking_parser.DEFAULT_DATA_DIR,
                   help="Folder to auto-detect the tracking log in (default: %(default)s)")
    p.add_argument("--tracking", default=None, help="Path to trackingData_*.txt (overrides auto-detect)")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output HTML path (default: %(default)s)")
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                   help="Number of animation frames sampled evenly in time (default: %(default)s)")
    p.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS,
                   help="Playback duration of one animation frame in ms (default: %(default)s)")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    tracking_path = args.tracking or tracking_parser.find_single_file(
        args.data_dir, tracking_parser.TRACKING_PATTERN)
    print(f"Tracking: {tracking_path}")

    t0 = time.time()
    data = tracking_parser.load_tracking(tracking_path)
    duration = (data.timestamps[-1] - data.timestamps[0]) / 1e9
    print(f"Parsed {len(data)} telemetry samples ({duration:.1f} s) in {time.time() - t0:.1f} s")

    indices = sample_indices(data.timestamps, args.frames)
    playback = len(indices) * args.frame_ms / 1000
    print(f"Animation: {len(indices)} frames (~every {len(data) / len(indices):.0f}th sample), "
          f"playback {playback:.1f} s = {duration / playback:.1f}x real time")

    fig = build_figure(data, indices, args.frame_ms)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True, auto_play=False)
    size_mb = os.path.getsize(args.output) / 2**20
    print(f"Saved {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
