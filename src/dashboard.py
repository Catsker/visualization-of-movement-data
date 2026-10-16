"""
Streamlit dashboard: browse the operator's 3D digital twin frame by frame.

Run from the project root::

    streamlit run src/dashboard.py
    streamlit run src/dashboard.py -- --tracking path/to/trackingData_*.txt

The tracking log is parsed once per server process (``st.cache_data``); every
slider move only builds a figure for the selected sample, so the browser never
receives more than one frame of data and no downsampling is needed. Styling and
trace construction are shared with ``visualize_3d.py``.
"""

import argparse
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import parser as tracking_parser
import visualize_3d as viz3d

CHART_HEIGHT_PX = 760


def parse_args():
    p = argparse.ArgumentParser(description="Streamlit 3D digital twin dashboard.")
    p.add_argument("--data-dir", default=tracking_parser.DEFAULT_DATA_DIR,
                   help="Folder to auto-detect the tracking log in")
    p.add_argument("--tracking", default=None, help="Path to trackingData_*.txt (overrides auto-detect)")
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


@st.cache_data(show_spinner="Parsing tracking log (once per server start)...")
def load_data(tracking_path):
    return tracking_parser.load_tracking(tracking_path)


@st.cache_data(show_spinner="Computing scene bounds...")
def world_ranges(tracking_path):
    """Fixed axis ranges over the whole log, so the scene does not rescale between frames."""
    data = load_data(tracking_path)
    hands = data.hands_joints[data.hands_active]                 # active hands only
    gaze_tips = np.array([viz3d.head_geometry(pose)[1] for pose in data.head_poses])
    points = np.concatenate([data.body_joints.reshape(-1, 3), hands.reshape(-1, 3),
                             data.head_poses[:, :3], gaze_tips])
    lo = np.nanmin(points, axis=0) - viz3d.AXIS_PAD_M
    hi = np.nanmax(points, axis=0) + viz3d.AXIS_PAD_M
    return [[float(a), float(b)] for a, b in zip(lo, hi)]


def frame_figure(data, i, ranges):
    traces = viz3d.build_traces()
    for trace, (x, y, z) in zip(traces, viz3d.sample_geometry(data, i)):
        trace.update(x=x, y=y, z=z)
    fig = go.Figure(traces)
    fig.update_layout(**viz3d.figure_layout(viz3d.frame_title(data, i), ranges),
                      height=CHART_HEIGHT_PX)
    return fig


def frame_selector(count):
    """Slider for coarse scrubbing + number input for an exact frame, kept in sync."""
    for key in ("frame_slider", "frame_input"):
        st.session_state.setdefault(key, 0)

    def from_slider():
        st.session_state.frame_input = st.session_state.frame_slider

    def from_input():
        st.session_state.frame_slider = st.session_state.frame_input

    slider_col, input_col = st.columns([5, 1], vertical_alignment="bottom")
    slider_col.slider("Frame (log sample index)", min_value=0, max_value=count - 1, step=1,
                      key="frame_slider", on_change=from_slider,
                      help="Click the slider, then use the arrow keys to step one sample at a time.")
    input_col.number_input("Go to frame", min_value=0, max_value=count - 1, step=1,
                           key="frame_input", on_change=from_input,
                           help="Type an exact sample index and press Enter; +/- step by one.")
    return st.session_state.frame_slider


def main():
    st.set_page_config(page_title="XR Digital Twin", layout="wide")
    args = parse_args()
    tracking_path = args.tracking or tracking_parser.find_single_file(
        args.data_dir, tracking_parser.TRACKING_PATTERN)

    data = load_data(tracking_path)
    ranges = world_ranges(tracking_path)
    t_sec = (data.timestamps - data.timestamps[0]) / 1e9

    st.title("Operator digital twin")
    st.caption(f"World Space (OpenXR: x right, y up, z back, metres) - "
               f"`{tracking_path}` - {len(data)} samples, {t_sec[-1]:.1f} s")

    i = frame_selector(len(data))

    st.caption(f"timeStampNs `{data.timestamps[i]}`")

    cols = st.columns(4)
    cols[0].metric("Time", f"{t_sec[i]:.3f} s")
    cols[1].metric("Sample", f"{i} / {len(data) - 1}")
    for col, (h, name) in zip(cols[2:], enumerate(tracking_parser.HANDS)):
        col.metric(viz3d.STYLE[name]["label"],
                   "tracked" if data.hands_active[i, h] else "not tracked")

    # theme=None: keep the figure's own colours; Streamlit's chart theme would
    # override the font colours and make the legend unreadable on our surface.
    st.plotly_chart(frame_figure(data, i, ranges), width="stretch", theme=None, key="skeleton")


main()
