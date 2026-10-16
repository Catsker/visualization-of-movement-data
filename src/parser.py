"""
Parser for VR telemetry logs (``trackingData_*.txt``).

File layout
-----------
* Line 1 -- JSON header with camera calibration:
    - ``cameraIntrinsics``: string ``"[a, b, c, d]"`` with 4 floats.
    - ``cameraExtrinsics``: two row-major 4x4 matrices separated by ``|``
      (left eye first, right eye second).
* Lines 2..N -- one JSON object per tracking sample with ``timeStampNs``,
  ``Head.pose`` (7 floats), ``Hand.leftHand`` / ``Hand.rightHand``
  (26 OpenXR joints each) and ``Body.joints`` (24 body joints).
  All positions are in the same global tracking space (World Space).

The "comma bug"
---------------
Unity serialized the pose arrays with ``float.ToString()`` under a locale
that uses ``,`` as the decimal separator, and then joined the values with
``,`` as well. A pose ``(x, y, z, qx, qy, qz, qw)`` therefore looks like::

    "-0,316226125,-0,4188593,0,00140838465,0,3816821,..."

Every float contributes exactly two comma-separated tokens: the integer part
(with sign) and the fractional part (possibly with an exponent, e.g.
``"809443E-05"``). Splitting by ``,`` and re-joining consecutive pairs with
``.`` restores the original numbers: ``"-0" + "." + "316226125"``.
The same bug affects ``Head.pose``, hand joints and body joints.
"""

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

# Project root = parent of src/; data files are auto-detected in <root>/data.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VIDEO_PATTERN = "CameraRecord_*.mp4"
TRACKING_PATTERN = "trackingData_*.txt"

NUM_HAND_JOINTS = 26   # OpenXR XR_HAND_JOINT_COUNT_EXT
NUM_BODY_JOINTS = 24   # PICO body tracking (BodyTrackerRole 0..23)
POSE_FLOATS = 7        # x, y, z, qx, qy, qz, qw
HANDS = ("leftHand", "rightHand")   # keys inside the "Hand" object; order = hand axis in arrays


@dataclass
class CameraParams:
    """Camera calibration from the header line of the tracking log.

    Attributes:
        intrinsics_raw: The 4 numbers exactly as stored in ``cameraIntrinsics``.
        extrinsics: List of 4x4 row-major matrices from ``cameraExtrinsics``
            in file order (index 0 = first matrix = left eye).
        header_timestamp_ns: Timestamp of the first video frame.
    """
    intrinsics_raw: np.ndarray
    extrinsics: list
    header_timestamp_ns: int


@dataclass
class TrackingFrame:
    """One parsed telemetry line.

    Attributes:
        timestamp_ns: ``timeStampNs`` of the sample.
        head_pose: (7,) array ``[x, y, z, qx, qy, qz, qw]`` in World Space.
        hands_active: (2,) bool array, ``Hand.<hand>.isActive`` in :data:`HANDS` order.
        hands_joints: (2, 26, 3) joint positions ``(X, Y, Z)`` in :data:`HANDS` order.
        body_joints: (24, 3) body joint positions ``(X, Y, Z)``.

    Missing hands / body are filled with NaN (hands are also marked inactive).
    """
    timestamp_ns: int
    head_pose: np.ndarray
    hands_active: np.ndarray
    hands_joints: np.ndarray
    body_joints: np.ndarray


@dataclass
class TrackingData:
    """The whole log as stacked numpy arrays (N = number of samples).

    Attributes:
        camera: :class:`CameraParams` from the header.
        timestamps: (N,) int64 ``timeStampNs``.
        head_poses: (N, 7) ``[x, y, z, qx, qy, qz, qw]``.
        hands_joints: (N, 2, 26, 3); axis 1 follows :data:`HANDS`.
        hands_active: (N, 2) bool.
        body_joints: (N, 24, 3).
    """
    camera: CameraParams
    timestamps: np.ndarray
    head_poses: np.ndarray
    hands_joints: np.ndarray
    hands_active: np.ndarray
    body_joints: np.ndarray

    def __len__(self):
        return len(self.timestamps)


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def find_single_file(data_dir, pattern):
    """Return the only file in ``data_dir`` matching ``pattern`` (glob).

    If several files match, the lexicographically last one is used (file names
    embed a ``YYYYMMDD_HHMMSS`` stamp, so that is the most recent recording).
    """
    matches = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in '{data_dir}'")
    if len(matches) > 1:
        print(f"[warn] {len(matches)} files match '{pattern}', using {matches[-1]}",
              file=sys.stderr)
    return matches[-1]


# --------------------------------------------------------------------------- #
# Low-level parsing helpers
# --------------------------------------------------------------------------- #
def parse_bracketed_floats(text):
    """Parse a string like ``"[1.0, 2.5, -3]"`` into a float numpy array."""
    return np.array([float(v) for v in text.strip().strip("[]").split(",")],
                    dtype=np.float64)


def parse_comma_bug_floats(text, expected_count=POSE_FLOATS):
    """Recover floats from a Unity string where ``,`` is both separator and decimal point.

    Each float is split into exactly two tokens (integer part, fractional part),
    so the token list is regrouped into pairs and joined with ``.``::

        "-0,31622,-0,41885,0,0014"  ->  ["-0","31622","-0","41885","0","0014"]
                                    ->  [-0.31622, -0.41885, 0.0014]

    Scientific notation survives this because the exponent stays attached to
    the fractional token (``"-5,809443E-05"`` -> ``-5.809443E-05``).

    Args:
        text: The corrupted string.
        expected_count: Number of floats the string must contain.

    Raises:
        ValueError: If the token count is not ``2 * expected_count``. That would
            mean some value was printed without a fractional part (e.g. an
            exact ``0``), which makes the pairing ambiguous.
    """
    tokens = text.split(",")
    if len(tokens) != 2 * expected_count:
        raise ValueError(
            f"Expected {2 * expected_count} comma tokens for {expected_count} floats, "
            f"got {len(tokens)}: {text!r}")
    return np.array([float(f"{tokens[i]}.{tokens[i + 1]}")
                     for i in range(0, len(tokens), 2)], dtype=np.float64)


def parse_joint_positions(joints_json, expected_count, what):
    """Parse a list of ``{"p": "<7 comma-bug floats>", ...}`` into an (N, 3) array.

    Only ``x, y, z`` (the first 3 of the 7 pose floats) are kept.
    """
    if len(joints_json) != expected_count:
        raise ValueError(f"Expected {expected_count} {what} joints, got {len(joints_json)}")
    return np.array([parse_comma_bug_floats(j["p"])[:3] for j in joints_json])


# --------------------------------------------------------------------------- #
# Header & frame parsing
# --------------------------------------------------------------------------- #
def parse_header(line):
    """Parse the first line of the log into :class:`CameraParams`.

    ``cameraExtrinsics`` holds 16-element row-major matrices separated by ``|``.
    """
    header = json.loads(line)
    intrinsics = parse_bracketed_floats(header["cameraIntrinsics"])
    if intrinsics.size != 4:
        raise ValueError(f"cameraIntrinsics must have 4 values, got {intrinsics.size}")

    matrices = [parse_bracketed_floats(m).reshape(4, 4)
                for m in header["cameraExtrinsics"].split("|")]
    return CameraParams(
        intrinsics_raw=intrinsics,
        extrinsics=matrices,
        header_timestamp_ns=int(header.get("timeStampNs", 0)),
    )


def parse_hand(hand_json, hand_name):
    """Parse one ``Hand.<hand_name>`` object into ``(active, joints)``.

    A missing hand yields ``(False, NaN array)`` so that one absent hand does not
    drop the whole frame; a present but malformed hand raises ``ValueError``.
    """
    if not hand_json:
        return False, np.full((NUM_HAND_JOINTS, 3), np.nan)
    joints = parse_joint_positions(hand_json.get("HandJointLocations", []),
                                   NUM_HAND_JOINTS, hand_name)
    return bool(hand_json.get("isActive", 0)), joints


def parse_body(body_json):
    """Parse ``Body.joints`` into a (24, 3) array (NaN if the body is missing)."""
    joints_json = (body_json or {}).get("joints")
    if not joints_json:
        return np.full((NUM_BODY_JOINTS, 3), np.nan)
    return parse_joint_positions(joints_json, NUM_BODY_JOINTS, "body")


def parse_frame(line):
    """Parse one telemetry JSON line into a :class:`TrackingFrame`."""
    data = json.loads(line)

    hand_data = data.get("Hand", {})
    parsed_hands = [parse_hand(hand_data.get(name), name) for name in HANDS]

    return TrackingFrame(
        timestamp_ns=int(data["timeStampNs"]),
        head_pose=parse_comma_bug_floats(data["Head"]["pose"]),
        hands_active=np.array([active for active, _ in parsed_hands], dtype=bool),
        hands_joints=np.stack([joints for _, joints in parsed_hands]),
        body_joints=parse_body(data.get("Body")),
    )


def iter_frames(path):
    """Yield :class:`TrackingFrame` objects for every data line (skips the header).

    Streams the file line by line, so the multi-hundred-MB log is never held
    in memory as text.
    """
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line_no, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            try:
                yield parse_frame(line)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                print(f"[warn] line {line_no} skipped: {exc}", file=sys.stderr)


def read_camera_params(path):
    """Read and parse only the header line of the tracking log."""
    with open(path, "r", encoding="utf-8") as f:
        return parse_header(f.readline())


def load_tracking(path):
    """Parse the whole log into a :class:`TrackingData` of stacked numpy arrays."""
    frames = list(iter_frames(path))
    return TrackingData(
        camera=read_camera_params(path),
        timestamps=np.array([f.timestamp_ns for f in frames], dtype=np.int64),
        head_poses=np.array([f.head_pose for f in frames]),
        hands_joints=np.array([f.hands_joints for f in frames]),
        hands_active=np.array([f.hands_active for f in frames], dtype=bool),
        body_joints=np.array([f.body_joints for f in frames]),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser(description="Parse VR tracking logs and print a validation sample.")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help=f"Folder to search for {TRACKING_PATTERN} (default: %(default)s)")
    p.add_argument("--tracking", default=None,
                   help="Explicit path to a trackingData_*.txt file (overrides --data-dir)")
    p.add_argument("--full", action="store_true",
                   help="Also parse every line and print dataset statistics")
    return p


def main():
    args = build_arg_parser().parse_args()
    tracking_path = args.tracking or find_single_file(args.data_dir, TRACKING_PATTERN)
    print(f"Tracking file: {tracking_path}\n")

    np.set_printoptions(precision=6, suppress=True)
    camera = read_camera_params(tracking_path)
    print("cameraIntrinsics (raw, in file order):")
    print(f"  {camera.intrinsics_raw.tolist()}\n")
    print("cameraExtrinsics - first matrix (4x4):")
    print(camera.extrinsics[0], "\n")

    first = next(iter_frames(tracking_path))
    print(f"First frame: timeStampNs={first.timestamp_ns}")
    print(f"  Head.pose  xyz={first.head_pose[:3]}  quat(xyzw)={first.head_pose[3:]}")
    for h, name in enumerate(HANDS):
        x, y, z = first.hands_joints[h, 0]
        print(f"  {name:<9} isActive={first.hands_active[h]!s:<5} "
              f"joint 0 (Palm): X={x:.9f}  Y={y:.9f}  Z={z:.9f}")
    x, y, z = first.body_joints[0]
    print(f"  Body      joint 0 (Pelvis): X={x:.9f}  Y={y:.9f}  Z={z:.9f}")

    if args.full:
        data = load_tracking(tracking_path)
        ts = data.timestamps
        print(f"\nParsed frames: {len(data)}")
        print(f"  head_poses {data.head_poses.shape}, hands_joints {data.hands_joints.shape}, "
              f"body_joints {data.body_joints.shape}")
        for h, name in enumerate(HANDS):
            print(f"  {name} active in {data.hands_active[:, h].sum()} / {len(data)} frames")
        print(f"  Duration: {(ts[-1] - ts[0]) / 1e9:.2f} s")


if __name__ == "__main__":
    main()
