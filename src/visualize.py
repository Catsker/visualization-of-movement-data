"""
Project tracked hands and body (World Space) onto the first-person SBS video.

Transformation chain (4x4 homogeneous matrices, column vectors)
---------------------------------------------------------------
Hands, body and head are all given in the global tracking space (World).
The camera is rigidly attached to the moving head, so for every frame::

    Head_to_World   = [ R(q_head) | t_head ]            from Head.pose
    Camera_to_Head  = EXTRINSICS_TO_HEAD @ cameraExtrinsics[0]
    Camera_to_World = Head_to_World @ Camera_to_Head
    World_to_Camera = inv(Camera_to_World)

    [X, Y, Z, 1]^T  = World_to_Camera @ [x_w, y_w, z_w, 1]^T
    u = fx * X / Z + cx,   v = fy * Y / Z + cy           (pinhole, intrinsics only)

``cameraExtrinsics`` is a Camera->Head transform, but it is written in the
headset's *device* axis convention, not in the OpenXR head axes used by
``Head.pose``. ``EXTRINSICS_TO_HEAD`` is the fixed change of basis between the
two (a pure axis permutation, no offsets) -- see its definition below.

Coordinate conventions
----------------------
* World / Head (``Head.pose``, hands, body): OpenXR, right-handed,
  ``x`` right, ``y`` up, ``z`` backwards (the user looks along ``-z``).
* Extrinsics device frame: right-handed, ``x`` up, ``y`` left, ``z`` backwards.
* Camera: OpenCV, ``x`` right, ``y`` down, ``z`` forward.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

import parser as tracking_parser

DEFAULT_OUTPUT = os.path.join(tracking_parser.DEFAULT_DATA_DIR, "output_tracking.mp4")

# Video <-> telemetry time shift in seconds (positive = telemetry sampled later).
TIME_OFFSET_SEC = 0.0

# Which cameraExtrinsics matrix belongs to the left half of the SBS video.
EXTRINSICS_INDEX = 0

# Change of basis: extrinsics device frame (x up, y left, z back) -> OpenXR head
# frame (x right, y up, z back). Columns are the device axes expressed in head
# coordinates: x_dev -> +y_head, y_dev -> -x_head, z_dev -> +z_head.
# With it, the rotation of Camera_to_Head becomes ~diag(1, -1, -1): the camera
# looks forward (-z_head) with image x to the right and image y down, which is
# exactly the OpenCV camera convention. Without it the image would come out
# transposed (camera x pointing down), so this is a convention fix, not a fit.
EXTRINSICS_TO_HEAD = np.array([[0.0, -1.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])

MIN_DEPTH_M = 0.02   # points closer than this (or behind the camera) are not drawn

# ---- Hand skeleton (OpenXR XR_EXT_hand_tracking) ---------------------------- #
#   0 palm, 1 wrist, 2-5 thumb (metacarpal..tip), 6-10 index, 11-15 middle,
#   16-20 ring, 21-25 little (metacarpal, proximal, intermediate, distal, tip).
FINGERS = {
    "thumb": [1, 2, 3, 4, 5],
    "index": [1, 6, 7, 8, 9, 10],
    "middle": [1, 11, 12, 13, 14, 15],
    "ring": [1, 16, 17, 18, 19, 20],
    "little": [1, 21, 22, 23, 24, 25],
}
PALM_INDEX = 0
JOINT_COLOR = (255, 255, 255)

# Per-hand palettes (BGR): left hand in green tones, right hand in blue tones.
HAND_STYLES = {
    "leftHand": {
        "label": "L",
        "palm": (0, 255, 255),
        "fingers": {"thumb": (0, 255, 191), "index": (0, 230, 0), "middle": (80, 200, 0),
                    "ring": (130, 170, 0), "little": (60, 120, 0)},
    },
    "rightHand": {
        "label": "R",
        "palm": (0, 0, 255),
        "fingers": {"thumb": (255, 255, 0), "index": (255, 180, 0), "middle": (255, 110, 0),
                    "ring": (255, 0, 0), "little": (200, 0, 90)},
    },
}

# ---- Body skeleton (PICO BodyTrackerRole) ----------------------------------- #
#  0 pelvis, 1 L hip, 2 R hip, 3 spine1, 4 L knee, 5 R knee, 6 spine2, 7 L ankle,
#  8 R ankle, 9 spine3, 10 L foot, 11 R foot, 12 neck, 13 L collar, 14 R collar,
#  15 head, 16 L shoulder, 17 R shoulder, 18 L elbow, 19 R elbow, 20 L wrist,
#  21 R wrist, 22 L hand, 23 R hand.
BODY_BONES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
              (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15),
              (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
              (20, 22), (21, 23)]
BODY_LEFT = {1, 4, 7, 10, 13, 16, 18, 20, 22}
BODY_RIGHT = {2, 5, 8, 11, 14, 17, 19, 21, 23}
BODY_LABELS = {16: "L shoulder", 17: "R shoulder", 18: "L elbow", 19: "R elbow"}
BODY_COLORS = {"left": (0, 200, 255), "right": (255, 0, 255), "center": (200, 200, 200)}


# --------------------------------------------------------------------------- #
# Matrix algebra
# --------------------------------------------------------------------------- #
def quat_to_rot(q):
    """3x3 rotation matrix from a quaternion ``[qx, qy, qz, qw]`` (normalised here)."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_to_matrix(pose):
    """4x4 ``Head_to_World`` from ``Head.pose = [x, y, z, qx, qy, qz, qw]``.

    ``p_world = R(q) @ p_head + t``, i.e. ``[[R, t], [0, 1]]``.
    """
    m = np.eye(4)
    m[:3, :3] = quat_to_rot(pose[3:])
    m[:3, 3] = pose[:3]
    return m


def camera_to_head_matrix(extrinsics):
    """4x4 ``Camera_to_Head`` from a raw ``cameraExtrinsics`` matrix.

    The raw matrix maps OpenCV camera coordinates into the device frame;
    :data:`EXTRINSICS_TO_HEAD` re-expresses the result in OpenXR head axes.
    """
    return EXTRINSICS_TO_HEAD @ extrinsics


def world_to_camera_matrix(head_pose, camera_to_head):
    """``World_to_Camera = inv(Head_to_World @ Camera_to_Head)``."""
    camera_to_world = pose_to_matrix(head_pose) @ camera_to_head
    return np.linalg.inv(camera_to_world)


def build_camera_matrix(intrinsics_raw, eye_width, eye_height):
    """Build the 3x3 intrinsic matrix ``K`` for one eye of the SBS video.

    The header stores ``[cx, cy, fx, fy]`` (not ``[fx, fy, cx, cy]``): the values
    ``1079.5, 404.5`` are the exact centre of a 2160x810 image, and
    ``1373.67, 686.87`` are the focal lengths. They describe a 2160x810 native
    image that was squeezed horizontally into one 1080-pixel half of the SBS
    frame. The native size follows from the principal point
    (``W = 2 * (cx + 0.5)``) and the parameters are rescaled to the eye image::

        sx = eye_width / W,  sy = eye_height / H
        K = [[fx*sx, 0,     cx*sx],
             [0,     fy*sy, cy*sy],
             [0,     0,     1    ]]

    After scaling ``fx*sx ~= fy`` (~686.8 px), i.e. square pixels, which confirms
    the interpretation. Lens distortion is not provided and is ignored.
    """
    cx, cy, fx, fy = intrinsics_raw
    native_w, native_h = 2.0 * (cx + 0.5), 2.0 * (cy + 0.5)
    sx, sy = eye_width / native_w, eye_height / native_h
    return np.array([[fx * sx, 0.0, cx * sx],
                     [0.0, fy * sy, cy * sy],
                     [0.0, 0.0, 1.0]])


def project_points(points_world, world_to_camera, K):
    """Project (N, 3) World Space points to pixels.

    Points are lifted to homogeneous ``[x, y, z, 1]``, multiplied by
    ``World_to_Camera`` and then projected with the pinhole model using only
    ``K``. Returns ``(uv, valid)``: (N, 2) pixel coordinates and an (N,) mask of
    points in front of the camera (NaN inputs are invalid).
    """
    homogeneous = np.hstack([points_world, np.ones((len(points_world), 1))])
    p_cam = (world_to_camera @ homogeneous.T).T[:, :3]
    z = p_cam[:, 2]
    valid = z > MIN_DEPTH_M          # NaN compares False -> invalid
    safe_z = np.where(valid, z, 1.0)
    u = K[0, 0] * p_cam[:, 0] / safe_z + K[0, 2]
    v = K[1, 1] * p_cam[:, 1] / safe_z + K[1, 2]
    return np.stack([u, v], axis=1), valid


# --------------------------------------------------------------------------- #
# Synchronisation
# --------------------------------------------------------------------------- #
class TrackingSampler:
    """Samples telemetry at a fractional log index with linear interpolation.

    Head position, hand joints and body joints are interpolated linearly; the
    head orientation uses normalised linear interpolation (nlerp) of
    quaternions, which is accurate for the ~11 ms gap between 90 Hz samples.
    A hand counts as active only if both neighbouring samples are active, so
    it is never interpolated across a tracking gap.
    """

    def __init__(self, data):
        self.data = data
        self.indices = np.arange(len(data), dtype=np.float64)
        # float64 seconds relative to the first sample keep sub-ns precision
        self.rel_times = (data.timestamps - data.timestamps[0]).astype(np.float64)

    def __len__(self):
        return len(self.data)

    def index_for_time(self, t_ns):
        """Fractional log index for an absolute timestamp (clamped to the log range)."""
        return float(np.interp(t_ns - self.data.timestamps[0], self.rel_times, self.indices))

    def sample(self, frac_index):
        """Return ``(head_pose, hands_joints, hands_active, body_joints)`` at a fractional index."""
        d = self.data
        frac_index = min(max(frac_index, 0.0), len(self) - 1.0)
        i0 = int(np.floor(frac_index))
        i1 = min(i0 + 1, len(self) - 1)
        a = frac_index - i0

        def lerp(arr):
            return (1 - a) * arr[i0] + a * arr[i1]

        q0, q1 = d.head_poses[i0, 3:], d.head_poses[i1, 3:]
        if np.dot(q0, q1) < 0:          # same rotation, opposite hemisphere
            q1 = -q1
        quat = (1 - a) * q0 + a * q1
        head_pose = np.concatenate([lerp(d.head_poses[:, :3]), quat / np.linalg.norm(quat)])
        hands_active = d.hands_active[i0] & d.hands_active[i1]
        return head_pose, lerp(d.hands_joints), hands_active, lerp(d.body_joints)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def to_pixels(uv):
    return np.round(np.nan_to_num(uv)).astype(int)


def draw_hand(frame, uv, valid, style, draw_skeleton=True):
    """Draw joints and (optionally) bones of one hand onto ``frame`` in place."""
    pts = to_pixels(uv)
    if draw_skeleton:
        for name, chain in FINGERS.items():
            for a, b in zip(chain[:-1], chain[1:]):
                if valid[a] and valid[b]:
                    cv2.line(frame, tuple(pts[a]), tuple(pts[b]),
                             style["fingers"][name], 2, cv2.LINE_AA)
    for idx, (pt, ok) in enumerate(zip(pts, valid)):
        if not ok:
            continue
        if idx == PALM_INDEX:
            cv2.circle(frame, tuple(pt), 7, style["palm"], -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, tuple(pt), 4, JOINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pt), 4, (0, 0, 0), 1, cv2.LINE_AA)


def body_color(idx):
    if idx in BODY_LEFT:
        return BODY_COLORS["left"]
    if idx in BODY_RIGHT:
        return BODY_COLORS["right"]
    return BODY_COLORS["center"]


def draw_body(frame, uv, valid):
    """Draw the body skeleton; shoulders and elbows get bigger markers and labels."""
    pts = to_pixels(uv)
    for a, b in BODY_BONES:
        if valid[a] and valid[b]:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), body_color(b), 3, cv2.LINE_AA)
    for idx, (pt, ok) in enumerate(zip(pts, valid)):
        if not ok:
            continue
        radius = 9 if idx in BODY_LABELS else 5
        cv2.circle(frame, tuple(pt), radius, body_color(idx), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(pt), radius, (0, 0, 0), 1, cv2.LINE_AA)
        if idx in BODY_LABELS:
            cv2.putText(frame, BODY_LABELS[idx], (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, body_color(idx), 1, cv2.LINE_AA)


def draw_info(frame, frame_idx, log_idx, inactive_labels):
    """Overlay frame / log indices (and inactive hands) for debugging the sync."""
    text = f"frame {frame_idx}  log {log_idx:.1f}"
    if inactive_labels:
        text += f"  [inactive: {', '.join(inactive_labels)}]"
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser(description="Overlay tracked hands and body on the SBS video.")
    p.add_argument("--data-dir", default=tracking_parser.DEFAULT_DATA_DIR,
                   help="Folder to auto-detect input files in (default: %(default)s)")
    p.add_argument("--video", default=None, help="Path to CameraRecord_*.mp4 (overrides auto-detect)")
    p.add_argument("--tracking", default=None, help="Path to trackingData_*.txt (overrides auto-detect)")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output MP4 path (default: %(default)s)")
    p.add_argument("--sync", choices=["timestamp", "index"], default="timestamp",
                   help="timestamp: frame time = header timeStampNs + i/fps + TIME_OFFSET_SEC; "
                        "index: linear mapping of frame index onto log line index")
    p.add_argument("--hands", nargs="+", choices=list(tracking_parser.HANDS),
                   default=list(tracking_parser.HANDS), help="Hands to draw (default: both)")
    p.add_argument("--no-body", action="store_true", help="Do not draw Body.joints")
    p.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames")
    p.add_argument("--no-skeleton", action="store_true", help="Draw hand joints only, without bones")
    p.add_argument("--no-info", action="store_true", help="Do not overlay frame/log indices")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    video_path = args.video or tracking_parser.find_single_file(args.data_dir, tracking_parser.VIDEO_PATTERN)
    tracking_path = args.tracking or tracking_parser.find_single_file(args.data_dir, tracking_parser.TRACKING_PATTERN)
    print(f"Video:    {video_path}\nTracking: {tracking_path}")

    t0 = time.time()
    data = tracking_parser.load_tracking(tracking_path)
    sampler = TrackingSampler(data)
    print(f"Parsed {len(sampler)} telemetry samples in {time.time() - t0:.1f} s")
    hands_to_draw = [(tracking_parser.HANDS.index(name), name) for name in args.hands]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    eye_width = width // 2
    print(f"Video: {width}x{height} @ {fps:.3f} fps, {total_frames} frames -> eye {eye_width}x{height}")

    K = build_camera_matrix(data.camera.intrinsics_raw, eye_width, height)
    camera_to_head = camera_to_head_matrix(data.camera.extrinsics[EXTRINSICS_INDEX])
    np.set_printoptions(precision=4, suppress=True)
    print(f"K =\n{K}")
    print(f"Camera_to_Head (cameraExtrinsics[{EXTRINSICS_INDEX}] in head axes) =\n{camera_to_head}")
    print(f"TIME_OFFSET_SEC = {TIME_OFFSET_SEC}")

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (eye_width, height))
    if not writer.isOpened():
        sys.exit(f"Cannot open video writer: {args.output}")

    n_frames = total_frames if args.max_frames is None else min(total_frames, args.max_frames)
    t0 = time.time()
    frame_idx = 0
    while frame_idx < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = np.ascontiguousarray(frame[:, :eye_width])   # left eye of the SBS frame

        if args.sync == "timestamp":
            t_ns = data.camera.header_timestamp_ns + (frame_idx / fps + TIME_OFFSET_SEC) * 1e9
            log_idx = sampler.index_for_time(t_ns)
        else:
            log_idx = frame_idx * (len(sampler) - 1) / max(total_frames - 1, 1)

        head_pose, hands_joints, hands_active, body_joints = sampler.sample(log_idx)
        world_to_camera = world_to_camera_matrix(head_pose, camera_to_head)

        if not args.no_body:
            uv, valid = project_points(body_joints, world_to_camera, K)
            draw_body(frame, uv, valid)

        inactive_labels = []
        for h, name in hands_to_draw:
            style = HAND_STYLES[name]
            if not hands_active[h]:
                inactive_labels.append(style["label"])
                continue
            uv, valid = project_points(hands_joints[h], world_to_camera, K)
            draw_hand(frame, uv, valid, style, draw_skeleton=not args.no_skeleton)

        if not args.no_info:
            draw_info(frame, frame_idx, log_idx, inactive_labels)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  {frame_idx}/{n_frames} frames ({frame_idx / (time.time() - t0):.0f} fps)")

    cap.release()
    writer.release()
    print(f"Saved {frame_idx} frames to {args.output} in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
