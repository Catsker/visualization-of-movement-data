"""
Single CLI entry point for the project.

    python main.py video      [options]   -> src/visualize.py    (overlay MP4)
    python main.py html       [options]   -> src/visualize_3d.py (offline 3D player)
    python main.py dashboard  [options]   -> streamlit run src/dashboard.py

Everything after the command is passed to the target unchanged, e.g.
``python main.py video --max-frames 300`` or ``python main.py html --help``.
For the dashboard the extra options go to ``streamlit run`` (``--server.port 8502``);
script options follow ``--`` (``python main.py dashboard -- --tracking file.txt``).
"""

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
DASHBOARD = os.path.join(SRC, "dashboard.py")

COMMANDS = {
    "video": "render the 3D->2D skeleton overlay onto the left eye of the SBS video (MP4)",
    "html": "export the animated offline 3D player (data/skeleton_3d_animated.html)",
    "dashboard": "start the Streamlit dashboard for frame-by-frame inspection",
}


def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="main.py",
        description="XR telemetry visualization: video overlay, offline 3D player, web dashboard.",
        epilog="Commands:\n" + "\n".join(f"  {name:<10} {text}" for name, text in COMMANDS.items())
               + "\n\nRun 'python main.py <command> --help' for the command's own options.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=list(COMMANDS), help="what to run")
    p.add_argument("args", nargs=argparse.REMAINDER,
                   help="options forwarded unchanged to the command")
    return p


def run_dashboard(args):
    # Same call as `streamlit run src/dashboard.py`. Fall back to `python -m streamlit`
    # when the venv's Scripts/bin folder is not on PATH (venv not activated).
    # cwd=ROOT so Streamlit picks up .streamlit/config.toml (headless: no email
    # prompt) no matter where main.py is called from.
    launcher = ["streamlit"] if shutil.which("streamlit") else [sys.executable, "-m", "streamlit"]
    return subprocess.run(launcher + ["run", DASHBOARD] + args, cwd=ROOT).returncode


def main():
    args = build_arg_parser().parse_args()

    if args.command == "dashboard":
        return run_dashboard(args.args)

    # src/ modules import each other as top-level modules (`import parser`).
    sys.path.insert(0, SRC)
    if args.command == "video":
        import visualize as target
    else:
        import visualize_3d as target
    return target.main(args.args)


if __name__ == "__main__":
    sys.exit(main())
