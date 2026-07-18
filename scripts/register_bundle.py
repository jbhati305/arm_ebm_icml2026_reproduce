"""Register the complete reproduction workspace as a Trackio artifact."""

import os
from pathlib import Path

import trackio


PROJECT = "arm-ebm-icml2026-reproduction"
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # The bundle is referenced explicitly from Conclusion. Prevent Trackio's
    # auto-note hook from creating an extra page after the canonical last page.
    os.environ["TRACKIO_LOGBOOK_AUTONOTE"] = "0"
    trackio.init(
        project=PROJECT,
        name="register-reproduction-bundle",
        embed=False,
        auto_log_cpu=False,
        auto_log_gpu=False,
    )
    artifact = trackio.Artifact(
        name="reproduction-bundle",
        type="dataset",
        description=(
            "Self-contained CPU reproduction: implementation, tests, locked "
            "environment, results, poster, and canonical Trackio logbook."
        ),
        metadata={"arxiv_id": "2512.15605", "openreview_id": "997oprE4sh"},
    )
    for path in (
        "README.md",
        "REPRODUCTION_BUNDLE.md",
        "pyproject.toml",
        "uv.lock",
        "reproduce_cpu.sh",
        ".trackio/metadata.json",
    ):
        artifact.add_file(ROOT / path, name=path)
    for directory in (
        "repro_arm_ebm",
        "tests",
        "outputs/cpu_exact",
        ".trackio/logbook",
    ):
        root = ROOT / directory
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            if "__pycache__" in file_path.parts or file_path.suffix in {
                ".pyc",
                ".pyo",
            }:
                continue
            artifact.add_file(file_path, name=file_path.relative_to(ROOT).as_posix())
    logged = trackio.log_artifact(artifact)
    trackio.finish()
    version = str(logged.version)
    if not version.startswith("v"):
        version = f"v{version}"
    print(f"Registered {logged.project}/{logged.name}:{version}")


if __name__ == "__main__":
    main()
