# Conclusion


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_bundle_artifact", "created_at": "2026-07-19T00:00:06+00:00", "title": "Reproduction bundle", "artifact": "arm-ebm-icml2026-reproduction/reproduction-bundle:latest", "artifact_type": "dataset"}
-->
**📦 Artifact** `arm-ebm-icml2026-reproduction/reproduction-bundle:latest` · dataset

https://huggingface.co/buckets/jbhati305/arm-ebm-icml2026-artifacts#arm-ebm-icml2026-reproduction/reproduction-bundle:latest


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bundle_instructions", "created_at": "2026-07-19T00:00:07+00:00", "title": "Download and rerun"}
-->
The bundle contains the implementation, tests, locked environment, raw CSVs,
HTML figures, JSON report, Posterly source, and Trackio logbook. Download the
artifact, then run `./reproduce_cpu.sh`. The script recreates the environment,
runs all regression tests, and regenerates `outputs/cpu_exact/`.
