# First server smoke evaluation

Both runs completed five training episodes with a 100-action cap. This is an
engineering smoke test, not a paper benchmark or a calibrated method evaluation.
The active configuration produced zero active-verification triggers; therefore
these results do not establish any benefit from viewpoint selection.

The archived logs predate source-ID restoration: Habitat rewrote episode IDs to
0..4 when loading this combined split. `split_manifest.json` maps each index in
its `keys` list back to the original scene and source episode ID. Both runs used
the exact same split hash. Subsequent runs preserve source IDs in output logs.

The forced-review diagnostic is separate and must never be included in efficacy
tables. Calibration and full-budget evaluation are still required.
