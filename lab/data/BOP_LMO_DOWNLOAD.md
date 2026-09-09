# Regenerating the real BOP LM-O input data

`lab/data/bop_lmo/` (raw dataset archives and extracted RGB/depth/mask/model
files) is **not** committed to this repository: it is externally licensed,
third-party data (~254 MB extracted) with its own license terms (BOP
benchmark / LineMOD-Occlusion), and `CONTRIBUTING.md`'s own policy is that raw
sensor data may stay at its institutional archive when repository inclusion
is not appropriate, as long as the archive identifier and access conditions
are stated here.

Source (public, no login required at the time this was written):

```text
https://huggingface.co/datasets/bop-benchmark/lmo/resolve/main/lmo_base.zip
https://huggingface.co/datasets/bop-benchmark/lmo/resolve/main/lmo_models.zip
https://huggingface.co/datasets/bop-benchmark/lmo/resolve/main/lmo_test_bop19.zip
```

To regenerate `lab/data/bop_lmo/`:

```bash
mkdir -p lab/data/bop_lmo && cd lab/data/bop_lmo
for u in lmo_base.zip lmo_models.zip lmo_test_bop19.zip; do
  curl -sL -o "$u" "https://huggingface.co/datasets/bop-benchmark/lmo/resolve/main/$u"
done
unzip -o -q lmo_base.zip && unzip -o -q lmo_models.zip && unzip -o -q lmo_test_bop19.zip
```

This produces `lab/data/bop_lmo/test/000002/{rgb,depth,mask,mask_visib}/`,
`scene_camera.json`, `scene_gt.json`, and `lab/data/bop_lmo/models/obj_0000{01,05,06,08,09,10,11,12}.ply`.

Then regenerate the derived episode JSONL (already committed at
`lab/data/bop_lmo_episodes/*.jsonl`, deterministic/seeded, ~832 KB, contains
only derived observable features and oracle pose-error numbers, no raw
imagery) with:

```bash
python3 lab/generate_bop_episodes.py
```

and re-run the evaluator per `lab/results/real-bop-lmo-2026-09-09/RESULT.md`.
