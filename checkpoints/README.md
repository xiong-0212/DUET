# Main DUET checkpoints

This lightweight release includes the 30 checkpoints used for the reported
three-tower results:

- graph tower: 5 development + 5 confirmation seeds
- TCN tower: 5 development + 5 confirmation seeds
- DenseNet tower: 5 development + 5 confirmation seeds

`scripts/05_train_duet.py` skips a run when its checkpoint is already present.
Evaluation and reporting scripts read these files from
`checkpoints/controlled/`.

See `../README_WEIGHTS.md` and `../metadata/repository.sha256` for the inventory
and SHA-256 checksums.
