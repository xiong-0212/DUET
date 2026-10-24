# Data provenance

`data/raw/cobra/` contains the small CoBRA source tables used by the data
pipeline. These files are never edited in place.

`data/clean/manifests/` contains validated split manifests with:

- retained sequences and residue labels;
- content hashes for each record;
- whole-component assignments for the 80% identity protocol;
- itemized removal ledgers;
- retained and removed counts;
- manifest-level SHA-256 digests.

`data/external/external_sets.json` contains the curated external-set records.
Small metadata files are included; cached CIF files and embedding tensors are
excluded from Git.

Downstream training reads a validated manifest from `data/clean/manifests/`,
not `data/raw/` directly. Processed graphs are written to `data/processed/` and
are also excluded from Git.
