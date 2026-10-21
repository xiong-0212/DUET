# Checksums

`repository.sha256` contains SHA-256 checksums for the benchmark data and the
30 released model checkpoints included in this repository.

`release_assets.sha256` contains checksums for the two processed graph files
that must be downloaded from GitHub Releases and placed under
`data/processed/`.

Verify repository files from the project root:

```bash
sha256sum -c metadata/repository.sha256
```
