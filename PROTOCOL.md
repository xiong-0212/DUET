# Data de-duplication protocol

## 1. Input

The input is the CoBRA release copied verbatim to `data/raw/cobra`.
Train/validation IDs are read from the HARIBOSS+TR60 CSV files and sequence and
residue labels are read from the bindingsite file. Test sets are TE18, TL12,
JL10 and RB9.

## 2. Exact sequence identity

A duplicate sequence group contains records with byte-identical RNA strings.
Exact duplicates are handled before any alignment.

- Within each test set and across test sets, one representative is retained.
  Priority is TE18, TL12, JL10, RB9, then lexical `rna_id`.
- Training and validation records whose sequence occurs in a retained test set
  are removed.
- Training records whose sequence occurs in validation are removed.
- Within each split, one representative is retained when all members have the
  same labels.
- If the same sequence has different label vectors, the whole family is removed
  from training. A sequence-only model cannot resolve this target ambiguity.

## 3. Global sequence identity

Two sequences are compared only when:

```text
min(len(a), len(b)) / max(len(a), len(b)) >= 0.8
```

Comparable sequences are globally aligned with:

```text
match = +1, mismatch = -1, gap opening = -4, gap extension = -1
```

Selection uses:

```text
identity = matches / alignment columns
```

`matches / min(len(a), len(b))` is reported only as a containment diagnostic.
It is not used to remove records.

## 4. Exact-clean manifest

`exact_sequence_unique` is the reference manifest. It contains no repeated
sequence in any individual split, no exact sequence across train/validation,
and no exact sequence across either split and the test sets.

## 5. Stricter manifests

For each threshold in `{0.90, 0.80, 0.70, 0.50}`:

1. Remove validation records with comparable identity to any test record at or
   above the threshold.
2. Cluster remaining validation records greedily.
3. Remove training records with comparable identity to any test record at or
   above the threshold.
4. Remove training records with comparable identity to any remaining validation
   record at or above the threshold.
5. Cluster remaining training records greedily.

Greedy clustering iterates longest sequence first and lexical `rna_id` second.
A sequence joins an existing cluster when identity to that cluster's
representative reaches the threshold. This avoids transitive chaining, in which
A and C would be grouped solely because both are similar to B.

## 6. Validation

The build is accepted only when `scripts/02_validate.py` confirms:

- all per-split sequences are unique;
- exact train/test, validation/test and train/validation intersections are
  empty;
- no comparable pair above the declared threshold crosses splits;
- no comparable pair above the threshold remains within training or validation;
- retained and removed records do not overlap.

The original `duet/`, `duet_claude/`, `cobra_repro/`, `nar/` and other existing
directories are read-only inputs to this rebuild.
