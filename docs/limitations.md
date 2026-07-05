# Limitations and honest scope

This project is an **engineering-scale reference implementation**, not a production system or
a large-scale study. Specifically:

- **Scale.** Default corpora are thousands-to-millions of synthetic records, not web-scale.
  Distributed paths (DDP/FSDP, Kafka, object storage) are real code but validated on small
  clusters at most; throughput numbers are measured on the hardware stated next to them.
- **Data.** The primary corpus is synthetic (template-grammar text with planted, labeled
  defects) plus small public datasets. Synthetic data makes defect metrics exact, but absolute
  numbers do not transfer to natural corpora; treat trends, not magnitudes, as the finding.
- **Models.** The reference trainer uses a deliberately small transformer. Research results
  are about *data* effects at small scale; extrapolation to frontier scale is not claimed.
- **Measured vs illustrative numbers.** Any number in docs or reports is either (a) produced
  by a committed, seed-controlled run whose config is referenced next to it, or (b) explicitly
  labeled "illustrative". Nothing in between — benchmark numbers are never fabricated.

*(This file is updated each phase as real constraints are discovered.)*
