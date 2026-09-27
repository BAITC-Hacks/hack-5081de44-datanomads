# Offline Data/ML contracts

Use Python 3.11.15 for the training environment. The runtime service installs
`requirements.txt` separately; training packages stay out of its default image.

```bash
uv venv --python 3.11.15 .venv
uv pip sync --python .venv/bin/python ml-service/training/requirements.lock
PYTHONPATH=ml-service .venv/bin/python -m unittest discover \
  -s ml-service/training/tests -v
```

The lock resolves 43 packages for Python 3.11.15 on Linux x86_64. Direct
requirements and the core version constraints are kept separately so the lock
can be regenerated when training code changes. This lock pins package versions,
but does not pin wheel hashes or other operating systems.

`contracts.py` validates dataset, classifier, embedder and evaluation manifests.
All require explicit synthetic origin and SHA-256 evidence. Dataset versions
also require a saved seed, source checksums, a group split policy and a frozen
evaluation version. The model schemas are for future real training outputs;
the existing deterministic baseline and synthetic classifier demo are not
retroactively promoted to these contracts.

Offline evaluators must receive a local artifact path and run with
`HF_HUB_OFFLINE=1` to avoid Hub checks or downloads.
