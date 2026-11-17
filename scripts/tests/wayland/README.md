# Ordinary application Wayland reproduction

This optional maintainer experiment runs the installed phy 2.1.0 release and the
mounted checkout under Debian 13, Weston, and Mesa llvmpipe. It uses a disposable
copy of a public test dataset and does not access the host desktop or phy settings.
Docker is required; the normal phy installation does not use these files.

From the repository root, build the isolated environment:

```bash
docker build --target base -t phy-wayland-reproduction:base \
    -f scripts/tests/wayland/Dockerfile .
```

Use an existing **public or disposable, self-contained** Template GUI test
dataset. Its `params.py` must reference files within that directory. For example,
the public phylib test data normally live in `~/.phy/test_data/template`.

```bash
phy_wayland_evidence=$(mktemp -d)
docker run --rm --shm-size=256m \
    --mount "type=bind,src=$PWD,dst=/src,readonly" \
    --mount "type=bind,src=$HOME/.phy/test_data/template,dst=/data,readonly" \
    --mount "type=bind,src=$phy_wayland_evidence,dst=/work" \
    phy-wayland-reproduction:base
```

The image runs as UID/GID 1000. The new evidence directory must be writable by
that user. Source data and the checkout are mounted read-only. Fresh cache,
configuration and GUI state are isolated under the evidence directory. No
curation actions or saves are requested. The dataset copy can acquire normal
application bookkeeping; the source dataset remains unchanged.

`--source release` or `--source candidate` selects one implementation. The
default compares both, using the checkout through `PYTHONPATH=/src` for the
candidate. `--scale 1` changes the default compositor scale of 2.
`--without-raw-data` removes top-level `.dat`/`.bin` files from the disposable
copy to exercise the missing-recording case. It never deletes source files.

The runner saves `summary.json`, detailed per-run JSON traces, application logs,
stdout/stderr, and Weston logs in a new subdirectory. Each app observation is
bounded by a watchdog and the runner also imposes a subprocess timeout.

## What it checks

The application uses its normal `app.exec_()` event loop and default dock layout.
Timers select three clusters, including two actual Qt table clicks, and deliver
a wheel event to the exposed Correlogram View when available. Passive hooks
record selections, update requests, paints, swaps, exposure, geometry, DPR and
pan/zoom. They do not request extra redraws, pump events, read framebuffers, or
replace data providers. Hidden/unexposed views are explicitly skipped.

A `PASS` means the observed callback/paint sequence progressed and zoom changed;
it does **not** establish pixel correctness, physical presentation, physical
input delivery, or correctness with another dataset. Qt/Python/worker errors
invalidate the result. Verify that error handling itself works with the control:

```bash
# Append these arguments to docker run above; this must report FAIL and exit 1.
--source release --inject-worker-error
```

The package versions in `requirements.txt` match the affected installation's
recorded inventory. Python 3.12.13 substitutes for its reported 3.12.16, which
was unavailable through uv. Unrecorded transitive dependencies and Debian's
graphics libraries are not exact historical pins. The original six Phy source
hashes were independently matched against the installed 2.1.0 wheel.

## Optional Sommelier proxy

The optional target builds ChromiumOS Sommelier from a fixed source commit:

```bash
docker build --target sommelier -t phy-wayland-reproduction:sommelier \
    -f scripts/tests/wayland/Dockerfile .
```

Use the same mounts with image `phy-wayland-reproduction:sommelier` and argument
`--sommelier`. Its noop driver forwards clients to the isolated Weston socket.
This exercises the proxy but does not emulate ChromeOS Exo, virtgpu/virtwl,
Chromebook input delivery, or the affected machine's exact Sommelier version.

See [the investigation](../../../design/linux-wayland-investigation.md) for
results and the boundary on automatic backend changes.
