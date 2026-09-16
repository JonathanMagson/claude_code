#!/usr/bin/env python3
"""Optionally download the Sentinel-1 SLC and GRD zips beside the GA NRB rasters.

Reads the manifest written by ``make_before_after.py`` and, for each
acquisition, fetches the Level-1 products named in it from ASF into the *same*
folder as that acquisition's NRB rasters::

    data/before_after/hunter/t009_019128_iw3/20240603/
        ga_s1a_nrb_..._VV-gamma0.tif          <- already there
        ga_s1a_nrb_..._VH-gamma0.tif
        ga_s1a_nrb_..._mask.tif
        S1A_IW_SLC__1SDV_...zip               <- added here
        S1A_IW_GRDH_1SDV_...zip

Sizes are the thing to decide about first. An IW SLC is roughly 4 GB and a GRD
roughly 1 GB, so a full before/after set over three areas is on the order of
30 GB. ``--dry-run`` reports the real figures from ASF's metadata before
anything is fetched, and ``--annotation-only`` pulls just ``manifest.safe``
and the annotation XML - the calibration, noise and burst-id records, a few MB
- by reading the remote zip's central directory instead of the whole archive.

ASF needs an Earthdata login. Credentials are taken, in order, from
``~/.netrc`` (machine ``urs.earthdata.nasa.gov``), then the environment
(``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``), then ``--username`` with a
prompt. Register free at https://urs.earthdata.nasa.gov/.

Usage::

    python tools/fetch_s1_level1.py --dry-run
    python tools/fetch_s1_level1.py --products grd
    python tools/fetch_s1_level1.py --annotation-only
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_MANIFEST = Path("data/before_after/before_after.csv")
EARTHDATA_HOST = "urs.earthdata.nasa.gov"

#: Rough guide only; the dry run reports ASF's actual figures.
TYPICAL_BYTES = {"slc": 4_000_000_000, "grd": 1_000_000_000}


class AuthError(RuntimeError):
    """No usable Earthdata credentials."""


def _require_asf():
    try:
        import asf_search  # noqa: F401
    except ImportError:
        raise SystemExit(
            "needs asf_search:\n"
            "  conda install -c conda-forge asf_search -y\n"
            "  (or: python -m pip install asf_search)"
        )


def netrc_files() -> List[Path]:
    """Where a netrc might live, including the name Windows tools use."""
    home = Path.home()
    return [home / ".netrc", home / "_netrc"]


def credentials(username: Optional[str] = None) -> Tuple[str, str, str]:
    """Earthdata credentials, with the source they came from.

    The source matters when a login is rejected: a stale password in a netrc
    file that was written years ago fails in exactly the same way as a typo,
    and without knowing which one was used there is nothing to check.
    """
    if username is None:
        for path in netrc_files():
            if not path.exists():
                continue
            try:
                import netrc

                auth = netrc.netrc(str(path)).authenticators(EARTHDATA_HOST)
            except Exception as exc:
                print(f"  could not read {path}: {exc}", file=sys.stderr)
                continue
            if auth and auth[0] and auth[2]:
                return auth[0], auth[2], f"{path} (machine {EARTHDATA_HOST})"

        env_user = os.environ.get("EARTHDATA_USERNAME")
        env_pass = os.environ.get("EARTHDATA_PASSWORD")
        if env_user and env_pass:
            return env_user, env_pass, "EARTHDATA_USERNAME / EARTHDATA_PASSWORD"

    user = username or os.environ.get("EARTHDATA_USERNAME")
    if not user:
        raise AuthError(
            "no Earthdata credentials found.\n"
            f"  add a netrc entry for {EARTHDATA_HOST} ({' or '.join(str(p) for p in netrc_files())}),\n"
            "  or set EARTHDATA_USERNAME / EARTHDATA_PASSWORD, or pass --username.\n"
            "  register free at https://urs.earthdata.nasa.gov/"
        )
    password = os.environ.get("EARTHDATA_PASSWORD")
    if password:
        return user, password, "--username with EARTHDATA_PASSWORD"

    import getpass

    password = getpass.getpass(f"Earthdata password for {user}: ")
    return user, password, "typed at the prompt"


AUTH_HELP = """Earthdata rejected the login.

Three things account for most of these:

1. The username is not the email address. Earthdata logins have a separate
   username, and signing in with the email fails exactly like a wrong
   password. Check yours at https://urs.earthdata.nasa.gov/profile
2. The credentials came from a file you had forgotten about - the source is
   printed above. A netrc entry written for an older password fails silently
   in this way.
3. The account exists but has never accepted ASF's licence agreement. Sign in
   once at https://search.asf.alaska.edu/ and download anything by hand; that
   clears it.

To try a different account without editing anything:

    python tools/fetch_s1_level1.py --products grd --username YOUR_USERNAME"""


def use_system_certs(ca_bundle: Optional[Path] = None) -> str:
    """Make Python trust what the operating system trusts.

    On a network that inspects TLS, every request is re-signed by the
    organisation's own root CA. Windows and macOS trust that CA because the
    machine is managed; Python does not, because ``requests`` ships its own
    ``certifi`` bundle and looks nowhere else. The result is
    ``CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate
    chain`` against a site that opens fine in a browser.

    The corporate CA is legitimately trusted here - IT installed it on the
    machine - so the fix is to let Python see it, not to stop checking.
    Turning verification off would clear the error too, and would hand anyone
    on the path the ability to serve whatever they like in place of a
    multi-gigabyte file, so it is not offered.
    """
    if ca_bundle:
        path = str(Path(ca_bundle).resolve())
        os.environ["REQUESTS_CA_BUNDLE"] = path
        os.environ["SSL_CERT_FILE"] = path
        return f"using CA bundle {path}"

    if os.environ.get("REQUESTS_CA_BUNDLE"):
        return f"using CA bundle from the environment: {os.environ['REQUESTS_CA_BUNDLE']}"

    exported = export_system_ca_bundle()
    if exported:
        os.environ["REQUESTS_CA_BUNDLE"] = str(exported)
        os.environ["SSL_CERT_FILE"] = str(exported)
        return f"using system + certifi CA bundle: {exported}"
    return ""


def export_system_ca_bundle(cache: Optional[Path] = None) -> Optional[Path]:
    """Write the OS trust store plus certifi to one PEM, and return its path.

    ``truststore.inject_into_ssl()`` looks like the obvious answer and is not:
    it replaces ``ssl.SSLContext`` globally, and urllib3 setting
    ``context.verify_mode`` on the replacement recurses until the stack runs
    out. Exporting instead leaves every library's SSL machinery untouched -
    only the list of trusted roots changes.

    certifi is included as well as the OS roots, because a network that
    inspects HTTPS usually inspects only some hosts; the rest still present
    ordinary public certificates and must keep verifying.

    Returns None off Windows, where ``ssl.enum_certificates`` does not exist
    and the platform's roots are normally what Python already uses.
    """
    import ssl

    if not hasattr(ssl, "enum_certificates"):
        return None

    pems: List[str] = []
    for store in ("ROOT", "CA"):
        try:
            for cert, encoding, _trust in ssl.enum_certificates(store):
                if encoding == "x509_asn":
                    pems.append(ssl.DER_cert_to_PEM_cert(cert))
        except Exception:
            continue
    if not pems:
        return None

    try:
        import certifi

        base = Path(certifi.where()).read_text(encoding="utf-8")
    except Exception:
        base = ""

    cache = cache or Path.home() / ".cache" / "vegmon" / "ca-bundle.pem"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(base + "\n" + "\n".join(pems), encoding="utf-8")
    return cache


TLS_HELP = """TLS verification failed - the certificate chain ends in one your
Python does not trust. On a network that inspects HTTPS this is expected: the
proxy re-signs traffic with your organisation's root CA, which Windows trusts
and Python's bundled certifi does not.

On Windows this script now exports the OS trust store to a PEM automatically
and points requests at it, so re-running is usually enough. If the export
found nothing, get your organisation's root CA as a .pem - your browser can
export it from the certificate viewer, or IT can supply it - and pass it:

    python tools/fetch_s1_level1.py --ca-bundle C:\\path\\to\\corporate-root.pem

Do not disable certificate verification to get past this."""


def session(username: Optional[str] = None, ca_bundle: Optional[Path] = None):
    import asf_search as asf

    # run() has already set the trust store up; do not redo it or say so twice.
    user, password, source = credentials(username)
    print(f"  Earthdata user {user!r}, credentials from {source}", flush=True)
    try:
        return asf.ASFSession().auth_with_creds(user, password)
    except Exception as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc) or "SSLError" in type(exc).__name__:
            raise SystemExit(f"\n{TLS_HELP}\n\noriginal error: {exc}")
        if "ASFAuthenticationError" in type(exc).__name__ or "incorrect" in str(exc).lower():
            raise SystemExit(f"\n{AUTH_HELP}\n\noriginal error: {exc}")
        raise


def product_bytes(product) -> int:
    """Size of a product in bytes, tolerating the ways ASF reports it."""
    props = (product.properties or {}) if product is not None else {}
    for key in ("bytes", "sizeMB", "fileSize"):
        value = props.get(key)
        if value in (None, "", 0):
            continue
        try:
            size = float(value)
        except (TypeError, ValueError):
            continue
        return int(size * 1_000_000) if key == "sizeMB" else int(size)
    return 0


def _is_metadata(product) -> bool:
    """True for ASF's metadata-only companion products.

    A granule has more than one product under the same ``sceneName`` - the
    data archive and a small metadata record. They are indistinguishable by
    name, so keying on the name alone silently picks whichever came last, and
    a few-megabyte XML stands in for a four-gigabyte SLC.
    """
    props = (product.properties or {}) if product is not None else {}
    level = str(props.get("processingLevel", "")).upper()
    url = str(props.get("url", "")).lower()
    return level.startswith("METADATA") or url.endswith((".iso.xml", ".xml", ".png"))


def lookup(scene_ids: Sequence[str], debug: bool = False) -> Dict[str, object]:
    """ASF data products for a list of scene ids, keyed by scene name.

    Where a granule has several products, the metadata companions are dropped
    and the largest remaining one is kept - that is the archive to download.
    """
    import asf_search as asf

    candidates: Dict[str, List[object]] = {}
    # granule_search takes the full list, but a single bad id can empty the
    # response, so ask in small batches and keep what comes back.
    for start in range(0, len(scene_ids), 20):
        batch = list(scene_ids[start:start + 20])
        try:
            results = asf.granule_search(batch)
        except Exception as exc:
            if "CERTIFICATE_VERIFY_FAILED" in str(exc):
                raise SystemExit(f"\n{TLS_HELP}\n\noriginal error: {exc}")
            print(f"  ASF query failed for {len(batch)} scene(s): {exc}", file=sys.stderr)
            continue
        for product in results:
            name = (product.properties or {}).get("sceneName", "")
            if name:
                candidates.setdefault(name, []).append(product)

    found: Dict[str, object] = {}
    for name, products in candidates.items():
        if debug:
            print(f"\n  {name}: {len(products)} ASF product(s)")
            for product in products:
                props = product.properties or {}
                print(f"    level={props.get('processingLevel')!r:<18}"
                      f"bytes={product_bytes(product):>14,}  "
                      f"{'METADATA' if _is_metadata(product) else 'data':<9}"
                      f"{str(props.get('url'))[-60:]}")
        data = [p for p in products if not _is_metadata(p)] or products
        found[name] = max(data, key=product_bytes)
    return found


def read_manifest(path: Path, products: Sequence[str]) -> List[dict]:
    """Rows of (scene id, product type, destination folder) from the manifest."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    root = path.parent
    jobs: List[dict] = []
    for row in rows:
        folder = root / row["aoi"] / row["ga_burst_id"] / row["acquired"].replace("-", "")
        for kind in products:
            scene = row.get(f"{kind}_scene_id", "")
            if scene:
                jobs.append({
                    "scene": scene,
                    "kind": kind,
                    "dest": folder,
                    "aoi": row["aoi"],
                    "role": row.get("role", ""),
                    "acquired": row.get("acquired", ""),
                })
    return jobs


def annotation_only(product, dest: Path) -> List[Path]:
    """Pull manifest.safe and the annotation XML without the measurement data.

    A zip's central directory lists every member with its offset, so the few
    files that carry the calibration, noise and burst-id records can be ranged
    out of a multi-gigabyte archive directly.
    """
    try:
        from remotezip import RemoteZip
    except ImportError:
        raise SystemExit(
            "--annotation-only needs remotezip:\n  python -m pip install remotezip"
        )

    url = (product.properties or {}).get("url")
    if not url:
        return []
    out = dest / ((product.properties or {}).get("sceneName", "scene") + ".annotation")
    out.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    with RemoteZip(url) as archive:
        wanted = [
            name for name in archive.namelist()
            if name.endswith("manifest.safe")
            or ("/annotation/" in name and name.endswith(".xml"))
        ]
        for name in wanted:
            target = out / Path(name).name
            if target.exists() and target.stat().st_size:
                written.append(target)
                continue
            with archive.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            written.append(target)
    return written


def run(
    manifest: Path,
    products: Sequence[str],
    username: Optional[str] = None,
    dry_run: bool = False,
    annotation: bool = False,
    debug: bool = False,
    ca_bundle: Optional[Path] = None,
) -> int:
    _require_asf()
    note = use_system_certs(ca_bundle)
    if note:
        print(note)
    if not manifest.exists():
        print(f"manifest not found: {manifest}\nrun tools/make_before_after.py first",
              file=sys.stderr)
        return 1

    jobs = read_manifest(manifest, products)
    if not jobs:
        print(f"no {'/'.join(products)} scenes in {manifest}", file=sys.stderr)
        return 1

    scenes = sorted({j["scene"] for j in jobs})
    print(f"{len(jobs)} download(s), {len(scenes)} distinct scene(s)")
    print("querying ASF...", flush=True)
    records = lookup(scenes, debug=debug)

    missing = [s for s in scenes if s not in records]
    if missing:
        print(f"\n{len(missing)} scene(s) not found at ASF:", file=sys.stderr)
        for scene in missing:
            print(f"  {scene}", file=sys.stderr)

    total = 0
    estimated = 0
    for job in jobs:
        size = product_bytes(records.get(job["scene"]))
        if not size:
            size = TYPICAL_BYTES.get(job["kind"], 0)
            estimated += 1
        total += size

    print(f"\ntotal to download: {total / 1e9:.1f} GB"
          + (f"  ({estimated} size(s) estimated, ASF reported none)" if estimated else "")
          + ("  (annotation only, so far less in practice)" if annotation else ""))

    if dry_run:
        print()
        for job in jobs:
            product = records.get(job["scene"])
            size = product_bytes(product)
            if product is None:
                state = "missing at ASF"
            elif size:
                state = f"{size / 1e9:5.2f} GB"
            else:
                state = "size unknown"
            print(f"  {job['aoi']:<10}{job['role']:<7}{job['kind'].upper():<5}"
                  f"{state:>16}  -> {job['dest']}")
        if estimated:
            print("\n  re-run with --debug to see every product ASF returned per granule")
        return 0 if not missing else 1

    asf_session = session(username, ca_bundle)

    # The same acquisition can serve two areas. Copy the local file rather than
    # pulling four gigabytes twice.
    fetched: Dict[str, Path] = {}
    for job in jobs:
        product = records.get(job["scene"])
        if product is None:
            continue
        job["dest"].mkdir(parents=True, exist_ok=True)

        if annotation:
            written = annotation_only(product, job["dest"])
            print(f"  {job['scene']}: {len(written)} annotation file(s) -> {job['dest']}",
                  flush=True)
            continue

        target = job["dest"] / f"{job['scene']}.zip"
        if target.exists() and target.stat().st_size:
            print(f"  {job['scene']}: already present, skipped", flush=True)
            fetched.setdefault(job["scene"], target)
            continue

        source = fetched.get(job["scene"])
        if source and source.exists():
            shutil.copy2(source, target)
            print(f"  {job['scene']}: copied from {source.parent}", flush=True)
            continue

        print(f"  {job['scene']}: downloading -> {job['dest']}", flush=True)
        product.download(path=str(job["dest"]), session=asf_session)
        downloaded = job["dest"] / f"{job['scene']}.zip"
        if downloaded.exists():
            fetched[job["scene"]] = downloaded
            print(f"    {downloaded.stat().st_size:,} bytes", flush=True)

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help=f"before_after.csv to read (default {DEFAULT_MANIFEST})")
    ap.add_argument("--products", nargs="*", default=["slc", "grd"],
                    choices=["slc", "grd"])
    ap.add_argument("--username", default=None, help="Earthdata username")
    ap.add_argument("--annotation-only", action="store_true",
                    help="fetch manifest.safe and annotation XML instead of the zip")
    ap.add_argument("--dry-run", action="store_true",
                    help="report sizes and destinations, download nothing")
    ap.add_argument("--debug", action="store_true",
                    help="print every product ASF returns for each granule")
    ap.add_argument("--ca-bundle", type=Path, default=None,
                    help="PEM of your organisation's root CA, if TLS is inspected")
    args = ap.parse_args()

    try:
        return run(args.manifest, args.products, args.username,
                   args.dry_run, args.annotation_only, args.debug, args.ca_bundle)
    except AuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
