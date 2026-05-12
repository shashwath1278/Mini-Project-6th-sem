"""
UniProt REST ID mapping (async job): map RefSeq / INSDC accessions → UniProtKB.

Submit with **application/x-www-form-urlencoded** body fields from, to, ids
(not JSON alone — the API returns 400 if ids are only in JSON without form).
"""

from __future__ import annotations

import time
from typing import Any

import requests

IDMAP_RUN = "https://rest.uniprot.org/idmapping/run"
IDMAP_STATUS = "https://rest.uniprot.org/idmapping/status/{job_id}"
IDMAP_RESULTS = "https://rest.uniprot.org/idmapping/results/{job_id}"


def _first_uniprot_accession(result_row: dict[str, Any]) -> str | None:
    to = result_row.get("to")
    if isinstance(to, dict):
        return to.get("primaryAccession")
    if isinstance(to, str):
        return to
    return None


def run_id_mapping(
    session: requests.Session,
    from_db: str,
    to_db: str,
    ids: list[str],
    *,
    poll_interval_s: float = 0.6,
    max_wait_s: float = 180.0,
) -> dict[str, str]:
    """
    Map external IDs to UniProt accessions. Returns dict external_id → UniProt AC
    (one chosen target per source id when multiple hits exist).
    """
    if not ids:
        return {}
    r = session.post(
        IDMAP_RUN,
        data={
            "from": from_db,
            "to": to_db,
            "ids": "\n".join(ids),
        },
        timeout=60,
    )
    r.raise_for_status()
    job_id = r.json()["jobId"]
    deadline = time.monotonic() + max_wait_s
    payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        st = session.get(IDMAP_STATUS.format(job_id=job_id), timeout=60)
        st.raise_for_status()
        data = st.json()
        # Some responses include `results` while `jobStatus` is absent.
        if data.get("results") is not None:
            payload = data
            break
        status = data.get("jobStatus")
        if status == "FINISHED":
            res = session.get(
                IDMAP_RESULTS.format(job_id=job_id),
                headers={"Accept": "application/json"},
                timeout=60,
            )
            res.raise_for_status()
            payload = res.json()
            break
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"UniProt idmapping job failed: {data}")
        time.sleep(poll_interval_s)
    if payload is None:
        raise TimeoutError(f"UniProt idmapping job {job_id} did not finish in time")

    out: dict[str, str] = {}
    for row in payload.get("results") or []:
        ext = row.get("from")
        acc = _first_uniprot_accession(row)
        if ext and acc and ext not in out:
            out[ext] = acc
    return out


def map_one(
    session: requests.Session,
    from_db: str,
    external_id: str,
    *,
    to_db: str = "UniProtKB",
) -> str | None:
    m = run_id_mapping(session, from_db, to_db, [external_id])
    return m.get(external_id)
