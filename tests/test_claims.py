"""Check every claim in claims.json against the artifact that produced it.

This is the test that makes the README trustworthy rather than persuasive. The
README states numbers; claims.json binds each number to a measurement; this
file re-reads the measurements and fails if any of them moved. Regenerate the
artifacts with `make all` and this suite tells you which prose is now stale.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLAIMS = json.loads((ROOT / "claims.json").read_text())

FILTER_RE = re.compile(r"^(?P<list>\w*)\[(?P<filters>[^\]]+)\]\.(?P<field>[\w.]+)$")


def resolve(doc, query: str):
    """Resolve a query against a loaded artifact.

    Supported forms:
      field.subfield                  plain nested lookup
      rows[k=v,k2=v2].field           pick a row from a named list, then a field
      [k=v].field                     same, when the document root is a list
    """
    m = FILTER_RE.match(query)
    if m:
        seq = doc if m.group("list") == "" else doc[m.group("list")]
        wanted = {}
        for part in m.group("filters").split(","):
            k, v = part.split("=", 1)
            wanted[k.strip()] = v.strip()

        def matches(row):
            for k, v in wanted.items():
                actual = row.get(k)
                if str(actual) != v:
                    return False
            return True

        rows = [r for r in seq if matches(r)]
        if len(rows) != 1:
            raise AssertionError(f"query {query!r} matched {len(rows)} rows, expected exactly 1")
        node = rows[0]
        for key in m.group("field").split("."):
            node = node[key]
        return node

    node = doc
    for key in query.split("."):
        node = node[key]
    return node


def compare(actual, op: str, expected, tol):
    if op == "==":
        if tol is None:
            return actual == expected
        return abs(actual - expected) <= tol
    if op == "<":
        return actual < expected
    if op == "<=":
        return actual <= expected
    if op == ">":
        return actual > expected
    if op == ">=":
        return actual >= expected
    raise ValueError(f"unsupported op {op!r}")


def test_claims_file_is_well_formed():
    ids = [c["id"] for c in CLAIMS["claims"]]
    assert len(ids) == len(set(ids)), "duplicate claim ids"
    for c in CLAIMS["claims"]:
        assert c["text"].strip(), c["id"]
        assert c["op"] in {"==", "<", "<=", ">", ">="}, c["id"]


@pytest.mark.parametrize("claim", CLAIMS["claims"], ids=[c["id"] for c in CLAIMS["claims"]])
def test_readme_claim_still_holds(claim):
    path = ROOT / claim["artifact"]
    if not path.exists():
        pytest.skip(f"{claim['artifact']} not generated yet -- run `make all`")
    doc = json.loads(path.read_text())
    actual = resolve(doc, claim["query"])
    ok = compare(actual, claim["op"], claim["expected"], claim.get("tol"))
    assert ok, (
        f"\nCLAIM FAILED: {claim['id']}\n"
        f"  README says: {claim['text']}\n"
        f"  claims.json: {claim['query']} {claim['op']} {claim['expected']}"
        f"{' (tol ' + str(claim['tol']) + ')' if claim.get('tol') else ''}\n"
        f"  measured:    {actual}\n"
        f"  source:      {claim['artifact']}\n"
    )
