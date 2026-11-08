"""Small Onshape REST client using API keys, plus helpers for its JSON shapes.

Onshape answers in one of two JSON styles depending on the client:
  "message" style: {"type": 134, "typeName": "BTMFeature", "message": {...}}
  "btType"  style: {"btType": "BTMFeature-134", ...}
The client looks at what the server sends back and writes in the same style.
"""
from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

ACCEPT = "application/json;charset=UTF-8; qs=0.09"
URL_RE = re.compile(r"/documents/([0-9a-f]{24})/([wvm])/([0-9a-f]{24})/e/([0-9a-f]{24})")


class OnshapeError(RuntimeError):
    """An Onshape request failed or returned something unexpected."""


@dataclass(frozen=True)
class ElementRef:
    """Points at one Part Studio tab: document / workspace|version|microversion / element."""

    base_url: str
    did: str
    wvm: str
    wvmid: str
    eid: str

    @property
    def path(self) -> str:
        return f"d/{self.did}/{self.wvm}/{self.wvmid}/e/{self.eid}"

    @property
    def writable(self) -> bool:
        return self.wvm == "w"


def parse_url(url: str) -> ElementRef:
    m = URL_RE.search(url)
    if not m:
        raise ValueError(
            "Expected a Part Studio URL like https://cad.onshape.com/documents/<did>/w/<wid>/e/<eid>"
        )
    p = urlparse(url)
    return ElementRef(f"{p.scheme}://{p.netloc}", *m.groups())


# ---------------------------------------------------------------- JSON styles
class Node:
    """A typed Onshape object, written out in either JSON style."""

    def __init__(self, type_id: int, type_name: str, **fields):
        self.type_id, self.type_name, self.fields = type_id, type_name, fields


BTTYPE_KEYS = {"geometryIds": "deterministicIds"}   # same field, different name per style


def serialize(obj, style: str):
    if isinstance(obj, Node):
        fields = {k: serialize(v, style) for k, v in obj.fields.items()}
        if style == "message":
            return {"type": obj.type_id, "typeName": obj.type_name, "message": fields}
        fields = {BTTYPE_KEYS.get(k, k): v for k, v in fields.items()}
        return {"btType": f"{obj.type_name}-{obj.type_id}", **fields}
    if isinstance(obj, (list, tuple)):
        return [serialize(v, style) for v in obj]
    if isinstance(obj, dict):
        return {k: serialize(v, style) for k, v in obj.items()}
    return obj


def detect_style(features_response: dict) -> str:
    for f in features_response.get("features", []):
        if "typeName" in f and "message" in f:
            return "message"
        if "btType" in f:
            return "btType"
    return "message"


def fields(obj: dict) -> dict:
    """Return the payload of an object in either style."""
    if isinstance(obj, dict) and "message" in obj and ("typeName" in obj or "type" in obj):
        return obj["message"]
    return obj


def fs_value(v):
    """Decode a FeatureScript evaluation result (arrays, strings, numbers, booleans)."""
    if isinstance(v, dict):
        payload = fields(v)
        val = payload.get("value")
        if isinstance(val, list):
            return [fs_value(x) for x in val]
        return fs_value(val) if isinstance(val, dict) else val
    return v


# ---------------------------------------------------------------- client
class Client:
    def __init__(self, access_key: str | None = None, secret_key: str | None = None, session=None):
        access_key = access_key or os.environ.get("ONSHAPE_ACCESS_KEY")
        secret_key = secret_key or os.environ.get("ONSHAPE_SECRET_KEY")
        if not access_key or not secret_key:
            raise OnshapeError(
                "Set ONSHAPE_ACCESS_KEY and ONSHAPE_SECRET_KEY "
                "(create a key pair at https://dev-portal.onshape.com/keys)."
            )
        token = base64.b64encode(f"{access_key}:{secret_key}".encode()).decode()
        self.session = session or requests.Session()
        self.session.headers.update(
            {"Authorization": f"Basic {token}", "Accept": ACCEPT, "Content-Type": "application/json"}
        )

    def _request(self, method: str, ref: ElementRef, path: str, **kw):
        url = f"{ref.base_url}/api/{path}"
        r = self.session.request(method, url, timeout=180, **kw)
        if r.status_code >= 400:
            raise OnshapeError(f"{method} {url} -> HTTP {r.status_code}: {r.text[:400]}")
        return r.json() if r.content else None

    # read
    def parts(self, ref: ElementRef) -> list[dict]:
        return self._request("GET", ref, f"parts/{ref.path}")

    def tessellation(self, ref: ElementRef, part_ids=None) -> list[dict]:
        params = {
            "outputFaceAppearances": "false",
            "outputVertexNormals": "false",
            "outputFacetNormals": "true",
            "outputTextureCoordinates": "false",
            "outputIndexTable": "false",
            "outputErrorFaces": "false",
        }
        if part_ids:
            params["partId"] = list(part_ids)
        return self._request("GET", ref, f"partstudios/{ref.path}/tessellatedfaces", params=params)

    def features(self, ref: ElementRef) -> dict:
        return self._request("GET", ref, f"partstudios/{ref.path}/features")

    def volumes(self, ref: ElementRef) -> dict[str, float]:
        """Volume of each solid body in m^3."""
        mp = self._request("GET", ref, f"partstudios/{ref.path}/massproperties", params={"massAsGroup": "false"})
        return {k: v["volume"][0] for k, v in (mp.get("bodies") or {}).items()}

    def eval_featurescript(self, ref: ElementRef, script: str):
        res = self._request("POST", ref, f"partstudios/{ref.path}/featurescript", data=json.dumps({"script": script, "queries": []}))
        return fs_value(res.get("result"))

    # write
    def add_feature(self, ref: ElementRef, feature: Node) -> dict:
        """Append a feature; returns {"featureId", "status"}."""
        current = self.features(ref)
        style = detect_style(current)
        body = {
            "serializationVersion": current.get("serializationVersion"),
            "sourceMicroversion": current.get("sourceMicroversion"),
            "rejectMicroversionSkew": False,
            "feature": serialize(feature, style),
        }
        res = self._request("POST", ref, f"partstudios/{ref.path}/features", data=json.dumps(body))
        feat = fields(res.get("feature", {}))
        state = fields(res.get("featureState", {}))
        return {"featureId": feat.get("featureId"), "status": state.get("featureStatus"), "style": style}

    def delete_feature(self, ref: ElementRef, feature_id: str) -> None:
        self._request("DELETE", ref, f"partstudios/{ref.path}/features/featureid/{feature_id}")
