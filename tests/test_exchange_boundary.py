"""Canonical RFS boundary tests: no provider, credentials or generated code."""
import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from rfs.contract import scientific_fingerprint, sha256_text, canonical_json, validate_figure_spec
from rfs.interop import (RFSValidationError, assert_compatible, descriptor, json_bytes, load_spec,
                         prepare_request, read_object, request_argv, validate_request, verify_result)
from rfs.artifacts import export_bundle
from skill.exchange import main


def spec_fixture():
    source = "Input goes to Encoder."
    evidence = [{"source_id": "source", "quote": source}]
    return {"schema_version": "1.0.0", "figure_id": "fixture", "kind": "diagram",
            "nodes": [{"id": "input", "label": "Input", "kind": "input", "evidence": evidence},
                      {"id": "encoder", "label": "Encoder", "kind": "module", "evidence": evidence}],
            "edges": [{"id": "flow", "source": "input", "target": "encoder", "kind": "data_flow", "directed": True, "evidence": evidence}],
            "groups": [], "labels": [], "datasets": [],
            "sources": [{"id": "source", "kind": "user_text", "content": source, "sha256": sha256_text(source)}],
            "constraints": {"locked": True, "required_labels": ["Input", "Encoder"], "layout": [], "visual": [], "negative": []},
            "design": {"layout": "left to right", "composition": "two objects", "palette": ["#326699"], "typography": "exact labels", "candidate_variation": ""},
            "provenance": {"origin": "user_supplied", "scientific_verification": "unverified", "external_consumer_verified": False,
                           "notes": ["Synthetic protocol fixture, not a research result."]}}


class ExchangeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = validate_figure_spec(spec_fixture())
        self.path = self.root / "spec.json"
        self.path.write_bytes(json_bytes(self.spec))

    def invoke(self, argv):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(argv)
        return status, json.loads(output.getvalue()) if output.getvalue().strip() else None

    def test_descriptor_no_network_or_account_claim(self):
        status, data = self.invoke(["describe"])
        self.assertEqual(status, 0)
        self.assertEqual(data, descriptor())
        self.assertIs(data["network_calls"], False)

    def test_request_source_spec_roundtrip(self):
        path = prepare_request(self.path, self.root / "request")
        request, spec = validate_request(path)
        self.assertEqual(spec, self.spec)
        self.assertEqual(request["scientific_fingerprint"], scientific_fingerprint(self.spec))
        status, data = self.invoke(["validate-request", "--request", str(path)])
        self.assertEqual(status, 0)
        self.assertEqual(data["status"], "VALID_REQUEST_NOT_EXECUTED")

    def test_run_without_authorization_stops_before_generation(self):
        path = prepare_request(self.path, self.root / "request")
        status, _ = self.invoke(["run", "--request", str(path)])
        self.assertEqual(status, 2)
        self.assertFalse((path.parent / "generated").exists())

    def test_authorized_wrapper_delegates_exact_canonical_cli(self):
        path = prepare_request(self.path, self.root / "request", candidates=3)
        called = []
        stub = types.ModuleType("skill.run")
        stub.main = lambda args: (called.append(args), 7)[1]
        with patch.dict(sys.modules, {"skill.run": stub}):
            status, _ = self.invoke(["run", "--request", str(path), "--allow-external"])
        self.assertEqual(status, 7)
        self.assertEqual(called, [request_argv(path)])
        self.assertEqual(load_spec(path.parent / "figure-spec.json"), self.spec)

    def test_existing_generation_is_not_overwritten(self):
        path = prepare_request(self.path, self.root / "request")
        (path.parent / "generated").mkdir()
        status, _ = self.invoke(["run", "--request", str(path), "--allow-external"])
        self.assertEqual(status, 2)

    def test_changed_protocol_is_rejected(self):
        d = descriptor()
        d["schema_version"] = "2.0.0"
        with self.assertRaises(RFSValidationError):
            assert_compatible(d)

    def test_options_do_not_accept_credentials(self):
        with self.assertRaises(RFSValidationError):
            prepare_request(self.path, self.root / "request", api_key="placeholder")
        self.assertFalse((self.root / "request").exists())

    def test_strict_request_json(self):
        path = self.root / "bad.json"
        for value in ('{"x":1,"x":2}', '{"x":NaN}'):
            path.write_text(value)
            with self.assertRaises(RFSValidationError):
                read_object(path)

    def test_actual_exporter_to_boundary_verifier(self):
        from PIL import Image
        stream = io.BytesIO()
        Image.new("RGBA", (16, 16), (10, 20, 30, 128)).save(stream, "PNG")
        raw = stream.getvalue()
        key = "target_diagram_desc0"
        record = {"candidate_id": "0", "stage": "design", "provider": "synthetic-fixture", "model": "not-live",
                  "prompt": "fixture prompt", "design_description": "Two boxes and one connector. Synthetic fixture.",
                  "figure_spec": self.spec, "figure_spec_sha256": sha256_text(canonical_json(self.spec)),
                  "image_base64": base64.b64encode(raw).decode(), "image_mime": "image/png"}
        data = {"candidate_id": "0", "status": "succeeded", "eval_image_field": key + "_base64_jpg", "rfs_artifacts": {key: record}}
        bundle = self.root / "bundle"
        export_bundle(data, bundle)
        manifest, spec, files = verify_result(bundle, self.spec)
        self.assertEqual(spec, self.spec)
        self.assertEqual(files["image"], raw)
        self.assertEqual(manifest["provenance"]["scientific_verification"], "unverified")
        status, result = self.invoke(["verify-bundle", "--spec", str(self.path), "--bundle", str(bundle)])
        self.assertEqual(status, 0)
        self.assertEqual(result["scientific_review"], "PENDING")

    def test_request_source_tamper_is_rejected(self):
        path = prepare_request(self.path, self.root / "request")
        (path.parent / "source.txt").write_text("unrelated source")
        with self.assertRaises(RFSValidationError):
            validate_request(path)


if __name__ == "__main__":
    unittest.main()
