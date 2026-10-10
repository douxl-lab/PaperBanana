import base64
import copy
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from rfs import RFSValidationError, export_bundle, verify_bundle
from rfs.contract import canonical_json, sha256_text


@pytest.fixture
def spec():
    return json.loads((Path(__file__).parents[1] / "rfs/examples/encoder-pipeline.figure-spec.json").read_text())


def image_bytes(fmt="PNG"):
    image = Image.new("RGBA" if fmt == "PNG" else "RGB", (32, 24), (15, 30, 70, 64) if fmt == "PNG" else (15, 30, 70))
    stream = io.BytesIO()
    image.save(stream, format=fmt)
    return stream.getvalue()


def result_for(spec, fmt="PNG"):
    raw = image_bytes(fmt)
    key = "target_diagram_stylist_desc0"
    record = {"candidate_id": "candidate-0", "stage": "candidate", "provider": "mock", "model": "test-image",
              "prompt": "EXACT prompt with no added newline", "system_prompt": "EXACT separate system prompt",
              "design_description": "Detailed left-to-right composition with independent editable nodes.",
              "figure_spec_sha256": sha256_text(canonical_json(spec)), "figure_spec": copy.deepcopy(spec),
              "image_base64": base64.b64encode(raw).decode(), "image_mime": "image/png" if fmt == "PNG" else "image/jpeg",
              "options": {"quality": "high", "api_key": "do-not-export", "nested": {"authorization": "secret", "temperature": 1}},
              "provenance": {"mock": True}}
    return {"candidate_id": "candidate-0", "status": "succeeded", "eval_image_field": key + "_base64_jpg",
            "figure_spec": spec, "rfs_input_figure_spec": copy.deepcopy(spec), "rfs_artifacts": {key: record}}, raw


def test_lossless_four_part_export_and_manifest_hashes(tmp_path, spec):
    data, raw = result_for(spec)
    target = tmp_path / "exchange"
    manifest = export_bundle(data, target)
    assert (target / "image.png").read_bytes() == raw
    assert (target / "prompt.txt").read_text() == "EXACT prompt with no added newline"
    assert (target / "prompt.system.txt").read_text() == "EXACT separate system prompt"
    assert json.loads((target / "figure_spec.input.json").read_text()) == spec
    assert manifest == verify_bundle(target)
    assert {"image", "prompt", "design", "figure_spec"} <= set(manifest["files"])
    assert manifest["provenance"]["external_consumer_verified"] is False
    assert "do-not-export" not in (target / "manifest.json").read_text()
    assert "PPTX reconstruction has not been produced" in (target / "design.md").read_text()
    with Image.open(target / "image.png") as image:
        assert image.mode == "RGBA" and image.getpixel((0, 0))[3] == 64
    with pytest.raises(FileExistsError):
        export_bundle(data, target)
    assert (target / "image.png").read_bytes() == raw


def test_jpeg_original_bytes_are_preserved_separately(tmp_path, spec):
    data, raw = result_for(spec, "JPEG")
    target = tmp_path / "jpeg"
    manifest = export_bundle(data, target)
    assert (target / "image.original.jpg").read_bytes() == raw
    assert manifest["files"]["original_image"]["mime"] == "image/jpeg"
    with Image.open(target / "image.png") as image:
        assert image.format == "PNG"


def test_selected_stage_prompt_matches_selected_image(tmp_path, spec):
    data, _raw = result_for(spec)
    data["rfs_artifacts"]["target_diagram_desc0"] = {"prompt": "wrong stage"}
    target = tmp_path / "selected"
    manifest = export_bundle(data, target)
    assert manifest["stage"] == "candidate"
    assert "wrong stage" not in (target / "prompt.txt").read_text()


def test_failed_empty_candidate_gets_honest_failure_manifest(tmp_path):
    target = tmp_path / "failed"
    manifest = export_bundle({"candidate_id": "failed", "status": "failed", "error": {"stage": "planner", "message": "invalid JSON"}}, target)
    assert manifest["status"] == "failed"
    assert manifest["files"] == {}
    assert manifest["provider"] is None
    assert manifest["provenance"]["image_bytes_preserved"] is False
    assert verify_bundle(target) == manifest


def test_failed_review_retains_valid_image_without_success(tmp_path, spec):
    data, raw = result_for(spec)
    data["status"] = "failed"
    data["review_status"] = "failed"
    manifest = export_bundle(data, tmp_path / "review-failed")
    assert manifest["status"] == "failed"
    assert (tmp_path / "review-failed/image.png").read_bytes() == raw


def test_success_cannot_fabricate_prompt_or_image(tmp_path, spec):
    with pytest.raises(RFSValidationError):
        export_bundle({"candidate_id": "missing", "status": "succeeded", "figure_spec": spec}, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(image_base64="Error"),
    lambda r: r.update(image_mime="image/jpeg"),
    lambda r: r.update(candidate_id="different"),
    lambda r: r.update(figure_spec_sha256="0" * 64),
    lambda r: r.update(prompt=""),
])
def test_invalid_artifact_prevents_export(tmp_path, spec, mutation):
    data, _raw = result_for(spec)
    mutation(next(iter(data["rfs_artifacts"].values())))
    with pytest.raises(RFSValidationError):
        export_bundle(data, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_verify_catches_tampering_and_unsafe_paths(tmp_path, spec):
    data, _raw = result_for(spec)
    target = tmp_path / "tamper"
    export_bundle(data, target)
    (target / "prompt.txt").write_text("modified")
    with pytest.raises(RFSValidationError, match="checksum"):
        verify_bundle(target)
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["files"]["prompt"]["path"] = "../outside"
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RFSValidationError, match="relative"):
        verify_bundle(target)


def test_export_symlink_rejected(tmp_path, spec):
    data, _raw = result_for(spec)
    original = tmp_path / "original"
    original.mkdir()
    link = tmp_path / "link"
    link.symlink_to(original, target_is_directory=True)
    with pytest.raises(RFSValidationError, match="symlink"):
        export_bundle(data, link / "result")
    assert not (original / "result").exists()


@pytest.mark.parametrize("status", ["queued", "running", "completed", "unknown", None])
def test_export_rejects_nonterminal_or_unknown_candidate_status(tmp_path, spec, status):
    data, _raw = result_for(spec)
    data["status"] = status
    with pytest.raises(RFSValidationError, match="terminal"):
        export_bundle(data, tmp_path / "not-terminal")
    assert not (tmp_path / "not-terminal").exists()


def test_reproducible_plot_code_is_exported(tmp_path, spec):
    data, _raw = result_for(spec)
    key = "target_diagram_stylist_desc0"
    data["rfs_artifacts"][key]["code"] = "# exact reproducible code\n"
    manifest = export_bundle(data, tmp_path / "with-code")
    assert "plot_code" in manifest["files"]
    assert (tmp_path / "with-code/plot_code.py").read_text() == "# exact reproducible code\n"


def test_export_retains_request_provenance(tmp_path, spec):
    data, _raw = result_for(spec)
    selected = data["rfs_artifacts"]["target_diagram_stylist_desc0"]
    selected.update(request_id="controlled-request", usage={"image_count": 1}, attempts=2)
    manifest = export_bundle(data, tmp_path / "provenance")
    assert manifest["provenance"]["generation"]["request_id"] == "controlled-request"
    assert manifest["provenance"]["generation"]["attempts"] == 2
    assert manifest == verify_bundle(tmp_path / "provenance")


def test_failure_manifest_redacts_known_credential_patterns(tmp_path):
    key = "AIza" + "x" * 35
    manifest = export_bundle({"candidate_id": "failed", "status": "failed",
                              "error": {"message": f"failed with {key} Bearer abcdefghijklmnop secret=some-value"}}, tmp_path / "redacted")
    content = (tmp_path / "redacted/manifest.json").read_text()
    assert key not in content and "abcdefghijklmnop" not in content and "some-value" not in content
    assert "[redacted]" in content
    assert manifest == verify_bundle(tmp_path / "redacted")
