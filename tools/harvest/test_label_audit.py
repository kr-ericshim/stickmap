"""Safety boundaries and label-audit regressions; uses temporary synthetic inputs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.harvest import label_audit as audit


class LabelAuditTests(unittest.TestCase):
    def test_vault_exclusion_before_truth_or_sealed_metadata_open(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, config, songs = root / "manifest.jsonl", root / "vault.json", root / "songs"
            config.write_text(json.dumps({"cutoff": 100, "video_ids": ["sealed00001"]}))
            rows = [
                {"video": "allowed0001", "slug": "ok-allowed0001", "status": "read", "at": 50},
                {"video": "sealed00001", "slug": "sealed-sealed00001", "status": "read", "at": 50},
                {"video": "future00001", "slug": "future-future00001", "status": "compacted", "at": 101},
                {"video": "excluded001", "slug": "excluded-excluded001", "status": "excluded", "at": 50},
                # An old alias of an explicitly sealed ID must also stay closed.
                {"video": "alias000001", "slug": "alias-alias000001", "status": "read", "at": 50},
                {"video": "allowed0001", "status": "compacted", "at": 150},
            ]
            manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
            for r in rows[:5]:
                p = songs / r["slug"]
                p.mkdir(parents=True)
                (p / "truth").mkdir()
                (p / "truth/truth.json").write_text("DO NOT OPEN")
            (songs / "alias-alias000001/meta.json").write_text(json.dumps({"video_id": "sealed00001"}))
            opened = []
            original = Path.read_text
            def guarded(path, *args, **kwargs):
                opened.append(str(path))
                self.assertNotIn("/truth/", str(path))
                self.assertNotIn("sealed-sealed00001", str(path))
                self.assertNotIn("future-future00001", str(path))
                return original(path, *args, **kwargs)
            with patch.object(audit.vault, "CONFIG", config), patch.object(audit.vault, "MANIFEST", manifest), patch.object(Path, "read_text", guarded):
                selected, skipped = audit.eligible_rows(audit.vault.Vault(), songs)
            self.assertEqual([r["video"] for r in selected], ["allowed0001"])
            self.assertEqual(selected[0]["status"], "compacted")
            self.assertEqual(skipped, {"vault": 3, "other_status": 1})

    def test_detector_cluster_candidates_exclude_unisons_and_out_of_range(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "teacher0.npy"
            act = np.zeros((200, 5), dtype=np.float32)
            act[97:104, 1] = .95
            act[117:124, 1] = .95
            np.save(path, act)
            events = [{"time": t, "inst": c, "system": 1, "shape": "filled", "pos": 3}
                      for t, c in ((1.0, "tom"), (1.2, "tom"), (1.21, "kick"), (8.0, "tom"))]
            candidates, stats = audit.detector_candidates(events, path, audit.Inputs())
            self.assertEqual(len(candidates[1]), 1)
            self.assertEqual(candidates[1][0]["time"], 1.0)
            self.assertEqual(candidates[1][0]["other"], "snare")
            self.assertEqual(stats["in_range"], 1)
            self.assertEqual(stats["non_unison"], 2)

    def test_geometry_one_line_but_not_zoom_or_half_space(self):
        a = {"lines": [70, 80, 90, 100, 110], "spacing": 10}
        self.assertIsNotNone(audit.shift_evidence(a, {"lines": [80, 90, 100, 110, 120], "spacing": 10}))
        self.assertIsNone(audit.shift_evidence(a, {"lines": [75, 85, 95, 105, 115], "spacing": 10}))
        self.assertIsNone(audit.shift_evidence(a, {"lines": [80, 91, 102, 113, 124], "spacing": 11}))

    def test_input_mutation_and_stale_review_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "truth.json"
            path.write_text("[]")
            inputs = audit.Inputs()
            inputs.json(path)
            path.write_text("[1]")
            with self.assertRaises(RuntimeError):
                inputs.verify()
        with self.assertRaises(ValueError):
            audit.attach_reviews([{"video": "v", "system": 1, "crop_sha256": "new"}],
                                 [{"video": "v", "system": 1, "crop_sha256": "old"}])


if __name__ == "__main__":
    unittest.main()
