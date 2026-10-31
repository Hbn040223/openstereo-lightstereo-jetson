"""Regression test copied into the bootstrapped OpenStereo checkout."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo.modeling.models.lightstereo.backbone import Backbone


class TestBackboneInitialization(TestCase):
    def test_mobilenet_uses_checkpoint_weights_without_network_pretraining(self):
        fake_model = SimpleNamespace(
            conv_stem=nn.Identity(),
            bn1=nn.Identity(),
            act1=nn.Identity(),
            blocks=nn.ModuleList([nn.Identity() for _ in range(6)]),
        )

        with patch(
            "stereo.modeling.models.lightstereo.backbone.timm.create_model",
            return_value=fake_model,
        ) as create_model:
            Backbone("MobileNetv2")

        create_model.assert_called_once_with(
            "mobilenetv2_100", pretrained=False, features_only=True
        )
