import pytest

torch = pytest.importorskip("torch")

from nca_segmentation.v2.models import StreetSceneNCA, StreetSceneNCAConfig


def test_v2_forward_shapes():
    model = StreetSceneNCA(StreetSceneNCAConfig(num_classes=4, nca_steps=2))
    output = model(torch.randn(2, 3, 32, 48))
    assert output["semantic_logits"].shape == (2, 4, 32, 48)
    assert output["objectness_logits"].shape == (2, 1, 32, 48)
    assert output["embeddings"].shape == (2, 8, 32, 48)