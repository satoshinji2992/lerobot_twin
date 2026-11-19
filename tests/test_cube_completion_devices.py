import sys
from types import SimpleNamespace

import numpy as np
import pytest
from twin_depot.cube_completion import Completion


def test_explicit_cuda_never_falls_back_to_cpu(monkeypatch):
    cpu_only = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", cpu_only)
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider unavailable"):
        Completion("cuda")
    model = Completion("auto")
    assert model.device == "CPU · OpenCV"
    assert model.forward(np.zeros((2, 2, 40, 40), np.float32)).shape == (2, 17)
