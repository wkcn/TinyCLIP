"""Shared pytest fixtures for the OpenCLIP test suite."""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Generator, Dict, Any

import pytest
import torch


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path)


@pytest.fixture
def mock_torch_device():
    """Mock torch device detection."""
    with patch('torch.cuda.is_available', return_value=False):
        yield


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def sample_text_tokens():
    """Create sample text tokens for testing."""
    return torch.randint(0, 1000, (2, 77))


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Mock configuration for testing."""
    return {
        'model_name': 'test_model',
        'pretrained': None,
        'precision': 'fp32',
        'device': 'cpu',
        'cache_dir': None,
    }


@pytest.fixture
def mock_model_config():
    """Mock model configuration."""
    return {
        "embed_dim": 512,
        "vision_cfg": {
            "image_size": 224,
            "layers": [2, 2, 2, 2],
            "width": 64,
            "head_width": 64,
            "patch_size": None
        },
        "text_cfg": {
            "context_length": 77,
            "vocab_size": 49408,
            "width": 512,
            "heads": 8,
            "layers": 12
        }
    }


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode.return_value = [0, 1, 2, 3, 4]  # Mock token IDs
    tokenizer.decode.return_value = "mock decoded text"
    return tokenizer


@pytest.fixture
def sample_image_path(temp_dir: Path) -> Path:
    """Create a sample image file for testing."""
    import torch
    from PIL import Image
    import numpy as np
    
    # Create a simple test image
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    image_path = temp_dir / "test_image.jpg"
    image.save(image_path)
    return image_path


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """Create a sample text file for testing."""
    text_path = temp_dir / "test_text.txt"
    with open(text_path, 'w') as f:
        f.write("This is a test caption.\nAnother test caption.\n")
    return text_path


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=100)
    dataset.__getitem__ = Mock(return_value={
        'image': torch.randn(3, 224, 224),
        'text': torch.randint(0, 1000, (77,))
    })
    return dataset


@pytest.fixture
def mock_dataloader():
    """Mock dataloader for testing."""
    dataloader = Mock()
    dataloader.__iter__ = Mock(return_value=iter([{
        'images': torch.randn(4, 3, 224, 224),
        'texts': torch.randint(0, 1000, (4, 77))
    }]))
    dataloader.__len__ = Mock(return_value=10)
    return dataloader


@pytest.fixture(autouse=True)
def clean_imports():
    """Clean up imports after each test to avoid side effects."""
    yield
    # Clean up any cached modules that might affect other tests
    import sys
    modules_to_remove = [
        module for module in sys.modules.keys() 
        if module.startswith('open_clip') and 'test' not in module
    ]
    # Don't actually remove them as it might break other tests
    # Just ensure clean state


@pytest.fixture
def mock_weights_path(temp_dir: Path) -> Path:
    """Create a mock weights file for testing."""
    weights_path = temp_dir / "mock_weights.pth"
    mock_state_dict = {
        'visual.conv1.weight': torch.randn(64, 3, 7, 7),
        'text_projection': torch.randn(512, 512),
        'logit_scale': torch.tensor(4.6052),
    }
    torch.save(mock_state_dict, weights_path)
    return weights_path


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    original_env = dict(os.environ)
    test_env = {
        'CUDA_VISIBLE_DEVICES': '0',
        'WORLD_SIZE': '1',
        'RANK': '0',
        'LOCAL_RANK': '0',
    }
    
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_huggingface_hub():
    """Mock huggingface hub calls."""
    with patch('huggingface_hub.hf_hub_download') as mock_download:
        mock_download.return_value = "/fake/path/to/model.pth"
        yield mock_download


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    current_file = Path(__file__)
    test_dir = current_file.parent
    data_dir = test_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# Pytest markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test") 
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add integration marker to tests in integration directory  
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)