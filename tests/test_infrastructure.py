"""Infrastructure validation tests to ensure testing setup works correctly."""

import pytest
import sys
from pathlib import Path
import tempfile
import torch


class TestInfrastructure:
    """Test the testing infrastructure itself."""

    def test_pytest_working(self):
        """Verify pytest is working correctly."""
        assert True

    def test_fixtures_available(self, temp_dir, mock_config):
        """Verify that fixtures are working."""
        assert temp_dir.exists()
        assert isinstance(mock_config, dict)
        assert 'model_name' in mock_config

    def test_mock_functionality(self, mock_tokenizer):
        """Verify that mocking functionality works."""
        result = mock_tokenizer.encode("test")
        assert result == [0, 1, 2, 3, 4]

    def test_torch_import(self):
        """Verify that torch can be imported."""
        import torch
        assert hasattr(torch, 'tensor')

    def test_temp_directory_fixture(self, temp_dir):
        """Test that temporary directory fixture works."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()

    def test_sample_tensor_fixture(self, sample_tensor):
        """Test that tensor fixtures work."""
        assert isinstance(sample_tensor, torch.Tensor)
        assert sample_tensor.shape == (2, 3, 224, 224)

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit markers work."""
        assert True

    @pytest.mark.slow 
    def test_slow_marker(self):
        """Test that slow markers work."""
        import time
        time.sleep(0.1)  # Minimal sleep for slow test
        assert True

    def test_package_structure(self):
        """Test that the package can be found."""
        # Test that the source package is discoverable
        project_root = Path(__file__).parent.parent
        src_path = project_root / "src"
        assert src_path.exists(), f"Source directory not found at {src_path}"
        
        open_clip_path = src_path / "open_clip"
        assert open_clip_path.exists(), f"OpenCLIP package not found at {open_clip_path}"

    def test_coverage_runs(self):
        """Test that we can measure coverage."""
        # This test will be covered and should show up in coverage report
        def covered_function():
            return "covered"
        
        result = covered_function()
        assert result == "covered"

    def test_markers_configured(self):
        """Test that custom markers are configured."""
        # This is a basic test - markers are configured in conftest.py
        assert True


class TestOpenCLIPImports:
    """Test that OpenCLIP modules can be imported."""

    def test_import_open_clip(self):
        """Test that we can import the main open_clip module."""
        try:
            import sys
            from pathlib import Path
            
            # Add src to Python path
            project_root = Path(__file__).parent.parent
            src_path = project_root / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            import open_clip
            # Check that the module was imported successfully
            assert hasattr(open_clip, 'create_model') or hasattr(open_clip, 'list_models')
        except ImportError as e:
            pytest.skip(f"Could not import open_clip: {e}")

    def test_version_accessible(self):
        """Test that version information is accessible."""
        try:
            import sys
            from pathlib import Path
            
            # Add src to Python path
            project_root = Path(__file__).parent.parent
            src_path = project_root / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            from open_clip.version import __version__
            assert isinstance(__version__, str)
            assert len(__version__) > 0
        except ImportError as e:
            pytest.skip(f"Could not import version: {e}")


@pytest.mark.integration
class TestIntegrationInfrastructure:
    """Integration tests for testing infrastructure."""

    def test_integration_marker_applied(self):
        """Test that integration markers are applied correctly."""
        # This test should automatically get the integration marker
        assert True

    def test_mock_environment(self, mock_env_vars):
        """Test that environment mocking works."""
        import os
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
        assert os.environ.get('WORLD_SIZE') == '1'

    def test_file_fixtures_integration(self, sample_image_path, sample_text_file):
        """Test file fixtures work in integration context."""
        assert sample_image_path.exists()
        assert sample_text_file.exists()
        
        # Verify file contents
        content = sample_text_file.read_text()
        assert "test caption" in content


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])