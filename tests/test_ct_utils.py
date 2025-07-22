import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pydicom
import nibabel as nib

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import ct_utils


class TestCTUtils(unittest.TestCase):
    """Comprehensive tests for ct_utils module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock DICOM dataset
        self.mock_dicom = Mock(spec=pydicom.Dataset)
        self.mock_dicom.ImagePositionPatient = [0, 0, 0]
        self.mock_dicom.SliceLocation = 0
        self.mock_dicom.SliceThickness = 1.0
        self.mock_dicom.PixelSpacing = [1.0, 1.0]
        self.mock_dicom.RescaleIntercept = -1024
        self.mock_dicom.RescaleSlope = 1
        self.mock_dicom.pixel_array = np.random.randint(0, 4096, (512, 512))
        
        # Create test volume
        self.test_volume = np.random.randint(-1000, 3000, (100, 100, 100))
        self.test_labels = np.random.randint(0, 5, (100, 100, 100))
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_get_dicom_spacing(self):
        """Test DICOM spacing extraction."""
        scan = [self.mock_dicom]
        spacing = ct_utils.get_dicom_spacing(scan)
        
        self.assertEqual(len(spacing), 3)
        self.assertEqual(spacing[0], 1.0)  # slice thickness
        self.assertEqual(spacing[1], 1.0)  # pixel spacing x
        self.assertEqual(spacing[2], 1.0)  # pixel spacing y
        self.assertEqual(spacing.dtype, np.float32)
    
    def test_convert_to_hounsfield(self):
        """Test Hounsfield unit conversion."""
        slices = [self.mock_dicom]
        hu_image = ct_utils.convert_to_hounsfield(slices)
        
        self.assertEqual(hu_image.dtype, np.int16)
        self.assertEqual(hu_image.shape, (1, 512, 512))
        
        # Check that conversion was applied
        expected_pixel = self.mock_dicom.pixel_array[0, 0] + self.mock_dicom.RescaleIntercept
        self.assertEqual(hu_image[0, 0, 0], expected_pixel)
    
    def test_resample(self):
        """Test volume resampling."""
        original_shape = (50, 50, 50)
        test_volume = np.random.rand(*original_shape)
        original_spacing = np.array([2.0, 2.0, 2.0])
        new_spacing = [1.0, 1.0, 1.0]
        
        resampled, actual_spacing = ct_utils.resample(test_volume, original_spacing, new_spacing)
        
        # Should approximately double the size due to halving spacing
        self.assertGreater(resampled.shape[0], original_shape[0])
        self.assertGreater(resampled.shape[1], original_shape[1])
        self.assertGreater(resampled.shape[2], original_shape[2])
        self.assertEqual(len(actual_spacing), 3)
    
    def test_filter_hounsfield_bounds(self):
        """Test Hounsfield filtering."""
        test_volume = np.array([-3000, -1000, 0, 1000, 5000])
        filtered = ct_utils.filter_hounsfield_bounds(test_volume, -2000, 4000)
        
        # Values outside bounds should be set to -2000
        expected = np.array([-2000, -1000, 0, 1000, -2000])
        np.testing.assert_array_equal(filtered, expected)
    
    def test_binarize(self):
        """Test image binarization."""
        test_volume = np.array([-3000, -2000, -1000, 0, 1000])
        binary = ct_utils.binarize(test_volume)
        
        # Only values > -2000 should be 1
        expected = np.array([0, 0, 1, 1, 1])
        np.testing.assert_array_equal(binary, expected)
    
    def test_extract_roi(self):
        """Test region of interest extraction."""
        volume = np.random.rand(100, 100, 100)
        min_region = [10, 20, 30]
        max_region = [50, 60, 70]
        
        roi = ct_utils.extract_roi(volume, min_region, max_region)
        
        expected_shape = (40, 40, 40)  # 50-10, 60-20, 70-30
        self.assertEqual(roi.shape, expected_shape)
        
        # Check that extracted region matches original
        np.testing.assert_array_equal(
            roi, volume[10:50, 20:60, 30:70]
        )
    
    def test_split_label_channels(self):
        """Test label channel splitting."""
        # Create labels with values 0-4
        labels = np.array([[[0, 1, 2], [3, 4, 0]]])
        
        split_labels = ct_utils.split_label_channels(labels, num_channels=4)
        
        # Should have 4 channels
        self.assertEqual(split_labels.shape[0], 4)
        self.assertEqual(split_labels.shape[1:], labels.shape)
        
        # Check individual channels
        self.assertEqual(split_labels[0, 0, 0, 1], 1)  # Label 1
        self.assertEqual(split_labels[1, 0, 0, 2], 1)  # Label 2
        self.assertEqual(split_labels[2, 0, 1, 0], 1)  # Label 3
        self.assertEqual(split_labels[3, 0, 1, 1], 1)  # Label 4
        
        # Background should not appear in any channel
        np.testing.assert_array_equal(split_labels[:, 0, 0, 0], [0, 0, 0, 0])
    
    def test_get_sample_label_id(self):
        """Test sample path parsing."""
        # Create test directory structure
        sample_dir = os.path.join(self.temp_dir, 'case-12345')
        os.makedirs(sample_dir)
        
        # Create dummy files
        data_file = os.path.join(sample_dir, 'data.nii')
        label_file = os.path.join(sample_dir, 'label.nii')
        
        with open(data_file, 'w') as f:
            f.write('dummy')
        with open(label_file, 'w') as f:
            f.write('dummy')
        
        result = ct_utils.get_sample_label_id(sample_dir, is_nifti_dataset=True)
        
        self.assertEqual(result['id'], '12345')
        self.assertIn('data', result['data'])
        self.assertIn('label', result['label'])
    
    def test_jaw_isolation_fixed_size(self):
        """Test jaw isolation with fixed size."""
        # Create volume with some high-density regions (bone)
        volume = np.full((100, 100, 100), -1000)  # Air background
        # Add bone region
        volume[40:60, 40:60, 40:60] = 1800  # Bone HU values
        
        min_box, max_box = ct_utils.jaw_isolation(
            volume, hu_threshold=(1600, 2000), size=[30, 30, 30]
        )
        
        self.assertEqual(len(min_box), 3)
        self.assertEqual(len(max_box), 3)
        
        # Check that bounding box is reasonable
        for i in range(3):
            self.assertGreaterEqual(min_box[i], 0)
            self.assertLessEqual(max_box[i], volume.shape[i])
            self.assertLess(min_box[i], max_box[i])
    
    @patch('matplotlib.pyplot.show')
    def test_plot_volume_histogram(self, mock_show):
        """Test volume histogram plotting."""
        volume = np.random.randn(10, 10, 10)
        
        # Should not raise any exceptions
        ct_utils.plot_volume_histogram(volume, bins=20, range=(-2, 2))
        
        # Verify show was called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_3d(self, mock_show):
        """Test 3D plotting."""
        # Create simple volume with some structure
        volume = np.zeros((20, 20, 20))
        volume[5:15, 5:15, 5:15] = 1
        
        # Should not raise any exceptions
        ct_utils.plot_3d(volume, threshold=0.5)
        
        # Verify show was called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_3d_with_labels(self, mock_show):
        """Test 3D plotting with labels."""
        # Create simple volume and labels
        volume = np.zeros((20, 20, 20))
        volume[5:15, 5:15, 5:15] = 1
        
        labels = np.zeros((20, 20, 20))
        labels[8:12, 8:12, 8:12] = 1
        
        # Test with show=True
        ax = ct_utils.plot_3d_with_labels(volume, labels, threshold=0.5, show=True)
        self.assertIsNotNone(ax)
        mock_show.assert_called_once()
        
        # Test with show=False
        mock_show.reset_mock()
        ax = ct_utils.plot_3d_with_labels(volume, labels, threshold=0.5, show=False)
        self.assertIsNotNone(ax)
        mock_show.assert_not_called()
    
    @patch('imageio.mimsave')
    def test_make_gifs(self, mock_mimsave):
        """Test GIF creation."""
        volume = np.random.randint(0, 255, (10, 50, 50), dtype=np.uint8)
        
        ct_utils.make_gifs(volume, 'test_output', ['axial', 'coronal'])
        
        # Should have called mimsave twice (axial and coronal)
        self.assertEqual(mock_mimsave.call_count, 2)
        
        # Check that correct filenames were used
        call_args = [call[0][0] for call in mock_mimsave.call_args_list]
        self.assertIn('test_output_axial.gif', call_args)
        self.assertIn('test_output_coronal.gif', call_args)


class TestDICOMConversion(unittest.TestCase):
    """Test DICOM to NIfTI conversion functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.utils.ct_utils.load_scan')
    @patch('src.utils.ct_utils.convert_to_hounsfield')
    @patch('nibabel.load')
    @patch('nibabel.save')
    def test_convert_dicom_dataset_to_nifti(self, mock_nib_save, mock_nib_load, 
                                           mock_convert_hu, mock_load_scan):
        """Test DICOM to NIfTI conversion."""
        # Mock the functions
        mock_scan = [Mock()]
        mock_scan[0].SliceThickness = 1.0
        mock_scan[0].PixelSpacing = [1.0, 1.0]
        
        mock_load_scan.return_value = mock_scan
        mock_convert_hu.return_value = np.random.randint(-1000, 3000, (50, 50, 50))
        
        # Mock NIfTI loading
        mock_nifti = Mock()
        mock_nifti.dataobj = np.random.randint(0, 5, (50, 50, 50))
        mock_nib_load.return_value = mock_nifti
        
        # Test data
        data_paths = [
            {
                'data': '/fake/data/path',
                'label': '/fake/label/path.nii',
                'id': '12345'
            }
        ]
        
        # Run conversion
        ct_utils.convert_dicom_dataset_to_nifti(
            data_paths, self.temp_dir, separate_label_channels=True
        )
        
        # Verify functions were called
        mock_load_scan.assert_called_once()
        mock_convert_hu.assert_called_once()
        mock_nib_load.assert_called_once()
        
        # Should save two files (data and labels)
        self.assertEqual(mock_nib_save.call_count, 2)


if __name__ == '__main__':
    # Create tests directory if it doesn't exist
    os.makedirs('/home/gaetano/utooth/tests', exist_ok=True)
    
    unittest.main()