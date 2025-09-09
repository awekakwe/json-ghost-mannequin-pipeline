"""
Unit tests for Step 1: Preprocessing & JSON Generation.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from photostudio.steps.step1_preprocess import (
    ImagePreprocessor,
    FabricAnalyzer,
    ColorExtractor,
    QualityAssessment,
    ConstructionHeuristics
)


class TestImagePreprocessor:
    """Test the ImagePreprocessor class."""
    
    def test_initialization(self):
        """Test ImagePreprocessor initialization."""
        preprocessor = ImagePreprocessor()
        assert preprocessor is not None
    
    def test_load_image(self, sample_image, temp_dir):
        """Test image loading functionality."""
        # Save sample image to temp file
        image_path = temp_dir / "test_image.jpg"
        sample_image.save(image_path)
        
        preprocessor = ImagePreprocessor()
        loaded_image = preprocessor.load_image(image_path)
        
        assert loaded_image is not None
        assert loaded_image.size == sample_image.size
        assert loaded_image.mode == 'RGB'
    
    def test_validate_image_size(self, sample_image):
        """Test image size validation."""
        preprocessor = ImagePreprocessor()
        
        # Test valid image
        assert preprocessor._validate_image_size(sample_image) == True
        
        # Test too small image
        small_image = Image.new('RGB', (100, 100), color='white')
        assert preprocessor._validate_image_size(small_image) == False
    
    def test_normalize_image(self, sample_image):
        """Test image normalization."""
        preprocessor = ImagePreprocessor()
        normalized = preprocessor._normalize_image(sample_image)
        
        assert normalized.size == (512, 512)  # Should be resized to 512x512
        assert normalized.mode == 'RGB'
    
    @patch('cv2.grabCut')
    def test_segment_garment(self, mock_grabcut, sample_image):
        """Test garment segmentation with GrabCut."""
        preprocessor = ImagePreprocessor()
        
        # Mock GrabCut result
        mock_grabcut.return_value = None
        
        # Create a simple foreground mask for testing
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:400, 100:400] = 255  # Simple square mask
        
        with patch.object(preprocessor, '_apply_grabcut', return_value=mask):
            result_mask = preprocessor.segment_garment(sample_image)
        
        assert result_mask is not None
        assert result_mask.size == sample_image.size
        assert result_mask.mode == 'L'


class TestColorExtractor:
    """Test the ColorExtractor class."""
    
    def test_initialization(self):
        """Test ColorExtractor initialization."""
        extractor = ColorExtractor()
        assert extractor is not None
    
    def test_extract_dominant_color(self, sample_image, sample_mask):
        """Test dominant color extraction."""
        extractor = ColorExtractor()
        
        # Extract color from masked region
        color_info = extractor.extract_dominant_color(sample_image, sample_mask)
        
        assert 'rgb' in color_info
        assert 'hex' in color_info
        assert 'name' in color_info
        assert 'lab' in color_info
        
        # Check RGB values are in valid range
        rgb = color_info['rgb']
        assert all(0 <= c <= 255 for c in rgb)
        
        # Check hex format
        assert color_info['hex'].startswith('#')
        assert len(color_info['hex']) == 7
    
    def test_rgb_to_lab_conversion(self):
        """Test RGB to LAB color space conversion."""
        extractor = ColorExtractor()
        
        # Test with known values
        rgb = [128, 128, 128]  # Middle gray
        lab = extractor._rgb_to_lab(rgb)
        
        assert len(lab) == 3
        assert isinstance(lab[0], float)  # L*
        assert isinstance(lab[1], float)  # a*
        assert isinstance(lab[2], float)  # b*
        
        # L* should be around 50 for middle gray
        assert 45 <= lab[0] <= 55
    
    def test_get_color_name(self):
        """Test color name detection."""
        extractor = ColorExtractor()
        
        # Test known colors
        red_name = extractor._get_color_name([255, 0, 0])
        assert 'red' in red_name.lower()
        
        blue_name = extractor._get_color_name([0, 0, 255])
        assert 'blue' in blue_name.lower()
        
        white_name = extractor._get_color_name([255, 255, 255])
        assert 'white' in white_name.lower()


class TestFabricAnalyzer:
    """Test the FabricAnalyzer class."""
    
    def test_initialization(self):
        """Test FabricAnalyzer initialization."""
        analyzer = FabricAnalyzer()
        assert analyzer is not None
    
    def test_analyze_texture(self, sample_image, sample_mask):
        """Test fabric texture analysis."""
        analyzer = FabricAnalyzer()
        
        analysis = analyzer.analyze_texture(sample_image, sample_mask)
        
        assert 'texture_complexity' in analysis
        assert 'pattern_type' in analysis
        assert 'fabric_type' in analysis
        assert 'sheen_level' in analysis
        
        # Check value ranges
        assert 0 <= analysis['texture_complexity'] <= 1
        assert 0 <= analysis['sheen_level'] <= 1
        assert analysis['pattern_type'] in ['solid', 'striped', 'checkered', 'floral', 'geometric', 'other']
    
    def test_calculate_lbp_features(self):
        """Test Local Binary Pattern feature calculation."""
        analyzer = FabricAnalyzer()
        
        # Create test image with known pattern
        test_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_array)
        
        features = analyzer._calculate_lbp_features(test_image)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert all(f >= 0 for f in features)  # All features should be non-negative
    
    def test_detect_sheen(self, sample_image, sample_mask):
        """Test fabric sheen detection."""
        analyzer = FabricAnalyzer()
        
        sheen_level = analyzer._detect_sheen(sample_image, sample_mask)
        
        assert 0 <= sheen_level <= 1
        assert isinstance(sheen_level, float)


class TestQualityAssessment:
    """Test the QualityAssessment class."""
    
    def test_initialization(self):
        """Test QualityAssessment initialization."""
        assessment = QualityAssessment()
        assert assessment is not None
    
    def test_assess_image_quality(self, sample_image, sample_mask):
        """Test overall image quality assessment."""
        assessment = QualityAssessment()
        
        quality_metrics = assessment.assess_image_quality(sample_image, sample_mask)
        
        assert 'image_sharpness' in quality_metrics
        assert 'color_consistency' in quality_metrics
        assert 'segmentation_confidence' in quality_metrics
        assert 'overall_quality' in quality_metrics
        
        # Check value ranges
        for metric in quality_metrics.values():
            assert 0 <= metric <= 1
    
    def test_calculate_sharpness(self, sample_image):
        """Test image sharpness calculation."""
        assessment = QualityAssessment()
        
        sharpness = assessment._calculate_sharpness(sample_image)
        
        assert isinstance(sharpness, float)
        assert sharpness >= 0  # Sharpness should be non-negative
    
    def test_assess_color_consistency(self, sample_image, sample_mask):
        """Test color consistency assessment."""
        assessment = QualityAssessment()
        
        consistency = assessment._assess_color_consistency(sample_image, sample_mask)
        
        assert 0 <= consistency <= 1
        assert isinstance(consistency, float)
    
    def test_calculate_mask_confidence(self, sample_mask):
        """Test segmentation mask confidence calculation."""
        assessment = QualityAssessment()
        
        confidence = assessment._calculate_mask_confidence(sample_mask)
        
        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)


class TestConstructionHeuristics:
    """Test the ConstructionHeuristics class."""
    
    def test_initialization(self):
        """Test ConstructionHeuristics initialization."""
        heuristics = ConstructionHeuristics()
        assert heuristics is not None
    
    def test_analyze_construction(self, sample_image, sample_mask):
        """Test garment construction analysis."""
        heuristics = ConstructionHeuristics()
        
        construction = heuristics.analyze_construction(sample_image, sample_mask)
        
        assert 'garment_category' in construction
        assert 'style' in construction
        assert 'sleeve_type' in construction
        assert 'neckline' in construction
        assert 'fit' in construction
        
        # Check valid categories
        assert construction['garment_category'] in ['tops', 'bottoms', 'dresses', 'outerwear', 'accessories']
    
    def test_detect_garment_type(self, sample_mask):
        """Test garment type detection from mask shape."""
        heuristics = ConstructionHeuristics()
        
        garment_type = heuristics._detect_garment_type(sample_mask)
        
        assert isinstance(garment_type, str)
        assert len(garment_type) > 0
    
    def test_analyze_proportions(self, sample_mask):
        """Test garment proportion analysis."""
        heuristics = ConstructionHeuristics()
        
        proportions = heuristics._analyze_proportions(sample_mask)
        
        assert 'width_height_ratio' in proportions
        assert 'area_ratio' in proportions
        
        # Check ratios are positive
        assert proportions['width_height_ratio'] > 0
        assert proportions['area_ratio'] > 0
    
    def test_detect_sleeves(self, sample_mask):
        """Test sleeve detection."""
        heuristics = ConstructionHeuristics()
        
        sleeve_info = heuristics._detect_sleeves(sample_mask)
        
        assert 'has_sleeves' in sleeve_info
        assert 'sleeve_type' in sleeve_info
        assert isinstance(sleeve_info['has_sleeves'], bool)


class TestIntegrationScenarios:
    """Integration tests for complete preprocessing workflow."""
    
    def test_complete_preprocessing_workflow(self, sample_image, temp_dir):
        """Test the complete preprocessing workflow."""
        # Save sample image
        input_path = temp_dir / "input.jpg"
        sample_image.save(input_path)
        
        # Create preprocessor
        preprocessor = ImagePreprocessor()
        
        # Load and preprocess image
        image = preprocessor.load_image(input_path)
        assert image is not None
        
        # Segment garment (mock the GrabCut operation)
        with patch.object(preprocessor, 'segment_garment') as mock_segment:
            mock_mask = Image.new('L', (512, 512), color=255)
            mock_segment.return_value = mock_mask
            
            mask = preprocessor.segment_garment(image)
            assert mask is not None
        
        # Extract color information
        color_extractor = ColorExtractor()
        color_info = color_extractor.extract_dominant_color(image, mask)
        assert 'rgb' in color_info
        
        # Analyze fabric
        fabric_analyzer = FabricAnalyzer()
        fabric_analysis = fabric_analyzer.analyze_texture(image, mask)
        assert 'texture_complexity' in fabric_analysis
        
        # Assess quality
        quality_assessment = QualityAssessment()
        quality_metrics = quality_assessment.assess_image_quality(image, mask)
        assert 'overall_quality' in quality_metrics
        
        # Analyze construction
        construction_heuristics = ConstructionHeuristics()
        construction_analysis = construction_heuristics.analyze_construction(image, mask)
        assert 'garment_category' in construction_analysis
    
    def test_error_handling_invalid_image(self, temp_dir):
        """Test error handling with invalid image."""
        preprocessor = ImagePreprocessor()
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            preprocessor.load_image(temp_dir / "nonexistent.jpg")
        
        # Test with invalid image file
        invalid_file = temp_dir / "invalid.jpg"
        invalid_file.write_text("not an image")
        
        with pytest.raises(Exception):
            preprocessor.load_image(invalid_file)
    
    def test_edge_cases_small_mask(self):
        """Test handling of edge cases like very small masks."""
        color_extractor = ColorExtractor()
        
        # Create very small mask (only few pixels)
        small_image = Image.new('RGB', (100, 100), color='red')
        small_mask = Image.new('L', (100, 100), color=0)
        
        # Add only a few white pixels
        pixels = np.array(small_mask)
        pixels[50:55, 50:55] = 255
        small_mask = Image.fromarray(pixels, mode='L')
        
        color_info = color_extractor.extract_dominant_color(small_image, small_mask)
        
        # Should still work, even with small mask
        assert 'rgb' in color_info
        assert color_info['rgb'] is not None
