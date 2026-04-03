"""
Test suite for search_engine_fastapi.py
Tests designed to catch the 6 intentional bugs

10 total tests:
- 4 passing (basic functionality that works)
- 6 failing (each catches one of the bugs)
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the buggy module
import search_engine_fastapi as search_module


# ============== FIXTURES ==============

@pytest.fixture
def mock_csv_data():
    """Create mock CSV data for testing"""
    return pd.DataFrame({
        'sku': ['SKU001', 'SKU002', 'SKU003'],
        'product_title': ['Laptop Computer', 'Wireless Mouse', 'USB Cable'],
        'product_category': ['Electronics', 'Accessories', 'Cables'],
        'total_reviews': [150, 80, 45]
    })


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing"""
    return np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ], dtype=np.float32)


# ============== PASSING TESTS (4) ==============

def test_health_endpoint_basic():
    """
    PASSING TEST: Health endpoint returns proper structure
    This works because the endpoint structure is correct
    """
    # We can't test the actual endpoint without running the server,
    # but we can test the health function logic
    search_module.is_ready = True
    search_module.product_skus = ['SKU001', 'SKU002']
    
    result = search_module.health()
    
    assert "ready" in result
    assert "embedding_backend" in result
    assert "total_products" in result


def test_normalize_vector_correct():
    """
    PASSING TEST: normalize_vector function works correctly
    This function doesn't have bugs
    """
    vec = np.array([3.0, 4.0])
    normalized = search_module.normalize_vector(vec)
    
    # Should be unit vector
    norm = np.linalg.norm(normalized)
    assert abs(norm - 1.0) < 1e-6, "Normalized vector should have unit length"


def test_csv_loader_finds_required_columns():
    """
    PASSING TEST: CSV loader correctly identifies required columns
    Column checking logic works fine
    """
    with patch('pandas.read_csv') as mock_read:
        mock_df = pd.DataFrame({
            'sku': ['A'],
            'product_title': ['Test'],
            'total_reviews': [10]
        })
        mock_read.return_value = mock_df
        
        # Should not raise error for required columns
        result = search_module.load_products_from_csv('dummy.csv')
        assert 'sku' in result.columns
        assert 'product_title' in result.columns


def test_product_result_model_validation():
    """
    PASSING TEST: ProductResult model accepts valid data
    Pydantic model validation works correctly
    """
    from search_engine_fastapi import ProductResult
    
    product = ProductResult(
        title="Test Product",
        sku="SKU123",
        total_reviews=50
    )
    
    assert product.title == "Test Product"
    assert product.sku == "SKU123"
    assert product.total_reviews == 50


# ============== FAILING TESTS (6) - Each catches a bug ==============

def test_bug_1_normalize_matrix_wrong_axis():
    """
    BUG 1: normalize_matrix uses wrong axis (axis=0 instead of axis=1)
    
    Expected: Each row should be normalized independently
    Actual: Normalizes across rows (wrong!)
    """
    mat = np.array([
        [3.0, 4.0],
        [5.0, 12.0]
    ])
    
    normalized = search_module.normalize_matrix(mat)
    
    # Each row should have unit length
    for i in range(len(normalized)):
        row_norm = np.linalg.norm(normalized[i])
        assert abs(row_norm - 1.0) < 1e-6, \
            f"Bug 1: Row {i} has norm {row_norm}, expected 1.0 (wrong axis used)"


def test_bug_2_safe_combine_missing_sku():
    """
    BUG 2: safe_combine uses 'product_title' twice instead of including 'sku'
    
    Expected: Combined text should include SKU, title, and category
    Actual: Uses product_title twice, SKU is missing!
    """
    with patch('pandas.read_csv') as mock_read:
        mock_df = pd.DataFrame({
            'sku': ['LAPTOP123'],
            'product_title': ['Gaming Laptop'],
            'product_category': ['Electronics'],
            'total_reviews': [100]
        })
        mock_read.return_value = mock_df
        
        result = search_module.load_products_from_csv('dummy.csv')
        combined_text = result.iloc[0]['combined']
        
        # Combined text should contain the SKU
        assert 'laptop123' in combined_text.lower(), \
            "Bug 2: SKU should be in combined text but it's missing (used product_title twice)"


def test_bug_3_ready_flag_set_too_early():
    """
    BUG 3: is_ready flag is set to True BEFORE data is fully loaded
    
    Expected: is_ready should be False during loading, True only after complete
    Actual: Set to True at the beginning of lifespan!
    
    Note: This is a timing/order bug that's hard to test in unit tests
    In real usage, this causes race conditions where the API accepts requests
    before data is ready.
    """
    # Reset global state
    search_module.is_ready = False
    search_module.product_skus = None
    
    # Simulate the bug: is_ready is set before data loads
    # In the actual code, this happens at line "is_ready = True"
    # which is BEFORE all the data loading completes
    
    # The bug is that this line appears too early in the lifespan function
    # It should be at the END, not the beginning
    
    # We'll test by checking if the pattern exists
    import inspect
    source = inspect.getsource(search_module.lifespan)
    
    # Find where is_ready = True appears
    lines = source.split('\n')
    ready_line_idx = None
    products_line_idx = None
    
    for i, line in enumerate(lines):
        if 'is_ready = True' in line and 'global' not in line:
            ready_line_idx = i
        if 'load_products_from_csv' in line:
            products_line_idx = i
    
    assert ready_line_idx is not None and products_line_idx is not None, \
        "Bug 3: Could not find is_ready or load_products lines"
    
    assert ready_line_idx > products_line_idx, \
        f"Bug 3: is_ready set at line {ready_line_idx} but products loaded at {products_line_idx}. " \
        "Ready flag should be set AFTER loading, not before!"


def test_bug_4_empty_query_not_handled():
    """
    BUG 4: Empty query check is missing - will crash on empty string
    
    Expected: Empty query should return empty results gracefully
    Actual: Tries to process empty string, causing errors!
    """
    # Set up minimal mock state
    search_module.is_ready = True
    search_module.embeddings_unit = np.array([[0.1, 0.2, 0.3]])
    search_module.product_skus = ['SKU001']
    search_module.review_counts = {'SKU001': 10}
    
    # Mock the embed_query to track if it gets called with empty string
    with patch.object(search_module, 'embed_query') as mock_embed:
        mock_embed.return_value = np.array([0.1, 0.2, 0.3])
        
        # Try to search with empty query
        try:
            # The bug is that empty query check is missing
            # So the function will try to embed an empty string
            # In the original code, there was: if not query.strip(): return ...
            # In the buggy code, this check is removed!
            
            # Check if the source code has the empty query check
            import inspect
            source = inspect.getsource(search_module.search)
            
            assert 'if not query.strip()' in source or 'if query.strip() == ""' in source, \
                "Bug 4: Empty query check is missing! Should check 'if not query.strip()' and return empty results"
            
        except AssertionError:
            raise  # Re-raise assertion errors
        except Exception as e:
            # If any other exception, it means empty query wasn't handled
            pytest.fail(f"Bug 4: Empty query caused error: {e}")


def test_bug_5_missing_last_result():
    """
    BUG 5: Using TOP_K-1 instead of TOP_K (misses the last result)
    
    Expected: Should return TOP_K results
    Actual: Returns TOP_K-1 results!
    """
    # Set up test data
    search_module.is_ready = True
    search_module.TOP_K = 5  # Set to small number for testing
    
    # Create mock data with more items than TOP_K
    search_module.product_skus = [f'SKU{i:03d}' for i in range(10)]
    search_module.embeddings_unit = np.random.rand(10, 128).astype(np.float32)
    search_module.review_counts = {sku: i*10 for i, sku in enumerate(search_module.product_skus)}
    search_module.products_df = pd.DataFrame({
        'product_title': [f'Product {i}' for i in range(10)],
    }, index=search_module.product_skus)
    
    with patch.object(search_module, 'embed_query') as mock_embed:
        mock_embed.return_value = np.random.rand(128).astype(np.float32)
        
        result = search_module.search("test query")
        
        # Should return TOP_K results, but bug makes it return TOP_K-1
        assert len(result.products) == search_module.TOP_K, \
            f"Bug 5: Expected {search_module.TOP_K} results but got {len(result.products)}. " \
            f"Code uses TOP_K-1 instead of TOP_K!"


def test_bug_6_sku_field_wrong_mapping():
    """
    BUG 6: ProductResult 'sku' field uses 'product_title' instead of actual SKU
    
    Expected: sku field should contain the SKU value
    Actual: sku field contains the product_title value (wrong mapping!)
    """
    # Set up test data
    search_module.is_ready = True
    search_module.product_skus = ['LAPTOP001']
    search_module.embeddings_unit = np.array([[0.1, 0.2, 0.3]])
    search_module.review_counts = {'LAPTOP001': 50}
    search_module.products_df = pd.DataFrame({
        'product_title': ['Gaming Laptop 15 inch'],
    }, index=['LAPTOP001'])
    
    with patch.object(search_module, 'embed_query') as mock_embed:
        mock_embed.return_value = np.array([0.1, 0.2, 0.3])
        
        result = search_module.search("laptop")
        
        assert len(result.products) > 0, "Should return at least one result"
        
        first_product = result.products[0]
        
        # Check that SKU field actually contains a SKU, not a title
        # SKUs should be short alphanumeric codes, not long product descriptions
        assert first_product.sku == 'LAPTOP001', \
            f"Bug 6: SKU field contains '{first_product.sku}' but should contain 'LAPTOP001'. " \
            f"The code maps 'product_title' to sku field instead of using the actual SKU!"


# ============== TEST SUMMARY ==============
"""
BUG SUMMARY:
1. normalize_matrix: Uses axis=0 instead of axis=1
2. safe_combine: Uses product_title twice instead of including sku
3. is_ready flag: Set too early (before data loaded)
4. Empty query: Missing check for empty/whitespace queries
5. TOP_K: Uses TOP_K-1 instead of TOP_K (misses last result)
6. SKU mapping: ProductResult sku field gets product_title value

PASSING TESTS (4):
- test_health_endpoint_basic
- test_normalize_vector_correct
- test_csv_loader_finds_required_columns
- test_product_result_model_validation

FAILING TESTS (6):
- test_bug_1_normalize_matrix_wrong_axis
- test_bug_2_safe_combine_missing_sku
- test_bug_3_ready_flag_set_too_early
- test_bug_4_empty_query_not_handled
- test_bug_5_missing_last_result
- test_bug_6_sku_field_wrong_mapping
"""
