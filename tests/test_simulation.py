import pytest
import torch
from unittest.mock import MagicMock, patch

def test_imports():
    """Basic test to verify that the core simulation modules can be imported."""
    try:
        import src.simulation.coordinator
        import src.simulation.farmer
        import src.simulation.adapters
        import src.simulation.data
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")

@patch('src.simulation.adapters.AdapterFactory.create_adapter')
def test_farmer_initialization(mock_create_adapter):
    """Test that a FarmerNode initializes correctly given a dummy dataset."""
    from src.simulation.farmer import FarmerNode
    from torch.utils.data import TensorDataset
    
    # Create simple dummy dataset
    dummy_data = torch.randn(10, 3, 224, 224)
    dummy_labels = torch.randint(0, 3, (10,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    
    mock_base_model = MagicMock()
    mock_adapter_model = MagicMock()
    mock_create_adapter.return_value = mock_adapter_model
    
    # Initialize farmer
    farmer = FarmerNode(
        node_id=0,
        base_model=mock_base_model,
        data_subset=dataset,
        device='cpu',
        total_rounds=10,
        adapter_type="lora"
    )
    
    assert farmer.node_id == 0
    assert farmer.total_rounds == 10
    assert len(farmer.data_subset) == 10
    assert farmer.current_round == 0

def test_coordinator_initialization():
    """Test the DiLoCoCoordinator initializes correctly."""
    from src.simulation.coordinator import DiLoCoCoordinator
    
    mock_base_model = MagicMock()
    
    with patch('src.simulation.adapters.AdapterFactory.create_adapter') as mock_create:
        mock_create.return_value = MagicMock()
        
        coordinator = DiLoCoCoordinator(
            base_model=mock_base_model,
            num_farmers=5,
            local_steps=50,
            adapter_type="lora"
        )
        
        assert coordinator.num_farmers == 5
        assert coordinator.local_steps == 50
        assert len(coordinator.farmer_nodes) == 0  # not initialized yet
        assert coordinator.adapter_type == "lora"
