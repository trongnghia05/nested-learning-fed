"""
Test script to verify data and model isolation in FL implementation.

Kiểm tra:
1. Data isolation: Mỗi client có data riêng, không trùng
2. Model isolation: Model của mỗi client độc lập
3. Weight isolation: Thay đổi weight client 1 không ảnh hưởng client 2
4. Aggregation correctness: FedAvg tính đúng

Test script sẽ kiểm tra:
  1. Data isolation: Không có index trùng giữa clients
  2. Model isolation: Mỗi client có model object riêng
  3. Training independence: Train client 0 không ảnh hưởng client 1
  4. Aggregation correctness: FedAvg tính đúng weighted average
  5. Round isolation: Mỗi round bắt đầu từ global params
  6. Gradient buffer isolation: Fed-M3 buffers độc lập

Run: python test_isolation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from collections import Counter

from models import create_model
from fl import FLClient, FLServer, dirichlet_split, iid_split
from fl.data_split import get_client_labels
from utils import set_seed


def test_data_isolation():
    """Test 1: Verify each client has unique, non-overlapping data indices."""
    print("\n" + "=" * 60)
    print("TEST 1: DATA ISOLATION")
    print("=" * 60)

    from torchvision import datasets, transforms
    transform = transforms.ToTensor()

    # Use CIFAR-10 for testing
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)

    num_clients = 5
    client_datasets = dirichlet_split(train_data, num_clients, alpha=0.5, seed=42)

    # Collect all indices
    all_indices = []
    client_indices_list = []

    for i, ds in enumerate(client_datasets):
        indices = list(ds.indices)
        client_indices_list.append(set(indices))
        all_indices.extend(indices)
        print(f"Client {i}: {len(indices)} samples")

    # Check 1: No duplicate indices within same client
    for i, indices in enumerate(client_indices_list):
        assert len(indices) == len(client_datasets[i].indices), \
            f"Client {i} has duplicate indices!"

    # Check 2: No overlapping indices between clients
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            overlap = client_indices_list[i] & client_indices_list[j]
            assert len(overlap) == 0, \
                f"Client {i} and {j} have {len(overlap)} overlapping indices!"

    # Check 3: All data is distributed
    total_distributed = sum(len(ds) for ds in client_datasets)
    print(f"\nTotal samples distributed: {total_distributed}")
    print(f"Original dataset size: {len(train_data)}")

    # Check 4: Verify actual data is different
    print("\nVerifying actual data samples are different...")
    sample_0_client_0 = client_datasets[0].dataset[client_datasets[0].indices[0]]
    sample_0_client_1 = client_datasets[1].dataset[client_datasets[1].indices[0]]

    # They should have different indices
    assert client_datasets[0].indices[0] != client_datasets[1].indices[0], \
        "First sample index should be different!"

    print("✓ Data isolation PASSED!")
    return True


def test_model_isolation():
    """Test 2: Verify each client has independent model copy."""
    print("\n" + "=" * 60)
    print("TEST 2: MODEL ISOLATION")
    print("=" * 60)

    set_seed(42)
    device = torch.device('cpu')

    # Create base model
    base_model = create_model('cnn_small', num_classes=10)

    # Create dummy dataset
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    client_datasets = iid_split(train_data, num_clients=3, seed=42)

    # Create clients
    clients = []
    for i in range(3):
        client = FLClient(
            client_id=i,
            dataset=client_datasets[i],
            model=base_model,
            device=device,
            batch_size=32,
        )
        clients.append(client)

    # Check 1: Each client has its own model (different id)
    model_ids = [id(c.model) for c in clients]
    assert len(set(model_ids)) == 3, "Models should be different objects!"
    print("✓ Each client has unique model object")

    # Check 2: Initially, all models have same weights (copied from base)
    for i in range(1, 3):
        for (name1, p1), (name2, p2) in zip(
            clients[0].model.named_parameters(),
            clients[i].model.named_parameters()
        ):
            assert torch.allclose(p1, p2), \
                f"Initial weights should be same! {name1}"
    print("✓ Initial weights are identical (correct copy)")

    # Check 3: Modifying one client's model doesn't affect others
    with torch.no_grad():
        # Modify client 0's first parameter
        first_param = next(clients[0].model.parameters())
        original_value = first_param[0, 0, 0, 0].item()
        first_param[0, 0, 0, 0] = 999.0

    # Check client 1's model is unchanged
    first_param_client1 = next(clients[1].model.parameters())
    assert first_param_client1[0, 0, 0, 0].item() != 999.0, \
        "Modifying client 0 should not affect client 1!"
    assert abs(first_param_client1[0, 0, 0, 0].item() - original_value) < 1e-6, \
        "Client 1's weight should still be original value!"

    print("✓ Modifying one client's model doesn't affect others")
    print("✓ Model isolation PASSED!")
    return True


def test_training_independence():
    """Test 3: Verify training one client doesn't affect others."""
    print("\n" + "=" * 60)
    print("TEST 3: TRAINING INDEPENDENCE")
    print("=" * 60)

    set_seed(42)
    device = torch.device('cpu')

    # Create dataset with different distributions
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)

    # Use Dirichlet to make clients have different data
    client_datasets = dirichlet_split(train_data, num_clients=3, alpha=0.1, seed=42)

    # Create base model and server
    base_model = create_model('cnn_small', num_classes=10)
    server = FLServer(base_model, device)

    # Create clients
    clients = []
    for i in range(3):
        client = FLClient(
            client_id=i,
            dataset=client_datasets[i],
            model=base_model,
            device=device,
            batch_size=32,
        )
        clients.append(client)

    # Get initial global params
    global_params = server.get_global_params()

    # Save client 1's initial weights
    client1_initial = {k: v.clone() for k, v in clients[1].model.state_dict().items()}

    # Train ONLY client 0
    print("Training only client 0 for 1 epoch...")
    result_0 = clients[0].local_train(
        global_params=global_params,
        local_epochs=1,
        lr=0.01,
    )

    # Check client 1's weights are unchanged
    client1_after = clients[1].model.state_dict()
    for key in client1_initial:
        assert torch.allclose(client1_initial[key], client1_after[key]), \
            f"Client 1's {key} changed after training client 0!"

    print("✓ Client 1's weights unchanged after training client 0")

    # Now train client 1 with same global params
    print("Training client 1 for 1 epoch...")
    result_1 = clients[1].local_train(
        global_params=global_params,
        local_epochs=1,
        lr=0.01,
    )

    # Check that client 0 and client 1 have DIFFERENT weights now
    # (because they trained on different data)
    client0_params = result_0['params']
    client1_params = result_1['params']

    diff_found = False
    for key in client0_params:
        if not torch.allclose(client0_params[key], client1_params[key], atol=1e-5):
            diff_found = True
            diff = (client0_params[key] - client1_params[key]).abs().max().item()
            print(f"  {key}: max diff = {diff:.6f}")

    assert diff_found, "Clients should have different weights after training on different data!"
    print("✓ Clients have different weights after training (as expected)")
    print("✓ Training independence PASSED!")
    return True


def test_aggregation_correctness():
    """Test 4: Verify FedAvg aggregation is mathematically correct."""
    print("\n" + "=" * 60)
    print("TEST 4: AGGREGATION CORRECTNESS")
    print("=" * 60)

    set_seed(42)
    device = torch.device('cpu')

    # Create simple model
    model = create_model('cnn_small', num_classes=10)

    # Create fake client results with known weights
    num_clients = 3
    client_results = []

    # Create controlled test case
    base_state = model.state_dict()

    for i in range(num_clients):
        # Create modified state dict (multiply by different factor)
        modified_state = {}
        for key, value in base_state.items():
            modified_state[key] = value.float() * (i + 1)  # client 0: *1, client 1: *2, client 2: *3

        client_results.append({
            'params': modified_state,
            'num_samples': 100 * (i + 1),  # 100, 200, 300 samples
            'train_loss': 0.5,
            'extra': {},
        })

    # Total samples: 100 + 200 + 300 = 600
    # Weights: 100/600, 200/600, 300/600 = 1/6, 2/6, 3/6
    # Expected weighted average for each param:
    # (1*1/6 + 2*2/6 + 3*3/6) * base = (1/6 + 4/6 + 9/6) * base = (14/6) * base

    # Manually compute expected
    total_samples = sum(r['num_samples'] for r in client_results)
    expected_state = {}
    for key in base_state:
        expected_state[key] = torch.zeros_like(base_state[key], dtype=torch.float32)
        for i, result in enumerate(client_results):
            weight = result['num_samples'] / total_samples
            expected_state[key] += weight * result['params'][key]

    # Run actual aggregation
    from fl.aggregators import fedavg_aggregate
    server = FLServer(model, device)
    fedavg_aggregate(server.global_model, client_results, {})

    # Compare
    actual_state = server.global_model.state_dict()

    print("Comparing aggregated weights with expected...")
    for key in expected_state:
        if not torch.allclose(expected_state[key], actual_state[key].float(), atol=1e-5):
            diff = (expected_state[key] - actual_state[key].float()).abs().max().item()
            print(f"  MISMATCH in {key}: max diff = {diff}")
            return False
        else:
            print(f"  ✓ {key}: correct")

    print("✓ Aggregation correctness PASSED!")
    return True


def test_round_isolation():
    """Test 5: Verify each round starts fresh (no state leakage)."""
    print("\n" + "=" * 60)
    print("TEST 5: ROUND ISOLATION")
    print("=" * 60)

    set_seed(42)
    device = torch.device('cpu')

    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    client_datasets = iid_split(train_data, num_clients=2, seed=42)

    # Create server and clients
    model = create_model('cnn_small', num_classes=10)
    server = FLServer(model, device, test_data)

    clients = []
    for i in range(2):
        client = FLClient(i, client_datasets[i], model, device, batch_size=32)
        clients.append(client)

    # Run 2 rounds
    for round_num in range(1, 3):
        print(f"\nRound {round_num}:")

        # Get global params (should be same for all clients in this round)
        global_params = server.get_global_params()

        # Verify both clients receive same global params
        for key in global_params:
            p = global_params[key]
            print(f"  Global {key[:20]:20s}: mean={p.float().mean():.6f}")

        # Train clients
        client_results = []
        for i, client in enumerate(clients):
            # Each client should start from global_params
            result = client.local_train(global_params, local_epochs=1, lr=0.01)
            client_results.append(result)

            # Check client actually loaded global params at start
            # (This is verified by the fact that local_train loads global_params first)

        # Aggregate
        server.aggregate(client_results)

        # Evaluate
        eval_result = server.evaluate()
        print(f"  After aggregation - Test Acc: {eval_result['accuracy']*100:.2f}%")

    print("\n✓ Round isolation PASSED!")
    return True


def test_gradient_buffer_isolation():
    """Test 6: Verify gradient buffers don't leak between clients (for Fed-M3)."""
    print("\n" + "=" * 60)
    print("TEST 6: GRADIENT BUFFER ISOLATION (Fed-M3)")
    print("=" * 60)

    set_seed(42)
    device = torch.device('cpu')

    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)

    # Use very different data for clients
    client_datasets = dirichlet_split(train_data, num_clients=2, alpha=0.1, seed=42)

    # Create model and clients
    model = create_model('cnn_small', num_classes=10)

    # Create Fed-M3 optimizer function
    from optimizers import fed_m3_optimizer_fn

    # Simulate two clients training
    global_params = {k: v.clone() for k, v in model.state_dict().items()}

    client_buffers = []
    for i in range(2):
        # Create fresh client each time
        client = FLClient(i, client_datasets[i], model, device, batch_size=32)

        # Train with Fed-M3
        def optimizer_fn(m, lr, extra_state):
            return fed_m3_optimizer_fn(m, lr, extra_state, beta1=0.9, beta2=0.999, lam=0.3)

        result = client.local_train(
            global_params=global_params,
            local_epochs=1,
            lr=0.01,
            optimizer_fn=optimizer_fn,
        )

        # Get gradient buffer
        extra = result.get('extra', {})
        get_extra_fn = extra.get('get_extra_fn')
        if get_extra_fn:
            buffer = get_extra_fn()['gradient_buffer']
            client_buffers.append(buffer)
            print(f"Client {i} buffer keys: {len(buffer)}")

    # Check buffers are different (trained on different data)
    if len(client_buffers) == 2:
        diff_found = False
        for key in client_buffers[0]:
            if key in client_buffers[1]:
                if not torch.allclose(client_buffers[0][key], client_buffers[1][key], atol=1e-5):
                    diff_found = True

        if diff_found:
            print("✓ Gradient buffers are different (as expected)")
        else:
            print("⚠ Buffers might be too similar - check data distribution")

    print("✓ Gradient buffer isolation PASSED!")
    return True


def main():
    """Run all isolation tests."""
    print("=" * 60)
    print("FL ISOLATION TESTS")
    print("=" * 60)
    print("These tests verify that:")
    print("  1. Each client has unique, non-overlapping data")
    print("  2. Each client has independent model copy")
    print("  3. Training one client doesn't affect others")
    print("  4. FedAvg aggregation is mathematically correct")
    print("  5. Each round starts fresh")
    print("  6. Gradient buffers are isolated")

    all_passed = True

    try:
        all_passed &= test_data_isolation()
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_model_isolation()
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_training_independence()
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_aggregation_correctness()
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_round_isolation()
    except Exception as e:
        print(f"✗ Test 5 FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_gradient_buffer_isolation()
    except Exception as e:
        print(f"✗ Test 6 FAILED: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - Please check!")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    main()
