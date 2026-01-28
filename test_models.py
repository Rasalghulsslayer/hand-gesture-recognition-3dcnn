import torch
from models.hrn import HRN
from models.lrn import LRN

def test():
    # Simulation d'un batch de 2 vidéos
    # HRN input: (Batch=2, Channel=1, Depth=32, Height=57, Width=125)
    dummy_hrn = torch.randn(2, 1, 32, 57, 125)
    # LRN input: (Batch=2, Channel=1, Depth=32, Height=28, Width=62)
    dummy_lrn = torch.randn(2, 1, 32, 28, 62)

    print("--- Test HRN ---")
    model_h = HRN(num_classes=34)
    out_h = model_h(dummy_hrn)
    print(f"Output shape: {out_h.shape}")
    assert out_h.shape == (2, 34), "Erreur dimension HRN"
    print("✅ HRN OK")

    print("\n--- Test LRN ---")
    model_l = LRN(num_classes=34)
    out_l = model_l(dummy_lrn)
    print(f"Output shape: {out_l.shape}")
    assert out_l.shape == (2, 34), "Erreur dimension LRN"
    print("✅ LRN OK")

if __name__ == "__main__":
    test()