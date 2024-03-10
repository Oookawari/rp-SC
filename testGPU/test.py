import torch
 
def test_gpu_availability():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "No GPU available"
    print(f"GPU available: {torch.cuda.is_available()}, Device: {device}")
 
if __name__ == "__main__":
    test_gpu_availability()