import torch

A = torch.rand(2, 3)
B = torch.rand(3, 3)
print("Before")
1/0
print("After")

try:
    C = A + B
    print(C)
except Exception as e:
    print("Error:", e)

input("\nPress Enter to exit...")
