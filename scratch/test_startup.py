import os
from datetime import datetime

print("Starting test script...")
unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = f"models/test_dir_{unique_id}"
os.makedirs(model_dir, exist_ok=True)
print(f"Created {model_dir}")
with open(f"{model_dir}/success.txt", "w") as f:
    f.write("Success!")
print("Done.")
