import dataclasses
from agents.model.flag_registry import REGISTRY

print("N =", len(REGISTRY))
r0 = REGISTRY[0]
print("fields:", [f.name for f in dataclasses.fields(r0)])
print(r0)
