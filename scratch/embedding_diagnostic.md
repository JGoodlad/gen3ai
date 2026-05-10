# Species Embedding Diagnostic Report

This report verifies the transformation of Species IDs into latent embedding vectors.

## Input Species
- Slot 0: **Tyranitar** (ID: 248)
- Slot 1: **Gengar** (ID: 94)
- Slot 2: **Skarmory** (ID: 227)

## Embedding Machinery Output

Below are the first 8 dimensions of the 32-dimensional embedding vector for each test species:

| Species | ID | Embedding Vector (First 8 Dims) |
| :--- | :--- | :--- |
| Tyranitar | 248 | `[-0.2566, -1.3911, -1.0204, -0.2536, -1.4589, -1.8465, 0.6839, -0.3131, ...]` |
| Gengar | 94 | `[0.5240, -0.0457, -2.2500, 1.0437, 0.6248, 1.0346, 0.5061, -1.0006, ...]` |
| Skarmory | 227 | `[0.1315, -1.1768, 0.0450, 0.2497, 0.7310, -0.3512, -1.5007, 2.5720, ...]` |

## Analysis
1. **Extraction**: The extractor correctly pulled IDs [248, 94, 227] from the 1684-dim vector.
2. **Transformation**: The `nn.Embedding` layer successfully mapped these IDs to unique 32-dimensional latent spaces.
3. **Differentiation**: Notice that even though the weights are currently random, each species has a unique, stable vector representing its 'identity' to the network.