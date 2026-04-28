# clean_normals.py
"""
Clean GLB file before passing to gltfpack:
1. Fix degenerate / non-unit normals (zero-length, NaN, Inf)
2. Re-normalize all normals to unit length
3. Propagate node names to mesh names (so Houdini can read them after gltfpack)
"""

import numpy as np
from pygltflib import GLTF2
import sys


def clean_normals_data(gltf, blob):
    """Fix bad normals in-place in the binary blob."""
    fixed_total = 0
    primitives_processed = 0
    
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            if prim.attributes.NORMAL is None:
                continue
            
            acc_idx = prim.attributes.NORMAL
            accessor = gltf.accessors[acc_idx]
            buffer_view = gltf.bufferViews[accessor.bufferView]
            
            offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
            count = accessor.count
            
            # Read normals as numpy array
            normals = np.frombuffer(
                bytes(blob),
                dtype=np.float32,
                count=count * 3,
                offset=offset
            ).reshape(-1, 3).copy()
            
            # Detect bad normals (zero-length, NaN, Inf)
            lengths = np.linalg.norm(normals, axis=1)
            bad_mask = (lengths < 1e-6) | ~np.isfinite(lengths)
            
            n_bad = int(bad_mask.sum())
            if n_bad > 0:
                # Replace with +Y up; gltfpack will recompute during decimation
                normals[bad_mask] = [0.0, 1.0, 0.0]
                fixed_total += n_bad
            
            # Re-normalize all normals to unit length
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            lengths[lengths < 1e-6] = 1.0
            normals = normals / lengths
            
            # Write back to blob
            normals_bytes = normals.astype(np.float32).tobytes()
            blob[offset:offset + len(normals_bytes)] = normals_bytes
            primitives_processed += 1
    
    return primitives_processed, fixed_total


def propagate_node_names_to_meshes(gltf):
    """
    glTF stores names on nodes, but Houdini reads them from meshes.
    Copy node.name -> mesh.name so Houdini can recover part names.
    
    If multiple nodes share the same mesh (instancing), the first node
    name wins. This is fine because we want the canonical name of the
    geometry, not the per-instance name.
    """
    propagated = 0
    
    for node in gltf.nodes:
        if node.mesh is None or not node.name:
            continue
        
        mesh = gltf.meshes[node.mesh]
        if not mesh.name:
            mesh.name = node.name
            propagated += 1
    
    return propagated


def clean_glb(input_path, output_path):
    print(f"Loading {input_path}...")
    gltf = GLTF2().load(input_path)
    blob = bytearray(gltf.binary_blob())
    
    # Step 1: clean normals
    n_prims, n_fixed = clean_normals_data(gltf, blob)
    
    # Step 2: propagate node names -> mesh names
    n_propagated = propagate_node_names_to_meshes(gltf)
    
    # Save
    gltf.set_binary_blob(bytes(blob))
    gltf.save(output_path)
    
    print(f"  Processed {n_prims} primitives")
    print(f"  Fixed {n_fixed} bad normals")
    print(f"  Propagated {n_propagated} node names to meshes")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_normals.py <input.glb> <output.glb>")
        sys.exit(1)
    
    clean_glb(sys.argv[1], sys.argv[2])