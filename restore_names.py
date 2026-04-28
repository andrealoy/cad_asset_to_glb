# restore_names.py
"""
After gltfpack, mesh.name is empty (gltfpack merges meshes by material).
But node.name is preserved (-kn). This script copies node.name -> mesh.name
so Houdini can recover part names via prim attribute.

Note: if multiple nodes share the same mesh (instancing), we concatenate
their names with '|' so we don't lose information.
"""

from pygltflib import GLTF2
import sys


def restore_mesh_names(input_path, output_path):
    gltf = GLTF2().load(input_path)
    
    # Build mesh_idx -> [node_name, ...] mapping
    mesh_to_nodes = {}
    for node in gltf.nodes:
        if node.mesh is not None and node.name:
            mesh_to_nodes.setdefault(node.mesh, []).append(node.name)
    
    # Apply names to meshes
    restored = 0
    for mesh_idx, names in mesh_to_nodes.items():
        if mesh_idx >= len(gltf.meshes):
            continue
        mesh = gltf.meshes[mesh_idx]
        if not mesh.name:
            # Concatenate with | if multiple instances point to same mesh
            mesh.name = "|".join(names) if len(names) > 1 else names[0]
            restored += 1
    
    gltf.save(output_path)
    print(f"  Restored {restored} mesh names from nodes")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python restore_names.py <input.glb> <output.glb>")
        sys.exit(1)
    
    restore_mesh_names(sys.argv[1], sys.argv[2])