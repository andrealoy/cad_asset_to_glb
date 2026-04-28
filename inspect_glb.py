# inspect.py
from pygltflib import GLTF2
import sys

gltf = GLTF2().load(sys.argv[1])

print(f"Total nodes: {len(gltf.nodes)}")
print(f"Total meshes: {len(gltf.meshes)}")
print(f"Nodes with name: {sum(1 for n in gltf.nodes if n.name)}")
print(f"Meshes with name: {sum(1 for m in gltf.meshes if m.name)}")

print("\nFirst 10 nodes:")
for i, node in enumerate(gltf.nodes[:10]):
    print(f"  [{i}] name={node.name!r}  mesh={node.mesh}")

print("\nFirst 10 meshes:")
for i, mesh in enumerate(gltf.meshes[:10]):
    print(f"  [{i}] name={mesh.name!r}  prims={len(mesh.primitives)}")