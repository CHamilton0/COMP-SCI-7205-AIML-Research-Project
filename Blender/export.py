import sys
from typing import Any

import bpy


def create_vertex_color_material(obj: Any, mat_name: str = "VertexColorMat") -> None:
    # Create a new material with vertex color node connected to base color
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")

    # Add Vertex Color node
    vc_node = nodes.new(type="ShaderNodeVertexColor")

    # Get the vertex color layer
    if obj.data.vertex_colors:
        vc_node.layer_name = obj.data.vertex_colors[0].name
    else:
        print(f"Warning: No vertex color layer found on object {obj.name}")
        vc_node.layer_name = "Col"

    # Link Vertex Color output to Base Color input
    links.new(vc_node.outputs["Color"], bsdf.inputs["Base Color"])

    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def main() -> None:
    argv = sys.argv

    args_start_index = argv.index("--") + 1
    argv = argv[args_start_index:]

    if len(argv) < 2:
        print("Usage: blender -b -P script.py -- input_file output_file")
        return

    input_file = argv[0]
    output_file = argv[1]

    # Reset to factory settings
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import model
    ext = input_file.split(".")[-1].lower()
    if ext == "ply":
        bpy.ops.import_mesh.ply(filepath=input_file)
    elif ext == "obj":
        bpy.ops.import_scene.obj(filepath=input_file)
    else:
        print(f"Unsupported input format: {ext}")
        return

    # Assign vertex color material to all meshes
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            create_vertex_color_material(obj)

    # Export glb with vertex colors and materials
    bpy.ops.export_scene.gltf(
        filepath=output_file,
        export_format="GLB",
        export_apply=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_colors=True,
        export_yup=True,
    )
    print(f"Exported {output_file} successfully.")


if __name__ == "__main__":
    main()
