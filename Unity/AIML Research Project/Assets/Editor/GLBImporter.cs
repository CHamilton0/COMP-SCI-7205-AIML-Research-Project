using UnityEngine;
using UnityEditor;
using UnityEditor.Callbacks;
using System.IO;

public class GLBImportHandler : AssetPostprocessor
{
    // Path to your custom material with a vertex color shader
    private static readonly string VertexColorMaterialPath = "Assets/Materials/VertexColor.mat";

    static void OnPostprocessAllAssets(
        string[] importedAssets,
        string[] deletedAssets,
        string[] movedAssets,
        string[] movedFromAssetPaths)
    {
        foreach (string assetPath in importedAssets)
        {
            if (assetPath.ToLower().EndsWith(".glb"))
            {
                AddGLBToScene(assetPath);
            }
        }
    }

    private static void AddGLBToScene(string assetPath)
    {
        // Load the imported GLB asset
        GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(assetPath);
        if (prefab == null)
        {
            Debug.LogWarning($"Failed to load GLB prefab at {assetPath}");
            return;
        }

        // Instantiate in the scene
        GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
        instance.transform.position = Vector3.zero;

        // Load the vertex color material
        Material vertexMat = AssetDatabase.LoadAssetAtPath<Material>(VertexColorMaterialPath);
        if (vertexMat != null)
        {
            foreach (var renderer in instance.GetComponentsInChildren<Renderer>())
            {
                renderer.sharedMaterial = vertexMat;
            }
        }
        else
        {
            Debug.LogWarning($"Vertex color material not found at {VertexColorMaterialPath}");
        }

        Debug.Log($"Auto-imported and added GLB to scene: {assetPath}");
    }
}
