using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections.Generic;
using System.Text.RegularExpressions;

public static class Slugifier
{
    public static string Slugify(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return string.Empty;

        // Trim, convert to lowercase
        text = text.Trim().ToLowerInvariant();

        // Replace any non-alphanumeric or dash characters with a dash
        text = Regex.Replace(text, @"[^a-z0-9\-]+", "-");

        // Trim leading/trailing dashes
        return text.Trim('-');
    }
}

public class GLBSceneImporter : AssetPostprocessor
{
    private static readonly string JSONPath = "Assets/scene.json";

    [System.Serializable]
    public class SceneObject
    {
        public string name;
        public Vector3 position;
        public Vector3 size;
        public Vector3 rotation_euler_angles_degrees;
    }

    [System.Serializable]
    public class SceneData
    {
        public List<SceneObject> objects;
        public SceneObject camera;
    }

    // --- AUTO: Triggered on asset import ---
    static void OnPostprocessAllAssets(
        string[] importedAssets,
        string[] deletedAssets,
        string[] movedAssets,
        string[] movedFromAssetPaths)
    {
        bool hasGLB = false;

        foreach (string assetPath in importedAssets)
        {
            if (assetPath.ToLower().EndsWith(".glb"))
            {
                hasGLB = true;
                Debug.Log($"GLB Imported: {assetPath}");
            }
        }

        if (hasGLB && File.Exists(JSONPath))
        {
            LoadAndPlaceSceneFromJSON(JSONPath);
        }
    }

    // --- MANUAL: Run from Unity menu ---
    [MenuItem("Tools/Rebuild GLB Scene")]
    public static void ManualRebuildScene()
    {
        if (!File.Exists(JSONPath))
        {
            Debug.LogError("Scene JSON not found: " + JSONPath);
            return;
        }

        LoadAndPlaceSceneFromJSON(JSONPath);
    }

    private static void LoadAndPlaceSceneFromJSON(string path)
    {
        string json = File.ReadAllText(path);
        SceneData scene = JsonUtility.FromJson<SceneData>(json);

        // Clear previous scene root
        GameObject oldRoot = GameObject.Find("SceneRoot");
        if (oldRoot != null)
        {
            Object.DestroyImmediate(oldRoot);
        }

        // Create a new root for organization
        GameObject sceneRoot = new GameObject("SceneRoot");

        foreach (var obj in scene.objects)
        {
            string glbPath = $"Assets/GLB/{obj.name}.glb";
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(glbPath);
            if (prefab == null)
            {
                Debug.LogWarning($"No GLB prefab found for '{obj.name}' at {glbPath}");
                continue;
            }

            GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
            instance.name = obj.name;
            instance.transform.SetParent(sceneRoot.transform);  // Parent under root
            instance.transform.position = obj.position;
            instance.transform.localScale = obj.size;
            instance.transform.rotation = Quaternion.Euler(obj.rotation_euler_angles_degrees);

            Debug.Log($"Placed '{obj.name}' at {obj.position} with size {obj.size}");
        }

        // Update the position and angle of the camera
        Camera mainCamera = Camera.main;
        if (mainCamera != null)
        {
            mainCamera.transform.position = scene.camera.position;
            mainCamera.transform.rotation = Quaternion.Euler(scene.camera.rotation_euler_angles_degrees);
        }
    }
}
