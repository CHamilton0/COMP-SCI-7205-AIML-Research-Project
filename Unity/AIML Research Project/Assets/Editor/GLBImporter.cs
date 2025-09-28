using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections.Generic;
using System.Text.RegularExpressions;

public static class Slugifier
{
    public static string Slugify(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return string.Empty;
        text = text.Trim().ToLowerInvariant();
        text = Regex.Replace(text, @"[^a-z0-9\-]+", "-");
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

    static void OnPostprocessAllAssets(
        string[] importedAssets,
        string[] deletedAssets,
        string[] movedAssets,
        string[] movedFromAssetPaths)
    {
        bool hasGLB = false;
        foreach (string assetPath in importedAssets)
            if (assetPath.ToLower().EndsWith(".glb")) hasGLB = true;

        if (hasGLB && File.Exists(JSONPath))
            EditorApplication.delayCall += () => LoadAndPlaceSceneFromJSON(JSONPath);
    }

    [MenuItem("Tools/Rebuild GLB Scene")]
    public static void ManualRebuildScene()
    {
        if (!File.Exists(JSONPath)) { Debug.LogError("Scene JSON not found"); return; }
        LoadAndPlaceSceneFromJSON(JSONPath);
    }

    private static void ClearScene()
    {
        GameObject oldRoot = GameObject.Find("SceneRoot");
        if (oldRoot != null) Object.DestroyImmediate(oldRoot);
    }

    private static void LoadAndPlaceSceneFromJSON(string path)
    {
        string json = File.ReadAllText(path);
        SceneData scene = JsonUtility.FromJson<SceneData>(json);

        ClearScene();

        GameObject sceneRoot = new GameObject("SceneRoot");

        foreach (var obj in scene.objects)
        {
            string glbPath = $"Assets/GLB/{Slugifier.Slugify(obj.name)}.glb";
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(glbPath);
            if (prefab == null) continue;

            GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab, sceneRoot.transform);
            instance.name = obj.name;

            // Apply rotation/scale first
            instance.transform.rotation = Quaternion.Euler(obj.rotation_euler_angles_degrees);
            instance.transform.localScale = obj.size;

            // Compute renderer bounds offset so visual center matches JSON position
            Renderer[] rends = instance.GetComponentsInChildren<Renderer>();
            if (rends.Length > 0)
            {
                Bounds b = rends[0].bounds;
                for (int i = 1; i < rends.Length; i++) b.Encapsulate(rends[i].bounds);
                Vector3 offset = b.center - instance.transform.position;
                instance.transform.position = obj.position - offset;
            }
            else
            {
                instance.transform.position = obj.position;
            }
        }

        // Camera
        Camera mainCamera = Camera.main;
        if (mainCamera != null && scene.camera != null)
        {
            mainCamera.transform.position = scene.camera.position;
            mainCamera.transform.rotation = Quaternion.Euler(scene.camera.rotation_euler_angles_degrees);
        }
    }
}
