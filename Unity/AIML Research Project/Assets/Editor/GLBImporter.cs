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

    static Bounds GetBounds(GameObject go)
    {
        var renderers = go.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0) return new Bounds(go.transform.position, Vector3.zero);

        Bounds bounds = renderers[0].bounds;
        foreach (var r in renderers)
        {
            bounds.Encapsulate(r.bounds);
        }
        return bounds;
    }

    static float CalculateCameraDistance(Bounds bounds, Camera cam)
    {
        float fovVertical = cam.fieldOfView * Mathf.Deg2Rad;
        float aspect = cam.aspect;

        // size of the bounds in world space
        float height = bounds.size.y;
        float width = bounds.size.x;

        // half angles
        float halfFovV = fovVertical / 2f;
        float halfFovH = Mathf.Atan(Mathf.Tan(halfFovV) * aspect);

        // Required distances for vertical and horizontal fit:
        float distV = (height / 2f) / Mathf.Tan(halfFovV);
        float distH = (width / 2f) / Mathf.Tan(halfFovH);

        // use the larger one to ensure it fits both horizontally and vertically
        return Mathf.Max(distV, distH);
    }

    public static void FrameObject(Camera cam, GameObject target, Vector3 viewDirection)
    {
        Bounds bounds = GetBounds(target);
        Vector3 center = bounds.center;

        float distance = CalculateCameraDistance(bounds, cam);

        // Make sure viewDirection is normalized
        Vector3 dir = viewDirection.normalized;

        // Position camera:
        cam.transform.position = center - dir * distance;

        // Look at center with a chosen up vector (usually Vector3.up):
        cam.transform.rotation = Quaternion.LookRotation(dir, Vector3.up);

        // Optional: adjust near/far planes if needed
        cam.nearClipPlane = distance * 0.1f;
        cam.farClipPlane = distance * 4f;
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
            mainCamera.fieldOfView = 60f;
        }

        FrameObject(mainCamera, sceneRoot, mainCamera.transform.forward);
        CameraUtils.TakeScreenshot(Application.dataPath + "/shot.png", 1920, 1080);
    }
}
