using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using static Codice.Client.Commands.WkTree.WorkspaceTreeNode;

public class GenerateMultipleScenes
{
    private static readonly string ScenesDirectory = "Assets/generated-scenes";

    [System.Serializable]
    public class JSONSceneObject
    {
        public string name;
        public Vector3 position;
        public Vector3 size;
        public Vector3 rotation_euler_angles_degrees;
    }

    [System.Serializable]
    public class JSONSceneData
    {
        public List<JSONSceneObject> objects;
        public JSONSceneObject camera;
    }

    [MenuItem("Tools/Generate Scenes")]
    public static void GenerateScenes()
    {
        // Loop through the scene directories and process each scene.json
        if (!Directory.Exists(ScenesDirectory))
        {
            Debug.LogError($"Scenes directory not found: {ScenesDirectory}");
            return;
        }

        string[] sceneDirs = Directory.GetDirectories(ScenesDirectory);
        foreach (string dir in sceneDirs)
        {
            LoadAndPlaceScene(dir);
            string jsonPath = Path.Combine(dir, "scene.json");
            if (File.Exists(jsonPath))
            {
                Debug.Log($"Processing scene: {dir}");
            }
            else
            {
                Debug.LogWarning($"No scene.json found in {dir}");
            }
        }
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

    private static Scene LoadAndPlaceSceneFromJSON(string dirPath)
    {
        string jsonPath = Path.Combine(dirPath, "scene.json");

        string json = File.ReadAllText(jsonPath);
        JSONSceneData sceneData = JsonUtility.FromJson<JSONSceneData>(json);

        foreach (var obj in sceneData.objects)
        {
            string glbPath = $"{dirPath}/GLB/{Slugifier.Slugify(obj.name)}.glb";
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(glbPath);
            if (prefab == null)
            {
                Debug.LogWarning($"Prefab not found at path: {glbPath}");
                continue;
            }


            Scene targetScene = SceneManager.GetActiveScene();
            GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab, targetScene);
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

        GameObject newCameraGameObject = new GameObject("Main Camera");
        newCameraGameObject.AddComponent<Camera>();

        Camera mainCamera = Camera.main;
        if (mainCamera != null && sceneData.camera != null)
        {
            mainCamera.transform.position = sceneData.camera.position;
            mainCamera.transform.rotation = Quaternion.Euler(sceneData.camera.rotation_euler_angles_degrees);
            mainCamera.fieldOfView = 60f;
        }

        return SceneManager.GetActiveScene();
    }

    private static void LoadAndPlaceScene(string dir)
    {
        // Create a new empty scene in this directory
        Scene newScene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        // Load the new scene in the editor
        EditorSceneManager.SaveScene(newScene, Path.Combine(dir, "Scene.unity"));
        EditorSceneManager.OpenScene(Path.Combine(dir, "Scene.unity"));

        Scene resultScene = LoadAndPlaceSceneFromJSON(dir);

        EditorSceneManager.SaveScene(resultScene, Path.Combine(dir, "Scene.unity"));
    }
}
