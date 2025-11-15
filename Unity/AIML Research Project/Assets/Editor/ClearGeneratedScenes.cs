using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public class ClearGeneratedScenes
{
    private static readonly string ScenesDirectory = "Assets/generated-scenes";

    [MenuItem("Tools/Clear Generated Scenes")]
    public static void ClearScenes()
    {
        if (!Directory.Exists(ScenesDirectory))
        {
            Debug.LogError($"Scenes directory not found: {ScenesDirectory}");
            return;
        }

        // Close any open scenes first
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        string[] sceneDirs = Directory.GetDirectories(ScenesDirectory);
        foreach (string dir in sceneDirs)
        {
            DeleteFileAndMeta(Path.Combine(dir, "Scene.unity"));
            DeleteFileAndMeta(Path.Combine(dir, "screenshot.png"));
            DeleteFileAndMeta(Path.Combine(dir, "Skybox.mat"));
        }

        AssetDatabase.Refresh();
    }

    private static void DeleteFileAndMeta(string path)
    {
        File.Delete(path);
        File.Delete(path + ".meta");
    }
}
