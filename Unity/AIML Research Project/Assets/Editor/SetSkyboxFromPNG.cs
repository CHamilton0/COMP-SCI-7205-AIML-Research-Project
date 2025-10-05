using UnityEngine;
using UnityEditor;
using System.IO;

public class AutoSkyboxProcessor : AssetPostprocessor
{
    private static readonly string TargetSkyboxPath = "Assets/background.png";
    private static readonly string OutputMaterialPath = "Assets/Materials/AutoSkybox.mat";

    static void OnPostprocessAllAssets(
        string[] importedAssets,
        string[] deletedAssets,
        string[] movedAssets,
        string[] movedFromAssetPaths)
    {
        foreach (string assetPath in importedAssets)
        {
            if (assetPath.Equals(TargetSkyboxPath, System.StringComparison.OrdinalIgnoreCase))
            {
                ApplySkyboxFromPNG(assetPath);
                break;
            }
        }
    }

    public static void ApplySkyboxFromPNG(string pngAssetPath)
    {
        // Adjust texture import settings
        TextureImporter importer = AssetImporter.GetAtPath(pngAssetPath) as TextureImporter;
        if (importer != null)
        {
            importer.textureShape = TextureImporterShape.Texture2D;
            importer.wrapMode = TextureWrapMode.Clamp;
            importer.sRGBTexture = true;
            importer.SaveAndReimport();
        }

        // Load texture
        Texture2D tex = AssetDatabase.LoadAssetAtPath<Texture2D>(pngAssetPath);
        if (tex == null)
        {
            Debug.LogError("Failed to load texture: " + pngAssetPath);
            return;
        }

        // Create or replace material
        Material skyboxMat = AssetDatabase.LoadAssetAtPath<Material>(OutputMaterialPath);
        if (skyboxMat == null)
        {
            skyboxMat = new Material(Shader.Find("Skybox/Panoramic"));
            AssetDatabase.CreateAsset(skyboxMat, OutputMaterialPath);
        }

        skyboxMat.shader = Shader.Find("Skybox/Panoramic");
        skyboxMat.SetTexture("_MainTex", tex);
        skyboxMat.SetFloat("_Mapping", 1);    // Latitude-Longitude layout
        skyboxMat.SetFloat("_ImageType", 0);  // 2D texture
        EditorUtility.SetDirty(skyboxMat);
        AssetDatabase.SaveAssets();

        // Assign to Render Settings
        RenderSettings.skybox = skyboxMat;
        DynamicGI.UpdateEnvironment();

        Debug.Log($"Skybox updated from PNG: {pngAssetPath}");
    }

    public static void ApplySkyboxToMaterial(Material skyboxMat, string pngAssetPath)
    {
        if (skyboxMat == null)
        {
            Debug.LogError("Skybox material is null.");
            return;
        }
        // Load texture
        Texture2D tex = AssetDatabase.LoadAssetAtPath<Texture2D>(pngAssetPath);
        if (tex == null)
        {
            Debug.LogError("Failed to load texture: " + pngAssetPath);
            return;
        }
        skyboxMat.shader = Shader.Find("Skybox/Panoramic");
        skyboxMat.SetTexture("_MainTex", tex);
        skyboxMat.SetFloat("_Mapping", 1);    // Latitude-Longitude layout
        skyboxMat.SetFloat("_ImageType", 0);  // 2D texture
        EditorUtility.SetDirty(skyboxMat);
        AssetDatabase.SaveAssets();
        
        RenderSettings.skybox = skyboxMat;
        DynamicGI.UpdateEnvironment();
        Debug.Log($"Skybox material updated from PNG: {pngAssetPath}");
    }
}
