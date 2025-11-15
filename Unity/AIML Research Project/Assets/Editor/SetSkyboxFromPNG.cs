using UnityEngine;
using UnityEditor;
using System.IO;

public class AutoSkyboxProcessor : AssetPostprocessor
{
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
