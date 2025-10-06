using Unity.VisualScripting;
using UnityEngine;
using System.Collections;
using UnityEditor;

public class CameraUtils : MonoBehaviour
{
    public static void TakeScreenshot(string filePath, int width, int height)
    {
        Camera cam = Camera.main;
        if (cam == null) { Debug.LogError("No MainCamera found!"); return; }

        RenderTexture rt = new RenderTexture(width, height, 24);
        cam.targetTexture = rt;
        cam.Render();

        RenderTexture.active = rt;
        Texture2D screenShot = new Texture2D(width, height, TextureFormat.RGB24, false);
        screenShot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        screenShot.Apply();

        cam.targetTexture = null;
        RenderTexture.active = null;
        DestroyImmediate(rt);

        byte[] bytes = screenShot.EncodeToPNG();
        System.IO.File.WriteAllBytes(filePath, bytes);

        Debug.Log("Saved screenshot to " + filePath);
    }
}
