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
