using System;
using static System.String;

namespace MyApp.Utils;

public static class StringExtensions
{
    public const string Empty = "";
    private const string DefaultPad = " ";

    public static string ToTitleCase(this string s)
    {
        if (IsNullOrEmpty(s)) return s;
        return char.ToUpper(s[0]) + s[1..].ToLower();
    }

    public static string Truncate(this string s, int maxLength)
    {
        return s.Length <= maxLength ? s : s[..maxLength];
    }

    internal static string PadCenter(this string s, int width)
    {
        int padding = width - s.Length;
        return s.PadLeft(s.Length + padding / 2).PadRight(width);
    }
}
