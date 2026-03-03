using System;
using System.Collections.Generic;

namespace MyApp.Models;

public class User
{
    private string _name;
    private string _email;
    protected int age;
    public static readonly int MaxNameLength = 100;

    public User(string name, string email)
    {
        _name = name;
        _email = email;
    }

    public string Name
    {
        get { return _name; }
        set { _name = value; }
    }

    public string Email { get; private set; } = string.Empty;

    public int Age { get; protected set; }

    public string GetName() { return _name; }

    public void SetName(string name) { _name = name; }

    internal string GetEmail() { return _email; }

    private void Validate()
    {
        if (string.IsNullOrEmpty(_name))
            throw new ArgumentException("Name required");
    }
}

public record PersonRecord(string Name, int Age);

public struct Coordinate
{
    public double Latitude { get; set; }
    public double Longitude { get; set; }
}
