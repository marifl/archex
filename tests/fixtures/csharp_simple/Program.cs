using System;
using MyApp.Models;
using MyApp.Services;

namespace MyApp;

public class Program
{
    public static void Main(string[] args)
    {
        var service = new UserServiceImpl();
        var users = service.FindAll();
        Console.WriteLine($"Found {users.Count} users");
    }
}
