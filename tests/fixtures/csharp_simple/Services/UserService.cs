using System.Collections.Generic;
using MyApp.Models;

namespace MyApp.Services;

public interface IUserService
{
    User FindById(int id);
    List<User> FindAll();
    void Save(User user);
}

public class UserServiceImpl : IUserService
{
    private readonly List<User> _users = new();

    public User FindById(int id)
    {
        return _users[id];
    }

    public List<User> FindAll()
    {
        return _users;
    }

    public void Save(User user)
    {
        _users.Add(user);
    }

    private void Validate(User user)
    {
        if (user == null) throw new ArgumentNullException(nameof(user));
    }
}
