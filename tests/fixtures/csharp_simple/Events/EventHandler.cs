using System;
using MyApp.Models;

namespace MyApp.Events;

public delegate void UserChangedHandler(User user);

public class UserEventSource
{
    public event UserChangedHandler? OnUserChanged;
    public event EventHandler? OnError;

    private int _eventCount;

    public void RaiseUserChanged(User user)
    {
        _eventCount++;
        OnUserChanged?.Invoke(user);
    }

    public void RaiseError()
    {
        OnError?.Invoke(this, EventArgs.Empty);
    }

    public int GetEventCount() { return _eventCount; }
}
