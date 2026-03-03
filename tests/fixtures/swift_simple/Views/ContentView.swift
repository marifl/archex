import SwiftUI
import Foundation

public struct ContentView {
    public var title: String
    private var isLoading: Bool = false

    public init(title: String) {
        self.title = title
    }

    public func render() -> String {
        return "<ContentView title=\(title)>"
    }

    public mutating func startLoading() {
        isLoading = true
    }

    public mutating func stopLoading() {
        isLoading = false
    }
}

public struct UserListView {
    private var users: [User] = []

    public init(users: [User]) {
        self.users = users
    }

    public subscript(index: Int) -> User? {
        return users.safeElement(at: index)
    }
}
