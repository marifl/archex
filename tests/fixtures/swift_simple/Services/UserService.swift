import Foundation

public protocol UserRepository {
    func findById(_ id: String) -> User?
    func findAll() -> [User]
    func save(_ user: User)
}

open class UserService: UserRepository {
    private var storage: [String: User] = [:]

    public init() {}

    public func findById(_ id: String) -> User? {
        return storage[id]
    }

    public func findAll() -> [User] {
        return Array(storage.values)
    }

    public func save(_ user: User) {
        storage[user.id] = user
    }

    public func addUser(_ user: User) {
        save(user)
    }

    fileprivate func internalCleanup() {
        storage.removeAll()
    }
}

extension UserService {
    public func userCount() -> Int {
        return storage.count
    }

    public func removeUser(id: String) {
        storage.removeValue(forKey: id)
    }
}
