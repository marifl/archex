import Foundation

public struct User {
    public let name: String
    public var age: Int
    private var _id: String = UUID().uuidString

    public init(name: String, age: Int) {
        self.name = name
        self.age = age
    }

    public func displayName() -> String {
        return "\(name) (\(age))"
    }

    private func validate() -> Bool {
        return !name.isEmpty && age >= 0
    }
}

public protocol Identifiable {
    var id: String { get }
    func describe() -> String
}

extension User: Identifiable {
    public var id: String { return _id }
    public func describe() -> String { return displayName() }
}

public enum UserStatus {
    case active
    case inactive
    case suspended(reason: String)

    public func label() -> String {
        switch self {
        case .active: return "Active"
        case .inactive: return "Inactive"
        case .suspended(let reason): return "Suspended: \(reason)"
        }
    }
}

public typealias UserID = String
