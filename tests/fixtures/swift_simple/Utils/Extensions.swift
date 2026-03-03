import Foundation

public typealias Completion = () -> Void
public typealias ResultHandler<T> = (Result<T, Error>) -> Void

extension String {
    public func trimmed() -> String {
        return trimmingCharacters(in: .whitespaces)
    }

    public var isBlank: Bool {
        return trimmed().isEmpty
    }
}

extension Array {
    public func safeElement(at index: Int) -> Element? {
        guard index >= 0 && index < count else { return nil }
        return self[index]
    }
}

public protocol Configurable {
    func configure(with options: [String: Any])
}
