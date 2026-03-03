import Foundation
import SwiftUI

@main
struct AppEntry {
    static func main() {
        print("Starting app")
    }
}

func bootstrap() {
    let service = UserService()
    service.addUser(User(name: "Alice", age: 30))
}
