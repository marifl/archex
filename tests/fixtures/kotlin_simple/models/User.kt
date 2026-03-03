package com.example.models

import com.example.utils.Extensions

data class User(
    val id: Int,
    val name: String,
    val email: String,
) {
    fun displayName(): String = "$name <$email>"

    private fun validate(): Boolean {
        return name.isNotBlank() && email.contains("@")
    }

    companion object {
        val MAX_NAME_LENGTH = 100

        fun fromMap(map: Map<String, Any>): User {
            return User(
                id = map["id"] as Int,
                name = map["name"] as String,
                email = map["email"] as String,
            )
        }
    }
}

sealed class UserResult {
    data class Success(val user: User) : UserResult()
    data class Error(val message: String) : UserResult()
}
