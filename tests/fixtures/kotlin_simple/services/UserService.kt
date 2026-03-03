package com.example.services

import com.example.models.User

interface UserService {
    fun findById(id: Int): User?
    fun findAll(): List<User>
    fun save(user: User): User
}

class UserServiceImpl : UserService {
    private val users = mutableListOf<User>()

    override fun findById(id: Int): User? {
        return users.find { it.id == id }
    }

    override fun findAll(): List<User> = users.toList()

    override fun save(user: User): User {
        users.add(user)
        return user
    }

    internal fun count(): Int = users.size
}

fun UserService.exists(id: Int): Boolean {
    return findById(id) != null
}
