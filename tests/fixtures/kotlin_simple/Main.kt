package com.example

import com.example.models.User
import com.example.services.UserService
import kotlin.collections.List

fun main() {
    val service = UserServiceImpl()
    val user = service.findById(1)
    println(user)
}

fun greet(name: String): String {
    return "Hello, $name!"
}

private fun internalHelper(): String {
    return "helper"
}
