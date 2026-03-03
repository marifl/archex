package com.example.utils

typealias StringMap = Map<String, String>
typealias UserList = List<String>

fun String.toTitleCase(): String {
    return split(" ").joinToString(" ") { word ->
        word.lowercase().replaceFirstChar { it.uppercase() }
    }
}

fun String.isEmail(): Boolean {
    return contains("@") && contains(".")
}

fun Int.clamp(min: Int, max: Int): Int {
    return when {
        this < min -> min
        this > max -> max
        else -> this
    }
}
