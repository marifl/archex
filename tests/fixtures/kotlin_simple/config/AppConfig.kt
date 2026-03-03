package com.example.config

object AppConfig {
    val appName = "MyApp"
    val version = "1.0.0"
    private val secret = "hidden"

    fun buildUrl(host: String, port: Int): String {
        return "http://$host:$port"
    }
}
