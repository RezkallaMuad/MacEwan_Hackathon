package com.macewan.hackathon

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform