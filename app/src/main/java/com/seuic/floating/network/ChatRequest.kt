package com.seuic.floating.network

data class ChatRequest(
    val model: String,
    val messages: List<Message>,
    val stream: Boolean

) {
    data class Message(
        val role: String,
        val content: List<Content>,
        val temperature: Float
    )

    data class Content(
        val type: String,
        val text: String? = null,
        val image_url: ImageUrl? = null
    )

    data class ImageUrl(
        val url: String
    )
}

data class ChatResponse(
    val choices: List<Choice>
) {
    data class Choice(
        val delta: Delta
    ) {
        data class Delta(
            val content: String?
        )
    }
}
