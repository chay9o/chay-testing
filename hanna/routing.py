from django.urls import path
from .consumers import LiveDataChatConsumer, ChatNoteConsumer, ImageQueryConsumer, SummaryGenerator, ChatConsumer

websocket_urlpatterns = [
    path('ws/live_data_chat_stream/', LiveDataChatConsumer.as_asgi()),
    path('ws/chat/', ChatConsumer.as_asgi()),
    path('ws/chat-note/', ChatNoteConsumer.as_asgi()),
    path('ws/image-query/', ImageQueryConsumer.as_asgi()),
    path('ws/summary-gen/', SummaryGenerator.as_asgi())
]
