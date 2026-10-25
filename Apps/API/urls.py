# Copyright NEXTGGTECH. Elastic License 2.0.

from django.urls import path

from . import views


# Register generation, model discovery, and context APIs served to external modules.
urlpatterns = [
    # Generation APIs.
    path("api/generate/", views.generate_api, name="generate_api"),
    path("api/chat/abort/", views.abort_generation_api, name="abort_generation_api"),

    # Model discovery APIs.
    path("api/models/", views.get_models_api, name="models_api"),
    path("api/model_info/", views.get_model_info_api, name="model_info_api"),
    path("api/model_capabilities/", views.get_model_capabilities_api, name="model_capabilities_api"),
    path("api/inference_info/", views.get_inference_info_api, name="inference_info_api"),

    # Context usage and compression APIs.
    path("api/context_usage/", views.get_context_usage_api, name="context_usage_api"),
    path("api/context_compress/", views.context_compress_api, name="context_compress_api"),
    path(
        "api/context_compression/build_event/",
        views.context_compression_build_event_api,
        name="context_compression_build_event_api",
    ),
    path(
        "api/context_compression/decide/",
        views.context_compression_decide_api,
        name="context_compression_decide_api",
    ),
]
