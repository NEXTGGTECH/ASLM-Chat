# Copyright NEXTGGTECH. Elastic License 2.0.

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("Apps.API.urls")),
    path("", include("Apps.UI.urls")),
]
