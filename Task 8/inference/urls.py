from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/predict/", views.predict_view, name="predict"),
    path("api/preview/", views.preview_frames_view, name="preview"),
]

