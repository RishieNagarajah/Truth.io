from django.urls import path
from . import views

urlpatterns = [
    path('process_content', views.process_content, name='process_content')
]
