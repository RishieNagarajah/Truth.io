from django.urls import path
from . import views

urlpatterns = [
    path('process_content_1', views.process_content_1, name='process_content_1'),
    path('process_content_2', views.process_content_2, name='process_content_2')
]
