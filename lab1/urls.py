from django.urls import path
from .views import Lab1View

app_name = 'lab1'

urlpatterns = [
    path('', Lab1View.as_view(), name='lab1'),
]
