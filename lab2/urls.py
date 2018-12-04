from django.urls import path
from .views import Lab2View

app_name = 'lab2'

urlpatterns = [
    path('', Lab2View.as_view(), name='lab2'),
]
