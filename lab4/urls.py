from django.urls import path
from .views import Lab4View

app_name = 'lab4'

urlpatterns = [
    path('', Lab4View.as_view(), name='lab4'),
]
