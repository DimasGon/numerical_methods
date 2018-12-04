from django.shortcuts import render
from django.views.generic import TemplateView

class Lab3View(TemplateView):
    template_name = 'lab3/laba.html'