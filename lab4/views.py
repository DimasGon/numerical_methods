from django.shortcuts import render
from django.views.generic import TemplateView
from django.shortcuts import render
from django.views.generic import View
import os
from .forms import InputDataForm

class Lab4View(View):

    template_name = 'lab4/laba.html'

    def get(self, request):

        form = InputDataForm()
        return render(request, self.template_name, {
            'form': form,
        })