from django.shortcuts import render
from django.views.generic import View
import os
from .forms import InputDataForm
from .laba import solve

class Lab1View(View):

    template_name = 'lab1/laba.html'
    show_graph = False

    def get(self, request):
        form = InputDataForm()
        return render(request, self.template_name, {
            'form': form,
        })

    def post(self, request):
        method = request.POST.get('method')
        approximation = request.POST.get('approximation')
        a = float(request.POST.get('a'))
        # x_end = float(request.POST.get('x_end'))
        t_end = float(request.POST.get('t_end'))
        num_split = int(request.POST.get('num_split'))
        sigma = float(request.POST.get('sigma'))
        solve(a, num_split, t_end, sigma, approximation, method)
        root_dir = os.path.abspath(os.curdir)
        os.remove(root_dir + '\common_static\img\graph.png')
        os.remove(root_dir + '\common_static\img\error.png')
        os.rename(root_dir + '\graph.png', root_dir + '\common_static\img\graph.png')
        os.rename(root_dir + '\error.png', root_dir + '\common_static\img\error.png')
        form = InputDataForm(request.POST)
        return render(request, self.template_name, {
            'form': form, 'show_graph': True,
        })