from django.shortcuts import render
from django.views.generic import View
import os
from .forms import InputDataForm
from .laba import solve

class Lab2View(View):

    template_name = 'lab2/laba.html'

    def get(self, request):

        form = InputDataForm()
        return render(request, self.template_name, {
            'form': form,
        })

    def post(self, request):

        method = request.POST.get('method')
        approximation = request.POST.get('approximation')
        second_initial_condition = request.POST.get('second_initial_condition')
        t_end = float(request.POST.get('t_end'))
        num_split = int(request.POST.get('num_split'))
        sigma = float(request.POST.get('sigma'))

        solve(method, approximation, second_initial_condition, t_end, num_split, sigma)
        root_dir = os.path.abspath(os.curdir)
        os.remove(root_dir + '/common_static/img/graph.png')
        os.remove(root_dir + '/common_static/img/error.png')
        os.rename(root_dir + '/graph.png', root_dir + '/common_static/img/graph.png')
        os.rename(root_dir + '/error.png', root_dir + '/common_static/img/error.png')
        form = InputDataForm(request.POST)

        return render(request, self.template_name, {
            'form': form, 'show_graph': True,
        })