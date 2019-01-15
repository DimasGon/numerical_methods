from django.shortcuts import render
from django.views.generic import TemplateView
from django.shortcuts import render
from django.views.generic import View
import os
from .forms import InputDataForm
from .laba import solve

class Lab4View(View):

    template_name = 'lab4/laba.html'

    def get(self, request):

        form = InputDataForm()
        return render(request, self.template_name, {
            'form': form,
        })

    def post(self, request):

        method = request.POST.get('method')
        a = float(request.POST.get('a'))
        t_end = float(request.POST.get('t_end'))
        num_split_t = int(request.POST.get('num_split_t'))
        num_split_x = int(request.POST.get('num_split_x'))
        num_split_y = int(request.POST.get('num_split_y'))

        solve(method, a, t_end, num_split_t, num_split_x, num_split_y)
        root_dir = os.path.abspath(os.curdir)
        os.remove(root_dir + '/common_static/img/graph_0.png')
        os.remove(root_dir + '/common_static/img/graph_mid.png')
        os.remove(root_dir + '/common_static/img/error_0.png')
        os.remove(root_dir + '/common_static/img/error_mid.png')
        os.rename(root_dir + '/graph_0.png', root_dir + '/common_static/img/graph_0.png')
        os.rename(root_dir + '/graph_mid.png', root_dir + '/common_static/img/graph_mid.png')
        os.rename(root_dir + '/error_0.png', root_dir + '/common_static/img/error_0.png')
        os.rename(root_dir + '/error_mid.png', root_dir + '/common_static/img/error_mid.png')
        form = InputDataForm(request.POST)

        return render(request, self.template_name, {
            'form': form, 'show_graph': True,
        })