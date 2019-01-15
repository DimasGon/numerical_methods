from django.shortcuts import render
from django.views.generic import TemplateView
from django.shortcuts import render
from django.views.generic import View
import os
from .forms import InputDataForm
from .laba import solve

class Lab3View(View):

    template_name = 'lab3/laba.html'

    def get(self, request):

        form = InputDataForm()
        return render(request, self.template_name, {
            'form': form,
        })

    def post(self, request):

        method = request.POST.get('method')
        relax = float(request.POST.get('relax'))
        eps = float(request.POST.get('eps'))
        num_split_x = int(request.POST.get('num_split_x'))
        num_split_y = int(request.POST.get('num_split_y'))
        
        solve(method, relax, eps, num_split_x, num_split_y)
        root_dir = os.path.abspath(os.curdir)
        os.remove(root_dir + '/common_static/img/graph_mid.png')
        os.remove(root_dir + '/common_static/img/graph_mid_x.png')
        os.rename(root_dir + '/graph_mid.png', root_dir + '/common_static/img/graph_mid.png')
        os.rename(root_dir + '/graph_mid_x.png', root_dir + '/common_static/img/graph_mid_x.png')
        form = InputDataForm(request.POST)

        return render(request, self.template_name, {
            'form': form, 'show_graph': True,
        })