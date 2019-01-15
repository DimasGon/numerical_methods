from django import forms

class InputDataForm(forms.Form):
    
    CHOICES_METHOD = (
        ('Метод Зейделя', 'Метод Зейделя'),
        ('Метод Либмана', 'Метод Либмана'),
        ('Метод простых итераций с верхней релаксацией', 'Метод простых итераций с верхней релаксацией')
    )

    method = forms.ChoiceField(label='Выберите метод решения', choices=CHOICES_METHOD, widget=forms.RadioSelect, required=True)
    relax = forms.FloatField(label='Параметр релаксации', required=True)
    eps = forms.FloatField(label='Введите эпсилон', required=True)
    num_split_x = forms.FloatField(label='Количество разбиений икса', required=True)
    num_split_y = forms.FloatField(label='Количество разбиений игрека', required=True)