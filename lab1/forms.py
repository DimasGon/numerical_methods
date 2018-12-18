from django import forms

class InputDataForm(forms.Form):
    
    CHOICES_METHOD = (
        ('Явный', 'Явный'),
        ('Неявный', 'Неявный'),
        ('Кранка-Николсона', 'Кранка-Николсона')
    )
    CHOICES_APPROXIMATION = (
        ('Двухточечная 1-ого порядка', 'Двухточечная 1-ого порядка'),
        ('Двухточечная 2-ого порядка', 'Двухточечная 2-ого порядка'),
        ('Трехточечная 2-ого порядка', 'Трехточечная 2-ого порядка'),
    )

    method = forms.ChoiceField(label='Выберите метод решения', choices=CHOICES_METHOD, widget=forms.RadioSelect, required=True)
    approximation = forms.ChoiceField(label='Выберите аппроксимацию', choices=CHOICES_APPROXIMATION, widget=forms.RadioSelect, required=True)
    a = forms.FloatField(label='Константа a', required=True)
    t_end = forms.FloatField(label='Окончание по времени', required=True)
    num_split = forms.FloatField(label='Количество разбиений икса', required=True)
    sigma = forms.FloatField(label='Число Куранта', required=True)