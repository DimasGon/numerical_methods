from django import forms

class InputDataForm(forms.Form):
    
    CHOICES_METHOD = (
        ('Явный', 'Явный'),
        ('Неявный', 'Неявный'),
    )
    CHOICES_APPROXIMATION = (
        ('Двухточечная 1-ого порядка', 'Двухточечная 1-ого порядка'),
        ('Двухточечная 2-ого порядка', 'Двухточечная 2-ого порядка'),
        ('Трехточечная 2-ого порядка', 'Трехточечная 2-ого порядка'),
    )
    CHOICES_SECOND_INITIAL_CONDITION = (
        ('1-ого порядка', '1-ого порядка'),
        ('2-ого порядка', '2-ого порядка'),
    )

    method = forms.ChoiceField(label='Выберите метод решения', choices=CHOICES_METHOD, widget=forms.RadioSelect, required=True)
    approximation = forms.ChoiceField(label='Выберите аппроксимацию', choices=CHOICES_APPROXIMATION, widget=forms.RadioSelect, required=True)
    second_initial_condition = forms.ChoiceField(label='Выберите аппроксимацию 2-ого начального условия', choices=CHOICES_SECOND_INITIAL_CONDITION, widget=forms.RadioSelect, required=True)
    t_end = forms.FloatField(label='Окончание по времени', required=True)
    num_split = forms.IntegerField(label='Количество разбиений икса', required=True)
    sigma = forms.FloatField(label='Число Куранта', required=True)