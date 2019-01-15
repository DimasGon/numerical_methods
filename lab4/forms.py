from django import forms

class InputDataForm(forms.Form):
    
    CHOICES_METHOD = (
        ('Метод переменных направлений', 'Метод переменных направлений'),
        ('Метод дробных шагов', 'Метод дробных шагов')
    )
    CHOICES_APPROXIMATION = (
        ('Двухточечная 1-ого порядка', 'Двухточечная 1-ого порядка'),
        ('Двухточечная 2-ого порядка', 'Двухточечная 2-ого порядка'),
        ('Трехточечная 2-ого порядка', 'Трехточечная 2-ого порядка'),
    )

    method = forms.ChoiceField(label='Выберите метод решения', choices=CHOICES_METHOD, widget=forms.RadioSelect, required=True)
    a = forms.FloatField(label='Константа a', required=True)
    t_end = forms.FloatField(label='Окончание по времени', required=True)
    num_split_t = forms.FloatField(label='Количество разбиений по времени', required=True)
    num_split_x = forms.FloatField(label='Количество разбиений икса', required=True)
    num_split_y = forms.FloatField(label='Количество разбиений игрека', required=True)