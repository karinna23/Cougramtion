from django import forms
from .models import PredResults


class PredictionForm(forms.ModelForm):
    class Meta:
        model = PredResults
        fields = ['first_name','last_name','sex','cet', 'gpa', 'strand']


