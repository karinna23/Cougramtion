from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name="index"),
    # path("login", views.login_view, name="login"),
    # path('Register', views.register, name='register'),
    path("Recommend", views.Recommend, name="recommend"),
    path("pdf/id=<int:id>/", views.pdf, name="pdf"),
    
   
]