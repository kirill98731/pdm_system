"""Pdm_system URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""


from django.contrib import admin
from django.urls import path, re_path
from pdm_site.views import start_page, log, sign, out, add_model, del_system, add_iot

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', start_page),
    path('login/', log),
    path('sign/', sign),
    path('logout/', out),
    path('add_model/', add_model),
    re_path(r'^del_system/(?P<system_id>\d+)', del_system),
    re_path(r'^add_iot/(?P<system_id>\d+)', add_iot)
]
