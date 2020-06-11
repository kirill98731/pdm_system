from django.contrib import admin

# Register your models here.
from .models import *

admin.site.register(user_system)
admin.site.register(system_iot)
#admin.site.register(system_manager)