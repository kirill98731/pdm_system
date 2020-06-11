from django.db import models
from django.contrib.auth.models import User
# Create your models here.


# class system_manager(models.Manager):
#     def create_system(self, system_name, user_id):
#         system = self.create(system_name=system_name, user_id=user_id)
#         return system


class user_system(models.Model):
    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    system_name = models.CharField(max_length=50, default="")
    rul = models.IntegerField(default=-1)


class system_iot(models.Model):
    system_id = models.ForeignKey(user_system, on_delete=models.CASCADE)
    sensor_id = models.IntegerField()
    date = models.DateField()
    values = models.TextField()
