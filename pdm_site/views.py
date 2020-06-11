from django.shortcuts import render
from django.shortcuts import redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, logout, login
from .machine_learning import create_new_model, update_model, add_iot, get_prediction
from .models import user_system
# Create your views here.


def start_page(request):
    if '_auth_user_id' not in request.session.keys():
        log = False
        array_system = []
    else:
        log = User.objects.get(id=request.session['_auth_user_id'])
        array_system = user_system.objects.filter(user_id=request.session['_auth_user_id'])

    if request.method == 'POST':
        name = request.POST['system_name']
        same_name = user_system.objects.filter(system_name=name, user_id=request.session['_auth_user_id'])
        if same_name:
            return render(request, 'pdm_site/start_page.html',
                          {'log': log, 'same_name': name, 'array_system': array_system})
        else:
            system = user_system.objects.create(system_name=name,
                                                user_id=User.objects.get(id=request.session['_auth_user_id']))
            system.save()

    return render(request, 'pdm_site/start_page.html', {'log': log, 'same_name': False, 'array_system': array_system})


def log(request):
    if request.method == 'POST':
        user = authenticate(username=request.POST['user'], password=request.POST['pass'])
        if user and user.is_active == True:
            login(request, user)
            return redirect('/')
        else:
            return render(request, 'pdm_site/log.html', {'message': 'Неверный логин или пароль'})
    return render(request, 'pdm_site/log.html')


def sign(request):
    if request.method == 'POST':
        try:
            user = User.objects.create_user(username=request.POST['user'], password=request.POST['pass'])
            user.save()
        except:
            return render(request, 'pdm_site/sign.html', {'message': 'Данный пользователь уже зарегистрирован'})
        return redirect('/login')
    return render(request, 'pdm_site/sign.html')


def out(request):
    logout(request)
    return redirect('/')


def add_model(request):
    if '_auth_user_id' not in request.session.keys():
        log = False
    else:
        log = User.objects.get(id=request.session['_auth_user_id'])
    try:
        if request.method == 'POST':
            restart = True if request.POST.get('restart') == 'on' else False
            if restart:
                mse, mae, r2 = create_new_model(request.FILES['file'])
                return render(request, 'pdm_site/add_model.html', {'log': log, 'mse': mse, 'mae': mae, 'r2': r2})
            else:
                mse, mae, r2 = update_model(request.FILES['file'])
                return render(request, 'pdm_site/add_model.html', {'log': log, 'mse': mse, 'mae': mae, 'r2': r2})
    except:
        pass
    return render(request, 'pdm_site/add_model.html', {'log': log, 'mse': False})


def del_system(request, system_id):
    if request.method == 'POST':
        try:
            user_system.objects.filter(id=system_id).delete()
        except:
            pass
    return redirect('/')


def add_iot(request, system_id):
    if request.method == 'POST':
        try:
            rul = round(get_prediction(request.FILES['file'])[0][0])
            user_system.objects.filter(id=system_id).update(rul=rul)
        except:
            pass
    return redirect('/')
