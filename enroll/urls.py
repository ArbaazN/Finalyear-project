from django.contrib import admin
from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.home,name="home"),
    path('signup',views.signup, name="signup"),
    path('signin',views.signin, name="signin"),
    path('signout',views.signout, name="signout"),
    path('job_search',views.job_search,name='job_search'),
    path('search',views.search,name='search'),
    path('pagination',views.pagination,name='pagination'),
    path('sea_pag',views.sea_pag,name='sea_pag'),
    path('rsm_a',views.rsm_a,name='rsm_a'),
    path('predict/',views.predict, name='predict'),
    path('predict/result',views.result),
    path('display',views.display,name='display'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)