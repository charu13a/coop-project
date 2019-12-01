from django.urls import path
from word_suggestions import views

urlpatterns = [
    path('', views.index, name='index'),
    path('similar-words', views.similar_words, name='similar-words'),
    path('similar-words-new', views.similar_words_new, name='similar-words-new')
]