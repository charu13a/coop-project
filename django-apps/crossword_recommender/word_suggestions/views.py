from django.shortcuts import render
from word_suggestions.model import suggest_words, suggest_words_new, suggest_words_new2
from django.http import JsonResponse

def index(request):
	return render(request, 'index.html', {})

def similar_words(request):
    words_list = request.GET.getlist('words')
    print(request.session)
    clues_list = request.GET.getlist('clues')
    print("Words List:", words_list)
    print("Clues List:", clues_list)
    [suggested_words, words_not_in_vocab, words_map, clue_nouns]= suggest_words(words_list, clues_list, request)
    print("suggested_words:", suggested_words)
    print("words_not_in_vocab:", words_not_in_vocab)
    print("words_map:", words_map)
    return JsonResponse({'suggested_words' : suggested_words,
    						'words_not_in_vocab' : words_not_in_vocab,
    						'words_map': words_map,
    						'clue_nouns': clue_nouns}, safe=False)

def similar_words_new(request):
    words_list = request.GET.getlist('words')
    print(request.session)
    clues_list = request.GET.getlist('clues')
    #negative_list = request.GET.getlist('negative_list')
    print("Words List:", words_list)
    print("Clues List:", clues_list)
    [suggested_words, words_not_in_vocab, words_map, clue_nouns]= suggest_words_new(words_list, clues_list, request)
    print("suggested_words:", suggested_words)
    print("words_not_in_vocab:", words_not_in_vocab)
    print("words_map:", words_map)
    return JsonResponse({'suggested_words' : suggested_words,
    						'words_not_in_vocab' : words_not_in_vocab,
    						'words_map': words_map,
    						'clue_nouns': clue_nouns}, safe=False)
