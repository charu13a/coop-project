from gensim.models import KeyedVectors
import logging
from itertools import chain
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob
import nltk
nltk.download('punkt')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
stemmer = SnowballStemmer("english")

EMBEDDING_FILE = './GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

model.similar_by_word('test', topn=1, restrict_vocab=None)

def suggest_words(words_list, clues_list, request):
  clue_nouns = list(map(extract_nouns, clues_list))
  clues_words_combined = []
  filtered_words_not_in_vocab = []
  for x in range(len(words_list)):
  	if len(clue_nouns) > 0:
  		clue_answers = [words_list[x]] + clue_nouns[x]
  	else:
  		clue_answers = [words_list[x]]
  	y = filter_words_not_in_vocab(model,clue_answers)
  	filtered_words_not_in_vocab =  filtered_words_not_in_vocab + list(set(clue_answers) - set(y))
  	if len(y) > 0:
  		clues_words_combined.append(y)
  # replace each word list by most similar word
  closest_words = []
  for x in clues_words_combined:
  	words = []
  	for key, value in request.session.items():
  		if(value == x):
  			print("Found closest_words:", key)
  			words = [key[:-5]]
  			break
  	if words == []:
  		words = model.most_similar(positive=x, topn=1)
  		words = [x[0] for x in words]
  		request.session[words[0]+'10001'] = x
  	closest_words += words

  flattened_clue_list = [item for sublist in clue_nouns for item in sublist]
  added_words_list = words_list + flattened_clue_list

  print("closest_words:", closest_words)
  [result_words, map_words] = find_highest_frequency(model, closest_words, request, nwords=40)
  result_words = list(filter(filter_words_with_same_root(added_words_list), result_words))
  print("result_words:", result_words)
  print("map_words:", map_words)
  # remove potential duplicates
  final_words = []
  final_words_map = {}
  for word in result_words:
    is_similar = False
    for x in final_words:
      if stemmer.stem(x) == stemmer.stem(word) or x in word or word in x:
        is_similar = True
        break
    if not is_similar:
      final_words.append(word.replace('_', ' '))
      for tup in map_words:
      	if tup[1][0] == word:
      		if word in final_words_map:
      			#if [tup[0],tup[1][1], closest_words.index(tup[0])]  not in final_words_map[word]:
      			final_words_map[word].append([tup[0], tup[1][1], closest_words.index(tup[0])])
      		else:
      			final_words_map[word] = [[tup[0], tup[1][1], closest_words.index(tup[0])]]
  return [final_words, filtered_words_not_in_vocab, final_words_map, clue_nouns]

def suggest_words_new(words_list, clues_list, request):
  clue_nouns = list(map(extract_nouns, clues_list))
  flattened_clue_list = [item for sublist in clue_nouns for item in sublist]
  added_words_list = words_list + flattened_clue_list
  final_words_list = filter_words_not_in_vocab(model, added_words_list)
  filtered_words_not_in_vocab = list(set(added_words_list) - set(final_words_list))
  [result_words, map_words] = find_highest_frequency(model, final_words_list, request, nwords=50)
  result_words = list(filter(filter_words_with_same_root(added_words_list), result_words))
  print("result_words:", result_words)
  # remove potential duplicates
  final_words = []
  final_words_map = {}
  for word in result_words:
    is_similar = False
    for x in final_words:
      if stemmer.stem(x) == stemmer.stem(word) or x in word or word in x:
        is_similar = True
        break
    if not is_similar:
      final_words.append(word.replace('_', ' '))
      for tup in map_words:
      	if tup[1][0] == word:
      		if word in final_words_map:
      			#if [tup[0],tup[1][1]]  not in final_words_map[word]:
      			final_words_map[word].append([tup[0], tup[1][1]])
      		else:
      			final_words_map[word] = [[tup[0], tup[1][1]]]
  return [final_words, filtered_words_not_in_vocab, final_words_map, clue_nouns]

def suggest_words_new2(words_list, clues_list, negative_list, request):
  clue_nouns = list(map(extract_nouns, clues_list))
  flattened_clue_list = [item for sublist in clue_nouns for item in sublist]
  added_words_list = words_list + flattened_clue_list
  final_words_list = filter_words_not_in_vocab(model, added_words_list)
  filtered_words_not_in_vocab = list(set(added_words_list) - set(final_words_list))
  result_words = model.most_similar(positive=final_words_list, topn=50)
  result_words = [x[0] for x in result_words]
  map_words = final_words_list
  result_words = list(filter(filter_words_with_same_root(added_words_list), result_words))
  print("result_words:", result_words)
  # remove potential duplicates
  final_words = []
  final_words_map = {}
  for word in result_words:
    is_similar = False
    for x in final_words:
      if stemmer.stem(x) == stemmer.stem(word) or x in word or word in x:
        is_similar = True
        break
    if not is_similar:
      final_words.append(word.replace('_', ' '))
      for tup in map_words:
      	if tup[1][0] == word:
      		if word in final_words_map:
      			#if [tup[0],tup[1][1]]  not in final_words_map[word]:
      			final_words_map[word].append([tup[0], tup[1][1]])
      		else:
      			final_words_map[word] = [[tup[0], tup[1][1]]]
  return [final_words, filtered_words_not_in_vocab, final_words_map, clue_nouns]

# filters words not in the model vocabulary
def filter_words_not_in_vocab(model, list_of_words):
  word_vectors = model.wv
  return list(filter(lambda x: x in word_vectors.vocab, list_of_words))

def filter_words_with_same_root(input_words):
  def should_filter(word):
    for x in input_words:
      if(stemmer.stem(x) == stemmer.stem(word) or x in word or word in x):
        return False
    return True
  return should_filter

def find_highest_frequency(model, list_of_words, request, nwords=20):
  closest_words = []
  map_words = []
  for word in list_of_words:
  	  if word in request.session:
  	  	dist_words = request.session[word]
  	  	words = [x[0] for x in dist_words]
  	  	print("Cached word:", word)
  	  else:
  	  	dist_words = model.similar_by_word(word, topn=50, restrict_vocab=None)
  	  	words = [x[0] for x in dist_words]
  	  	request.session[word] = dist_words
  	  for y in dist_words:
  	  	map_words.append([word, y])
  	  closest_words = closest_words + words
  freq_count = Counter(chain(closest_words)).most_common(nwords)
  print("freq_count:", freq_count)
  return [[x[0] for x in freq_count], map_words]

def extract_nouns(txt):
  return [w for (w, pos) in TextBlob(txt).pos_tags if pos[0] == 'N']