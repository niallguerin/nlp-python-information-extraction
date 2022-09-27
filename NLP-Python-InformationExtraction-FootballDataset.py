#!/usr/bin/env python
# coding: utf-8

# # Information Extraction - Assignment
# This assignment is based on the Information Extraction lecture and the lab.
#

# Name: Niall Guerin
# Student ID: 18235079

# In[2]:


import json
import nltk
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# mount google drive file system
# from google.colab import drive
# drive.mount('/content/gdrive')
from statistics import mode
from nltk.corpus import ieer

# In[4]:


inputfile = 'football_players.txt'  # Location of the file
buf = open(inputfile, encoding="UTF-8")
list_of_doc = buf.read().split('\n')

# validate the read
print(list_of_doc[0])


# # Task 1 (10 Marks)
# Write a function that takes each document and performs:
# 1) sentence segmentation 2) tokenization 3) part-of-speech tagging
#
# Please keep in mind that the expected output is a list within a list as shown below.
#

# In[5]:


def ie_preprocess(document):
    # each document = new player profile
    sentences = nltk.sent_tokenize(document)  # step 1
    sentences = [nltk.word_tokenize(sent) for sent in sentences]  # step 2
    pos_sentences = [nltk.pos_tag(sent) for sent in sentences]  # step 3

    # returning a player profile object with POS for each list item of original list
    return pos_sentences


def ie_filter_empty_documents(document_list):
    # check for empty list elements and delete
    filtered_document_list = list(filter(None, document_list))

    return filtered_document_list


# print(list_of_doc[0])
list_of_doc = ie_filter_empty_documents(list_of_doc)
first_doc = list_of_doc[0]
pos_sent = ie_preprocess(first_doc)

# validate part of speech tagging
print(pos_sent)

# Run the following code to check your result for the first document (Ronaldo).

# In[6]:


first_doc = list_of_doc[0]
pos_sent = ie_preprocess(first_doc)
pos_sent


# *Expected* output
#  [...[('He', 'PRP'),
#   ('is', 'VBZ'),
#   ('a', 'DT'),
#   ('forward', 'NN'),
#   ('and', 'CC'),
#   ('serves', 'NNS'),
#   ('as', 'IN'),
#   ('captain', 'NN'),
#   ('for', 'IN'),
#   ('Portugal', 'NNP'),
#   ('.', '.')], ...]

# # Task 2 (20 Marks)
# Write a function that will take the list of tokens with POS tags for each sentence and returns the named entities (NE).
#
# Hint: Use binary=True while calling NE chunk function

# In[7]:


def named_entity_finding(pos_sent):
    tree = nltk.ne_chunk(pos_sent, binary=True)
    named_entities = []

    for subtree in tree.subtrees():
        if subtree.label() == 'NE':
            entity = ""
            for leaf in subtree.leaves():
                entity = entity + leaf[0] + " "
            named_entities.append(entity.strip())

    return named_entities


def create_pos_tagged_sent_list(doc_list):
    pos_sents_list = []

    for idx, document in enumerate(doc_list):
        pos_temp = ie_preprocess(doc_list[idx])
        pos_sents_list.append(pos_temp)

    return pos_sents_list


# do NE task for single document unit test
# pos_sents=ie_preprocess(list_of_doc[0])
# named_entity_finding(pos_sents[0])

# do NE task for all documents unit test: task 1 used a single document index reference for the ie_preprocess function call. based on wording I created a list here
# in case it is required for the question so that the full list of segmented, tokenized, and POS-tagged sentences are stored in single container for all documents
# as later sections indicate we might need to reference this overall structure and not just the single profile documents.
pos_sents_container = create_pos_tagged_sent_list(list_of_doc)
named_entity_finding(pos_sents_container[0][0])


# Expected output ['Cristiano Ronaldo',
#  'Santos Aveiro',
#  'ComM',
#  'GOIH',
#  'Portuguese',
#  'Portuguese',
#  'Spanish',
#  'Real Madrid',
#  'Portugal']

# # Task 3 (10 Marks)
#
# Now use the named_entity_finding() function to extract all NEs for each document.
#
# Hint: pos_sents holds the list of lists of tokens with POS tags

# In[8]:


def NE_flat_list_fn(pos_sents):
    NE = []
    for pos_sent in pos_sents:
        # Single line code here. Call the funtion named_entity_finding(pos_sent) and
        # append the result to the NE list
        NE.append(named_entity_finding(pos_sent))
    # Single line code here. Flatten the list of lists to the single list NE_flat_list
    NE_flat_list = [item for sublist in NE for item in sublist]
    return NE_flat_list


# unit test: task 3 - single document
# pos_sents = ie_preprocess(list_of_doc[0])
# flat_list = NE_flat_list_fn(pos_sents)
# print(flat_list)

# unit test: task 3 - all player profile documents
named_entities = []
for idx, document in enumerate(list_of_doc):
    pos_sents = ie_preprocess(list_of_doc[idx])
    flat_list = NE_flat_list_fn(pos_sents)
    named_entities.append(flat_list)

for ne in named_entities:
    print(ne)


# # Task 4 (40 Marks)
#
# Write functions to extract the name of the player, country of origin and date of birth as well as the following relations: team(s) of the player and position(s) of the player.
#
# Hint: Use the re.compile() function to create the extraction patterns
#
# Reference: https://docs.python.org/3/howto/regex.html

# In[9]:


def name_of_the_player(doc):
    # I initially used the NE find function i.e. based on extracting regex pattern, I then ran it, and concatenated valid NE find results from that matched string
    # I removed it as found I was writing less with below pattern although in hindsight this pattern match while was shorter is potentially fragile (if some new pattern found in 20 new player documents for example) but so far in all unit tests it holds up so I left as is and avoided rolling back
    name = ""

    # I worked literally off the regex document link, labs on regex patterns, and stack overflow/regex tutorial sites for patterns: see bibliography
    # rule: match everything in each document up to the initial parenthesis based on corpus
    NAME_MATCH_PATTERN_1 = re.compile(r'^[^\(]+')

    # rule: check for existence of a comma in the result string: not all corpus documents are the same
    NAME_MATCH_PATTERN_2 = re.compile(r'[,]')

    # rule: extract only the string up to the comma instance
    NAME_MATCH_PATTERN_3 = re.compile(r'^[^\,]+')

    # pattern 1: search for anything up to parenthesis only and ignore rest
    pattern1 = NAME_MATCH_PATTERN_1.search(doc).group(0)
    # pattern 2: check the result of pattern 1 for any comma instances
    if pattern1 is not None:
        pattern2 = NAME_MATCH_PATTERN_1.search(pattern1).group(0)
    else:
        pattern1 = "Could not find the name of the player from this document."

    # if pattern 2/comma pattern is None exit and return name otherwise, extract string up to the first comma only and ignore rest
    if pattern2 is not None:
        pattern3 = NAME_MATCH_PATTERN_3.search(pattern2).group(0)
        name = pattern3
    else:
        name = pattern1

    return name


def country_of_origin(doc):
    country = ""

    # country of origin patterns: based on human review of corpus this is a dictionary lookup-list based on all available documents
    COUNTRY_ORIGIN_DICT = {"Argentine": "Argentina", "Brazilian": "Brazil", "English": "England", "German": "Germany",
                           "Portuguese": "Portugal", "Welsh": "Wales", "Spanish": "Spain", "Swedish": "Sweden"}

    # rule: search for the word preceding the occurrence of word professional in the corpus
    COUNTRY_PATTERN1 = re.compile(r'\w+(?= +professional\b)')
    COUNTRY_PATTERN2 = re.compile(r'\w+(?= +former\b)')
    COUNTRY_PATTERN3 = re.compile(r'\w+(?= +footballer\b)')

    # main search patterns: storing as list so we can amend or access as required for rule matchers in if decision tests
    patterns = ["former", "footballer"]

    # default search patterns:
    # do not do a group position extract until search result is confirmed as otherwise assignment will be null if search returns nothing
    search_pattern1 = COUNTRY_PATTERN1.search(doc)

    # regex pattern search either matches default pattern1, fails and matches pattern2, or has no match and we look up pattern3, or we find nothing with these rules.
    if search_pattern1 is not None and search_pattern1.group(0) != patterns[0]:
        country = COUNTRY_ORIGIN_DICT.get(search_pattern1.group(0))
    elif search_pattern1 is not None and search_pattern1.group(0) == patterns[0]:
        search_pattern2 = COUNTRY_PATTERN2.search(doc)
        country = COUNTRY_ORIGIN_DICT.get(search_pattern2.group(0))
    elif search_pattern1 is None:
        search_pattern3 = COUNTRY_PATTERN3.search(doc)
        country = COUNTRY_ORIGIN_DICT.get(search_pattern3.group(0))
    else:
        country = "No country of origin could be determined for this player."

    return country


def date_of_birth(doc):
    date = ""
    # rule: match born string and up to parenthesis only
    DOB_PATTERN = re.compile(r'(?<=born )[^)]*')

    # do not do a group position extract until search result is confirmed as otherwise assignment will be null if search returns nothing
    pattern = DOB_PATTERN.search(doc)

    if pattern is not None:
        date = pattern.group(0)
    else:
        date = "No date of birth found for this player."

    return date


def team_of_the_player(doc):
    # construct list based on human review of the corpus: derived from check of total corpus documents and output of task 3 showing named entity
    # originally the list was larger with key-value dict in form of team:CLUB, team: NATIONAL and search would only extract types = CLUB if a match
    clubs = ["Arsenal", "Atlético Mineiro", "Barcelona", "Everton", "Flamengo", "Grêmio", "Juventus", "LA Galaxy",
             "Manchester United", "Milan", "Paris Saint-Germain", "Preston North End", "Querétaro", "Real Madrid",
             "Santos", "Schalke", "Southampton" "Tottenham Hotspur", "Werder Bremen"]

    # this time we reverse the flow: the search is done against our humand reviewed list and checked against the input document
    # use a set so no duplicates are allowed. regex search will do a finall against the clubs list and if a match we add to their teams
    # unit test notes: Santos is included in a case where it should not be included for Ronaldo in document 1 so stricter regex pattern versus this list would resolve it i.e annotate document with NE PERSON versus NE ORGANISATION or custom like CLUB
    team = set()
    pattern = []
    for idx, club in enumerate(clubs):
        pattern = re.findall(clubs[idx], doc, flags=0)
        if len(pattern) != 0:
            # extract only the first pattern as the rest are just repetition finds of same pattern
            pattern = pattern[0]
            team.add(pattern)

    return team


def position_of_the_player(doc):
    #     initial search patterns in regex to test finding of strings based on position
    #     further testing was done on patterns using plays as a[position], as a[position] per types of multiple OR rules here: https://stackoverflow.com/questions/4389644/regex-to-match-string-containing-two-names-in-any-order
    #     due to errors I backed them out to save time and go with position-based list below and reverse search against the list with re
    #     POS_PATTERN1 = re.compile(r'(\bplays as a forward)')
    #     POS_PATTERN2 = re.compile(r'(\bis a forward)')
    #     POS_PATTERN3 = re.compile(r'(\ba forward)')
    #     POS_PATTERN4 = re.compile(r'(\bplays as a striker)')
    #     POS_PATTERN5 = re.compile(r'(\bas an attacking midfielder)')
    #     POS_PATTERN6 = re.compile(r'(\bmidfield roles)')
    #     POS_PATTERN7 = re.compile(r'(\bas a right winger)')
    #     POS_PATTERN8 = re.compile(r'(\bas a winger)')
    #     POS_PATTERN9 = re.compile(r'(\bplays as a central midfielder)')

    # players normally can can play in one or more positions as mostly strikers maintain same position: midfielders could be attacking, switch to wingers and so on
    # based on above single unit test regex patterns using same list construction used for teams as I would have a list of all outfield playing positions we search against
    positions = ["forward", "striker", "attacking midfielder", "midfield", "midfielder", "central midfielder",
                 "defensive midfielder", "right winger", "left winger", "winger", "sweeper", "left back", "right back",
                 "defender", "goalie", "goalkeeper"]
    pos = []
    pattern = []
    for idx, p in enumerate(positions):
        pattern = re.findall(positions[idx], doc, flags=0)
        if len(pattern) != 0:
            pattern = pattern[0]
            pos.append(pattern)

            # in testing most are variations of same one for given player: take first only as JSON section at end is looking for single not multiple. Misinterpreted this for team and position originally
    # both this and teams are pretty crude brute force searches and brute force extractions with high degree of information loss: after sketching this a few times, my preference was a custom defined gazeteer as I had tested constructing the trees myself on a single document from the corpus which did work and then using the lab nltk relations extractor on it
    # I originally had all multiple positions allowed e.g. Gareth Bale was moved from defense to other positions and the above captured that. After doing task 5, I switched this to a brute extract of a single position match to match json output single value requirements
    position = pos[
        0]  # added only to ensure single position otherwise a list of positions is generated which is valid for some players (for others, a rule could be extended that if 'forward' in all 3 results, take just the one list index value at start position as here)

    return position


# Task a: Print player name: must include full name not partial per discussion forum advice to students from tutors
# use list_of_doc_[2]: verified in additional unit tests for all documents in corpus to confirm full name is extracted in each test case
profile = list_of_doc[2]
player_name = name_of_the_player(profile)
print("The name of the player is:", player_name)

# Task b: Print player country of origin
player_country = country_of_origin(profile)
print("The country of origin of", player_name, "is:", player_country)

# Task c: Print player date of birth: verified for all corpus players
player_dob = date_of_birth(profile)
print("The date of birth of", player_name, "is:", player_dob)

# Task d: Print all player clubs, not national team
# construct list based on human review of the corpus
player_team = team_of_the_player(profile)
print("The teams that", player_name, "has played with are:", player_team)

# Task e: Print player current playing position
player_pos = position_of_the_player(profile)
print(player_name, "position(s):", player_pos)

# Execute the below command to check your fuction
#

# In[10]:


date_of_birth(list_of_doc[2])


# Expected output '5 February 1992'

# # Task 5 (10 Marks)
#
# Write a function using the outputs from the previous functions to generate JSON-LD output as follows.
#
# Reference: https://json-ld.org/primer/latest/
#
# { "@id": "http://my-soccer-ontology.com/footballer/name_of_the_player",
#
#     "name": "",
#     "born": "",
#     "country": "",
#     "position": [
#         { "@id": "http://my-soccer-ontology.com/position",
#             "type": ""
#         }
#      ]
#      "team": [
#         { "@id": "http://my-soccer-ontology.com/team",
#             "name": ""
#         }
#      ]
# }
#

# In[11]:


def generate_jsonld(jsonargs):
    # record length of jsonargs so extension in task6 does not break function in task4
    input_length = len(jsonargs)

    # json schema key-value templates
    id = "@id"
    id_schema_root = "http://my-soccer-ontology.com/footballer/name_of_the_player"
    id_schema_position = "http://my-soccer-ontology.com/position"
    id_schema_team = "http://my-soccer-ontology.com/team"
    id_schema_awards = "http://my-soccer-ontology.com/awards"

    # json keys
    type_key = "type"
    name_key = "name"
    born_key = "born"
    country_key = "country"
    position_key = "position"
    team_key = "team"
    award_key = "awards"

    # initialize the json values based in function input parameters
    # player_json is the root json object
    player_json = {}
    player_json[id] = id_schema_root
    player_json[name_key] = jsonargs[0]
    player_json[born_key] = jsonargs[1]
    player_json[country_key] = jsonargs[2]

    # pos_json is a nested json array of single object (can add more) for player positions
    pos_json = {}
    pos_json[id] = id_schema_position
    pos_json[type_key] = jsonargs[3]
    # need a container otherwise we only add a json object itself, not the array for 1 or many json objects per json template requirement
    pos_container = []
    pos_container.append(pos_json)
    # now assign the array of json object(s) to the main player_json position key
    player_json[position_key] = pos_container

    # team_list converts my set of teams to a list: I should really only have a single team here. I misinterpreted this originally and added ALL teams after I saw array of objects in task 5
    team_list = list(jsonargs[4])
    team_list_joined = ', '.join(map(str, team_list))

    # team_json works exactly like pos_json
    team_json = {}
    team_json[id] = id_schema_team
    team_json[name_key] = team_list_joined  # add the csv-delimited string of teams from prior list conversion
    # need a container here again as otherwise we only hold json object itself, not an array of json objects per original failed unit test
    team_container = []
    team_container.append(team_json)
    # now assign the array of json object(s) to the main player_json team key
    player_json[team_key] = team_container

    # awards_json: add condition so task4 continues to work
    if (input_length > 5):
        award_list = list(jsonargs[5])
        award_list_joined = ', '.join(map(str, award_list))
        # award_json works exactly like team_json
        award_json = {}
        award_json[id] = id_schema_awards
        award_json[name_key] = award_list_joined  # add the csv-delimited string of teams from prior list conversion
        # need a container here again as otherwise we only hold json object itself, not an array of json objects per original failed unit test
        award_container = []
        award_container.append(award_json)
        player_json[award_key] = award_container

    # main json generation function for the json_player raw structure: NB: per unit test utf-8 bug configure 'ensure_ascii = False' to avoid mangled characters in display of json results
    # Web Reference: https://stackoverflow.com/questions/18337407/saving-utf-8-texts-in-json-dumps-as-utf8-not-as-u-escape-sequence
    json_player = json.dumps(player_json, ensure_ascii=False)

    return json_player


# create arg list for json function: update profile list_of_doc[x] value to test for each player profile in corpus
profile = list_of_doc[2]
player_name = name_of_the_player(profile)
player_country = country_of_origin(profile)
player_dob = date_of_birth(profile)
player_pos = position_of_the_player(profile)
player_team = team_of_the_player(profile)

# list for json function arguments
json_args = []
json_args.append(player_name)
json_args.append(player_dob)
json_args.append(player_country)
json_args.append(player_pos)
json_args.append(player_team)
# print(json_args) # uncomment to validate the json args are valid

# unit test the json generator function for json-ld format output: validate in json lint to confirm valid json returned for all player document profiles
jsonld_player = generate_jsonld(json_args)
print(jsonld_player)


# open bugs based on unit tests for all players in the corpus
# 001: Santos returned for list_of_doc[0] as a team. incorrect as part of his name not his club (which is Brazilian): requires stricter regex as even with a NE match it would still strip out Santos as a named entity. Using both regex up to x point and what's before only and then doing NE find would likely prevent it.
# 002: Some of the players show a backlash so probably something related to fix I added for ensure_ascii which fixed most encoding issues. They should still be escaped as required to output full name


# # Task 6 (10 Marks)
# Identify one other relation (besides team and player) and write a function to extract this. Also extend the JSON-LD output accordingly.

# In[12]:


def awards_for_player(doc):
    awards = set()
    # rule: search for initial pattern after first word only and up to end word only: list of main fifa and other player association awards based on review of corpus
    AWARD_PATTERN1 = re.compile(r'(?s)FIFA World Player.*?Year')
    AWARD_PATTERN2 = re.compile(r'(?s)Ballon.*?Or')
    AWARD_PATTERN3 = re.compile(r'(?s)FIFA Puskás.*?Award')
    AWARD_PATTERN4 = re.compile(r'(?s)PFA Players.*?Year')
    AWARD_PATTERN5 = re.compile(r'(?s)PFA Young.*?Year')
    AWARD_PATTERN6 = re.compile(r'(?s)FWA.*?Year')
    AWARD_PATTERN7 = re.compile(r'(?s)Player of.*?Tournament')
    AWARD_PATTERN8 = re.compile(r'(?s)ranked first.*?assists')

    # do not do a group position extract until search result is confirmed as otherwise assignment will be null if search returns nothing
    pattern1 = AWARD_PATTERN1.search(doc)
    pattern2 = AWARD_PATTERN2.search(doc)
    pattern3 = AWARD_PATTERN3.search(doc)
    pattern4 = AWARD_PATTERN4.search(doc)
    pattern5 = AWARD_PATTERN5.search(doc)
    pattern6 = AWARD_PATTERN6.search(doc)
    pattern7 = AWARD_PATTERN7.search(doc)
    pattern8 = AWARD_PATTERN8.search(doc)

    # now check the search patterns: if a match, add it as an award, else nothing
    if pattern1 is not None:
        awards.add(pattern1.group(0))

    if pattern2 is not None:
        awards.add(pattern2.group(0))

    if pattern3 is not None:
        awards.add(pattern3.group(0))

    if pattern4 is not None:
        awards.add(pattern4.group(0))

    if pattern5 is not None:
        awards.add(pattern5.group(0))

    if pattern6 is not None:
        awards.add(pattern6.group(0))

    if pattern7 is not None:
        awards.add(pattern7.group(0))

    if pattern8 is not None:
        awards.add(pattern8.group(0))

    return awards


# unit test for awards
profile = list_of_doc[9]
awards = awards_for_player(profile)
# print(awards)

# unit test jsonld for awards input parameter
player_name = name_of_the_player(profile)
player_country = country_of_origin(profile)
player_dob = date_of_birth(profile)
player_pos = position_of_the_player(profile)
player_team = team_of_the_player(profile)
player_awards = awards_for_player(profile)

json_args_ext = []
json_args_ext.append(player_name)
json_args_ext.append(player_dob)
json_args_ext.append(player_country)
json_args_ext.append(player_pos)
json_args_ext.append(player_team)
json_args_ext.append(player_awards)
# print(json_args_ext)

# unit test for jsonld extended function: full player object with awards (if any)
json_extension = generate_jsonld(json_args_ext)
print(json_extension)

# **Bibliography**
# 1) NLP Labs 7, 8, 9 exercises
# 2) NLP Information Extraction 1 and Information Extraction II lecture materials
# 3) Regex patterns:
# https://stackoverflow.com/questions/13867860/match-everything-until-parenthesis
# https://stackoverflow.com/questions/2372573/how-do-i-remove-whitespace-from-the-end-of-a-string-in-python
# https://stackoverflow.com/questions/15340582/python-extract-pattern-matches
# https://stackoverflow.com/questions/13867860/match-everything-until-parenthesis
# https://stackoverflow.com/questions/12148784/extract-text-before-first-comma-with-regex
# https://stackoverflow.com/questions/5327206/regex-match-digits-comma-and-semicolon
# https://stackoverflow.com/questions/5457416/regex-only-numbers-and-dot-or-comma
# https://stackoverflow.com/questions/30326562/regular-expression-match-everything-after-a-particular-word
# http://www.nltk.org/howto/stem.html (was testing porter stemmer initially as was considering stemmer to before using key-value for country/country association)
# https://spacy.io/usage/linguistic-features#named-entities (interesting link from general NE research during assignment task 4)
# https://stackoverflow.com/questions/546220/how-to-match-the-first-word-after-an-expression-with-regex
# https://stackoverflow.com/questions/4389644/regex-to-match-string-containing-two-names-in-any-order
# https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
#  https://docs.python.org/3/tutorial/datastructures.html
#  https://stackoverflow.com/questions/2554185/match-groups-in-python
# https://www.tutorialspoint.com/python/python_reg_expressions.htm
# https://stackoverflow.com/questions/30150047/find-all-locations-cities-places-in-a-text
# https://stackoverflow.com/questions/2974022/is-it-possible-to-assign-the-same-value-to-multiple-keys-in-a-dict-object-at-onc
#
# 4) NLTK
#  https://www.nltk.org/book/ch07.html
# http://www.nltk.org/howto/relextract.html
# https://www.nltk.org/_modules/nltk/sem/relextract.html (not used in end)
# https://stackoverflow.com/questions/49387699/extracting-the-person-names-in-the-named-entity-recognition-in-nlp-using-python (again when initially attempting my own annotation of corpus and tree construction)
# https://stackoverflow.com/questions/7851937/extract-relationships-using-nltk (when constructing own single unit test for one document)
#
# 5) Python and JSON
# https://stackoverflow.com/questions/12453580/concatenate-item-in-list-to-strings
# https://stackoverflow.com/questions/22496596/how-does-one-insert-a-key-value-pair-into-a-python-list
# https://stackoverflow.com/questions/22496596/how-does-one-insert-a-key-value-pair-into-a-python-list
# https://stackoverflow.com/questions/11197818/how-do-i-make-a-json-object-with-multiple-arrays
#
# 6) Regex and final awards pattern research
# https://stackoverflow.com/questions/6109882/regex-match-all-characters-between-two-strings
# https://stackoverflow.com/questions/4419000/regex-match-everything-after-question-mark
# https://www.rexegg.com/regex-quickstart.html
