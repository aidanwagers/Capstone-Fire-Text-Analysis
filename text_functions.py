#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import re
from collections import Counter, defaultdict
from string import punctuation

import janitor
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import spacy
import sqlite3
from nltk.corpus import stopwords
from pprint import pprint
from textblob import TextBlob

# Download necessary nltk dataset
nltk.download('punkt')

# Stopwords for English
sw = stopwords.words('english')

# TextBlob initialization
tb = TextBlob('')


# In[10]:

# List of fire specific stop words for removal
firestops = ["fire", "area", "within", "concerns", "miles", "values", 
             "planning","high", "resources", "risk", "impacts", "creek", 
             "located", "sites", "natural", "proximity", "resource", 
             "include", "north", "south", "east", "west", "including", "due", 
             "areas", "area", "low", "lands", "threatened", "may", "along", 
             "multiple", "impact", "impacted", "burn", "burning", "river", "threat",
             "lake", "concern", "moderate", "road", "highway", "adjacent", "current", 
             "currently", "also", "near", "potential", "several", "adjacent", "land", 
             "critical", "concern", "significant", "location", "expected", "fires", 
             "approximately", "perimeter", "local", "forest", "numerous", "national", 
             "exist", "exists", "well",]


# In[11]:
def categorize_columns(df):
    """
    Categorize columns of a DataFrame based on certain keywords in their names.
    Used for creating notes, decisions, and identifiers lists.

    Parameters:
        df (DataFrame): The DataFrame to categorize columns.

    Returns:
        tuple: A tuple containing lists of column names categorized based on keywords.
            The lists include columns containing 'notes', 'desc', and '_id'.
    """
    notes = [column for column in df.columns if 'notes' in column]
    descs = [column for column in df.columns if 'desc' in column]
    ids = [column for column in df.columns if '_id' in column]
    
    return notes, descs, ids

def set_columns_to_string(df, notes):
    """
    Convert specified columns in a DataFrame to string data type.
    Used to ensure all notes fields are strings.

    Parameters:
        df (DataFrame): The DataFrame to convert columns.
        column_list (list): A list of column names to be converted to string data type.

    Returns:
        DataFrame: The DataFrame with specified columns converted to string data type.
    """
    
    for column_name in column_lis|t:
        df[column_name] = df[column_name].astype(str)
    return df

def get_last_value(col):
    """
    Get the last (previous) value of a column.

    Parameters:
        col (column): The column from which to get the last value.

    Returns:
        object: The last value of the column.
    """
    unique_values = col.unique()
    if len(unique_values) > 1:
        return col.iloc[-1]
    else:
        return unique_values[0]

# In[12]:


def process_text_column(df, column_name):
    """
    Process a text column in a DataFrame to extract unique tokens using NLTK's PunktSentenceTokenizer.

    Parameters:
        df (pandas DataFrame): The DataFrame containing the text column.
        column_name (str): The name of the text column to process.

    Returns:
        unique_tokens: A list of unique tokens extracted from the text column.
    """
    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    
    unique_tokens = []
    for i in range(len(df)):
        text = df.iloc[i][column_name]
        if not pd.isna(text):
            tokens = tokenizer.tokenize(text)
            for t in tokens:
                if t not in unique_tokens:
                    unique_tokens.append(t)
    return unique_tokens


# In[13]:

# List of common misspellings generated early in the process
misspellings = {
    "locateded": "located",
    "jurisdicstions": "jurisdictions",
    "extrem": "extreme",
    "aprox": "approximate",
    "elivation": "elevation",
    "dynaminc": "dynamic",
    "barrieres": "barriers",
    "preddiced" : "predicted",
    "aaround" : "around",
    "abililty" : "ability",
    "constuct" : "construct",
    "consultaion" : "consultation",
    "containd" : "contained",
    "hellicopter'" : "helicopter",
    'organiztional': 'organizational',
    'organzation': 'organization',
    'orgin': 'origin',
    'orgination': 'organization',
    'orginazatin': 'organization',
    'orgizational': 'organizational',
    'organizaiton': 'organization',
    'organizatinal': 'organizational',
    'organizationally': 'organizationally',
    'organizattional': 'organizational',
    "lightining": "lightning",
    "challanges": "challenges",
    "frenchglen": "french glen",
    "accross": "across",
    "implmentation": "implementation",
    "influnces": "influences",
    "lilley": "lily",
    "beow": "below",
    "employeed": "employed",
    "archaological": "archaeological",
    "readaly" : "readily",
    "expectred" : "expected",
    "committment" : "commitment",
    "intesity" : "intensity",
    "avaialble" : "available",
    "capabilites" : "capabilities",
    "inicluding" : "including",
    "infomation" : "information",
    'concersn': 'concern',
    'ammountn': 'amount',
    'communciaiton': 'communication',
    'howver': 'however',
    'expectied': 'expected',
    'implemenatation': 'implementation',
    'accuarte': 'accurate',
    'communciation': 'communication',
    'instablility': 'instability',
    'approximastely': 'approximately',
    'insolation': 'isolation',
    'beprimarily': 'be primarily',
    'evlevation': 'elevation',
    'expecet': 'expect',
    'approximatel': 'approximately',
    'forecastws': 'forecasts',
    'expectted': 'expected',
    "inlcuding": "including",
    "behaiver": "behavior",
    "jursidictional": "jurisdictional",
    "inadvertantly": "inadvertently",
    "appropiate": "appropriate",
    "jursidictions": "jurisdictions",
    "enviroment": "environment",
    "decadant": "decadent",
    "landwoners": "landowners",
    "iperation": "operation",
    "behaivor": "behavior",
    "invlove": "involve",
    "exept": "except",
    "exposre": "expose",
    "inbriefing": "in briefing",
    "implemenation": "implementation",
    "appromximately": "approximately",
    "evaucations": "evacuations",
    "infrastrucures": "infrastructures",
    "liekly": "likely",
    "burnning": "burning",
    "habitiat": "habitat",
    "critial": "critical",
    "intrastructure": "infrastructure",
    "intereest": "interest",
    "cunducive": "conducive",
    "firebehavior": "fire behavior",
    "burningseveral": "burning several",
    "infrastrucutre": "infrastructure",
    "becasue": "because",
    "desicion": "decision",
    "concidered": "considered",
    "infrasture": "infrastructure",
    "higer": "higher",
    "guage": "gauge",
    "evauctions": "evacuations",
    "burningi": "burning",
    "habtat": "habitat",
    "inuring": "injuring",
    "draininges": "drainages",
    "becaus": "because",
    "infuence": "influence",
    "adviser": "advisor",
    "idnentfied": "identified",
    "lmited": "limited",
    "identfied": "identified",
    "infristructure": "infrastructure",
    "idicies": "indices",
    "infuences" : "influences",
    "firday": "friday",
    "hiehgt": "height",
    "aroud": "around",
    "grwoth": "growth",
    "escalante": "escalate",
    "drainanges": "drainages",
    "drainge": "drainage",
    "criticial": "critical",
    "drough": "drought",
    "intermitten": "intermittent",
    "handfull": "handful",
    "liklihood": "likelihood",
    "irefigherts": "firefighters",
    "conditioon": "condition",
    "heightend": "heightened",
    "aformentioned" : "aforementioned",
    "logisitical": "logistical",
    "consumnes": "consumes",
    "logisticall": "logistical",
    "funcitioning": "functioning",
    "logisticts" : "logistics"
 }

# Dict of even more misspellings.
# This was a very fun process!
# It is noteworthy that this dict is not in comliance with the PEP 8
# maximum line length. Apologies.
badspell = {'hwy': 'highway', 'raods': 'roads', 'availble': 'available', 'helo': 'helicopter', 'warrent': 'warrant', 'approximatley': 'approximately', 'liason': 'liaison', 'potiental': 'potential', 'adjactent': 'adjacent', 'implimentation': 'implementation', 'competeing': 'competing', 'heavey': 'heavy', 'lacation': 'location', 'valuse': 'value', 'calfire': 'campfire', 'lnf': 'of', 'andf': 'and', 'woody': 'wood', 'inholdings': 'holdings', 'varibility': 'visibility', 'btu': 'but', 'dpa': 'pa', 'mitigage': 'mitigate', 'regualarly': 'regularly', 'dependant': 'dependent', 'mediium': 'medium', 'scappy': 'sappy', 'transitition': 'transition', 'represntative': 'representative', 'currenlty': 'currently', 'persistant': 'persistent', 'likey': 'likely', 'mtn': 'mountainn', 'approx': 'approximately', 'similiar': 'similar', 'infastructure': 'infrastructure', 'moderatley': 'moderately', 'intell': 'intel', 'difficuly': 'difficult', 'cooperators': 'operators', 'tactice': 'tactics', 'frid': 'grid', 'assesment': 'assessment', 'realtively': 'relatively', 'compressor': 'compressor', 'realatively': 'relatively', 'occurance': 'occurrence', 'activly': 'actively', 'burnied': 'burned', 'infranstructure': 'infrastructure', 'attainable': 'obtainable', 'curently': 'currently', 'personel': 'personnel', 'valles': 'valleys', 'difficulity': 'difficulty', 'complexites': 'complexities', 'oranizational': 'organizational', 'adacent': 'adjacent', 'rmainder': 'remainder', 'localized': 'localised', 'dificulties': 'difficulties', 'recreactional': 'recreational', 'ownserships': 'ownership', 'awarness': 'awareness', 'precip': 'precipitation', 'concumed': 'consumed', 'oeration': 'operation', 'continueous': 'continuous', 'managment': 'management', 'mitigatin': 'mitigating', 'apropriate': 'appropriate', 'resourcess': 'resources', 'troughout': 'throughout', 'moisure': 'moisture', 'discontinous': 'discontinuous', 'controling': 'controlling', 'boundry': 'boundary', 'accessable': 'accessible', 'continutiy': 'continuity', 'riskl': 'risk', 'agressive': 'aggressive', 'becuase': 'because', 'jurisdication': 'jurisdiction', 'interal': 'internal', 'hightened': 'heightened', 'potentail': 'potential', 'thier': 'their', 'baer': 'bear', 'occured': 'occurred', 'nlimited': 'unlimited', 'contiued': 'continued', 'throught': 'thought', 'goverment': 'government', 'propery': 'property', 'estern': 'eastern', 'linke': 'like', 'rouck': 'rock', 'unprecidented': 'unprecedented', 'forcast': 'forecast', 'interally': 'internally', 'reacdhes': 'reaches',
'fuesl': 'fuel', 'currenly': 'currently', 'barries': 'barrier', 'curretnly': 'currently', 'threatning': 'threatening', 'infrastucture': 'infrastructure', 'dificulty': 'difficulty', 'unaccessable': 'inaccessible', 'intertest': 'interest', 'getz': 'get', 'geogrphic': 'geographic', 'threatend': 'threatened', 'realtive': 'relative', 'concernd': 'concerned', 'availible': 'available', 'successfuly': 'successfully', 'activites': 'activities', 'recoverys': 'recovery', 'sould': 'would', 'threatnening': 'threatening', 'prefered': 'preferred', 'responsibilites': 'responsibilities', 'familar': 'familiar', 'histroic': 'historic', 'numerours': 'numerous', 'visable': 'visible', 'miminal': 'minimal', 'consistant': 'consistent', 'atttack': 'attack', 'initail': 'initial', 'successfull': 'successful', 'posesses': 'possesses', 'mimimal': 'minimal', 'mutiple': 'multiple', 'avaition': 'aviation', 'eradic': 'erattic', 'concentrations': 'concentration', 'sparce': 'sparse', 'abote': 'above', 'contiuity': 'continuity', 'alanta': 'atlanta', 'noticable': 'noticeable', 'manangment': 'management', 'inital': 'initial', 'relativly': 'relatively', 'boardered': 'bordered', 'thefire': 'the fire', 'mediun': 'medium', 'signifigant': 'significant', 'knose': 'knows', 'conterversy': 'controversy', 'nedia': 'media', 'pertains': 'certains', 'seral': 'serial', 'preperations': 'preparations', 'threatned': 'threatened', 'implentation': 'implementaition', 'manmade': 'manage', 'diffucult': 'difficult', 'sesason': 'season', 'exibiting': 'exhibiting', 'conituing': 'continuing', 'adjacen': 'adjacent', 'detrmines': 'determines', 'resoucres': 'resources', 'curretly': 'currently', 'progran': 'program', 'veresed': 'versed', 'neeeds': 'needs', 'contiune': 'continue', 'toching': 'touching', 'denses': 'dense', 'meduim': 'medium', 'experince': 'experience', 'sensistive': 'sensitive', 'identied': 'identified', 'operatiojns': 'operations', 'engb': 'end', 'firb': 'fire', 'distrubance': 'disturbance', 'mitiagte': 'mitigate', 'begining': 'beginning', 'smke': 'smoke', 'weater': 'weather', 'suppresion': 'suppression', 'dryer': 'drier', 'gridded': 'ridden', 'signficant': 'significant', 'polictical': 'political', 'juridiction': 'jurisdiction', 'perdicted': 'predicted', 'increaces': 'increases', 'slighly': 'slightly', 'mewet': 'meet', 'thre': 'three', 'likelyhood': 'likelihood', 'funtional': 'functional', 'approsimately': 'approximately', 'confin': 'confine', 'bolders': 'boulders', 'miostures': 'moisture', 'oganizational': 'organizational', 'governent': 'government', 'pygmy': 'pigmy', 'previoius': 'previous', 'signficance': 'significance', 'durn': 'turn', 'condtions': 'conditions', 'dept': 'department', 'droughty': 'drought', 'comm': 'communicate', 'mtns': 'mountains', 'buring': 'burning', 'terran': 'terrain', 'gullied': 'gullies', 'propability': 'probability', 'porr': 'poor', 'durring': 'during', 'intial': 'initial', 'medivac': 'medevac', 'competative': 'competitive', 'thos': 'this', 'possiblity': 'possibility', 'difficultly': 'difficulty', 'probabilty': 'probability', 'inaccessable': 'inaccessible', 'proxity': 'proximity', 'trhough': 'through', 'hyw': 'highway', 'typw': 'type', 'theese': 'these', 'areal': 'area', 'parimeter': 'parameter', 'mositure': 'moisture', 'currentlly': 'currently', 'comensurate': 'commensurate', 'monson': 'monsoon', 'condon': 'condone', 'resouces': 'resources', 'influeneces': 'influences', 'satus': 'status', 'numberous': 'numerous', 'boundries': 'boundaries', 'beging': 'begging', 'minimial': 'minimal', 'likley': 'likely', 'consuluting': 'consulting', 'supprsession': 'suppression', 'buckey': 'buckeye', 'rive': 'river', 'speard': 'speared', 'inferface': 'interface', 'primiarily': 'primarily', 'marial': 'martial', 'neccessary': 'necessary', 'conditon': 'condition', 'maintenace': 'maintenance', 'personell': 'personnel', 'sucessfully': 'successfully', 'plannning': 'planning', 'somes': 'some', 'intermittenly': 'intermittently', 'pearch': 'perch', 'modelling': 'modeling', 'acerage': 'acreage', 'seasoanl': 'seasonal', 'spead': 'speed', 'conitinues': 'continues', 'intital': 'initial', 'folowed': 'followed', 'acheive': 'achieve', 'redution': 'reduction', 'attact': 'attack', 'communites': 'communities', 'representitives': 'representatives', 'proccesses': 'processes', 'mulitple': 'multiple', 'sturctures': 'structures', 'difficutly': 'difficulty', 'recieved': 'received', 'substatial': 'substantial', 'burnin': 'burning', 'inacessibility': 'inaccessibility', 'sloped': 'slope', 'probablity': 'probability', 'reduceing': 'reducing', 'dectecion': 'detection', 'dispursed': 'dispersed', 'resoruces': 'resources', 'direciton': 'direction', 'potenital': 'potential', 'approching': 'approaching', 'adminstrative': 'administrative', 'organzization': 'organization', 'strucure': 'structure', 'deadicated': 'dedicated', 'reosources': 'resources', 'franch': 'french', 'locateed': 'located', 'curent': 'current', 'exihbited': 'exhibited', 'potentiall': 'potential', 'probablilty': 'probability', 'probabily': 'probably', 'cability': 'ability', 'notifications': 'modifications', 'vegitation': 'vegetation', 'additonal': 'additional', 'westerly': 'western', 'initiates': 'initiated', 'initital': 'initial', 'resoucces': 'resources', 'numerious': 'numerous', 'discontinuos': 'discontinuous', 'accesss': 'access', 'chosing': 'choosing', 'incicies': 'indicies', 'incies': 'indicies', 'crossess': 'crosses', 'strucuture': 'structure', 'aligns': 'aliens', 'crtitical': 'critical', 'revaluated': 'reevaluated', 'im': 'in', 'durning': 'during', 'indistrial': 'industrial', 'boundariy': 'boundary', 'althought': 'although', 'contaiment': 'containment', 'muncipal': 'municipal', 'posibility': 'possibility', 'significanly': 'significantly', 'surronding': 'surrounding', 'savannahs': 'savannah', 'errtic': 'erratic', 'woner': 'wonder', 'succussfully': 'successfully', 'controled': 'controlled', 'maanaging': 'managing', 'tomarrow': 'tomorrow', 'probabllity': 'probability', 'condictions': 'conditions', 'pulbic': 'public', 'condiions': 'conditions', 'managaged': 'managed', 'hundered': 'hundred', 'provice': 'province', 'appearence': 'appearance', 'preceptions': 'perceptions', 'tes': 'yes', 'archeological': 'archaeological', 'infulences': 'influences', 'summitt': 'summit', 'mangement': 'management', 'consitst': 'consist', 'occaisonal': 'occasional', 'seperation': 'separation', 'objectes': 'objects', 'owernship': 'ownership', 'supression': 'suppression', 'organziation': 'organization', 'expereinced': 'experienced', 'adajcent': 'adjacent', 'orgnanizational': 'organizational', 'trrain': 'terrain', 'fishook': 'fishhook', 'occurr': 'occur', 'haulted': 'halted', 'retardent': 'retardant', 'potencial': 'potential', 'evacutation': 'evacuation', 'barrers': 'barriers', 'smoke was': 'smoke was', 'focusing': 'rousing', 'depature': 'departure', 'aggresive': 'aggressive', 'proxcimity': 'proximity', 'significan': 'significant', 'behavoir': 'behavior', 'horst': 'horse', 'temparatures': 'temperatures', 'rivewr': 'river', 'tiem': 'time', 'particuraly': 'particularly', 'continures': 'continues', 'officiall': 'official', 'corrridor': 'corridor', 'seperated': 'separated', 'visibilty': 'visibility', 'additinal': 'additional', 'manangement': 'management', 'probablility': 'probability', 'visitiation': 'visitation', 'infrastrucre': 'infrastructure', 'avilability': 'availability', 'transfered': 'transferred', 'improvments': 'improvements', 'manegemnt': 'management', 'starteies' : 'strategies', 'managemnt': 'management', 'somewaht': 'somewhat', 'succesful': 'successful', 'speading': 'spreading', 'poetntial': 'potential', 'futrue': 'future', 'crwoning': 'crowning', 'communicatin': 'communication', 'conifir': 'conifer', 'hazrds': 'hazards', 'suppresson': 'suppression', 'orginization': 'organization', 'implimintation': 'implementation',
'typicall': 'typical', 'continious': 'continuous', 'wtih': 'with', 'forcasts': 'forecasts', 'adventagious': 'advantageous', 'valueable': 'valuable', 'stucture': 'structure', 'occure': 'occur', 'wthin': 'within', 'resouce': 'resource', 'futher': 'further', 'appoximately': 'approximately', 'econimic': 'economic', 'aviaiton': 'aviation', 'proximtiy': 'proximity', 'risidents': 'residents', 'prixmity': 'proximity', 'conerns': 'concerns', 'teh': 'the', 'fuctional': 'functional', 'organiation': 'organization', 'recieve': 'receive', 'dimished': 'diminished', 'aera': 'area', 'restortion': 'restoration', 'histoic': 'historic', 'leopord': 'leopard', 'barrriers': 'barriers', 'familair': 'familiar', 'availablity': 'availability', 'rtrained': 'retrained', 'threating': 'threatening', 'extremee': 'extreme', 'desne': 'dense', 'proximety': 'proximity', 'probibility': 'probability', 'polulation': 'population', 'predominantely': 'predominantly', 'witnin': 'within', 'cultrural': 'cultural', 'repetative': 'repetitive', 'clamer': 'clamor', 'acheived': 'achieved', 'thru': 'thou', 'thie': 'the', 'mocing': 'moving', 'fouth': 'south', 'protion': 'portion', 'cooridor': 'corridor', 'outreah': 'outreach', 'sherriff': 'sheriff', 'miosture': 'moisture', 'structues': 'structures', 'moders': 'modern', 'distirct': 'distinct', 'supprestion': 'suppression', 'anticiapate': 'anticipate', 'tourisum': 'tourism', 'potentioal': 'potential', 'barriors': 'barriers', 'reigion': 'region', 'aveaging': 'averaging', 'hazzards': 'hazards', 'dedicatied': 'dedicated', 'influances': 'influences', 'retrdant': 'retardant', 'utillized': 'utilized', 'cooridors': 'corridors', 'assests': 'assets', 'lave': 'have', 'theatening': 'threatening', 'erradic': 'erratic', 'relience': 'reliance', 'infrasturcture': 'infrastructure', 'reservior': 'reservoir', 'funcitons': 'functions', 'saftey': 'safety', 'natrual': 'natural', 'dissapointment': 'disappointment', 'overalll': 'overall', 'mechnical': 'mechanical', 'precentile': 'percentile', 'resistrictions': 'restrictions', 'temporay': 'temporary', 'opertations': 'operations', 'resistence': 'resistance', 'signifcant': 'significant', 'behaver': 'behavior', 'engauged': 'engaged', 'developement': 'development', 'addional': 'additional', 'ammount': 'amount', 'limmited': 'limited', 'occassional': 'occasional', 'regardind': 'regarding', 'straighforward': 'straightforward', 'adject': 'abject', 'postive': 'positive', 'concenr': 'concern', 'senstive': 'sensitive', 'juristiction': 'jurisdiction', 'discontiuous': 'discontinuous', 'intrest': 'interest', 'undurned': 'unturned', 'terrian': 'terrain', 'stratgies': 'strategies', 'unactive': 'inactive', 'adits': 'admits', 'unsucessful': 'unsuccessful', 'movementto': 'movement to', 'comittment': 'commitment', 'ordef': 'order', 'brigde': 'bridge', 'firest': 'fires', 'artisit': 'artist', 'recommeded': 'recommended', 'expereince': 'experience','ciritical': 'critical','imediate': 'immediate', 'assments': 'assesments', 'importatant': 'important', 'dueo': 'due', 'twisp': 'twist', 'provate': 'private', 'unusal': 'unusual', 'methow': 'method', 'currrently': 'currently', 'tacticts': 'tactics', 'avilable': 'available', 'klone': 'clone', 'tyee': 'type', 'entiat': 'entire', 'mgmt': 'management', 'threatinging': 'threatening', 'extreame': 'extreme', 'increas': 'increase', 'mostl': 'most', 'hystorical': 'historical', 'planing': 'planting', 'offire': 'office', 'linehas': 'linehands', 'intellegence': 'intelligence', 'adequatly': 'adequately', 'hamperedby': 'hampered', 'dampen': 'damper', 'adeuqaute': 'adequate', 'extern': 'external', 'cemetary': 'cemetery', 'avaialbility': 'availability', 'conditons': 'conditions', 'bilk': 'bill', 'corrider': 'corridor', 'duation': 'duration', 'reccommended': 'recommended', 'undoubtably': 'undoubtedly', 'severly': 'severely', 'histroric': 'historic', 'gorwing': 'growing', 'compexity': 'complexity', 'potenial': 'potential', 'primarly': 'primary', 'implmented': 'implemented', 'proximetly': 'proximity', 'moblization': 'mobilization', 'midddle': 'middle', 'administrated': 'administrated', 'responsability': 'responsibility', 'detering': 'deterring', 'relyed': 'relied', 'previouse': 'previous', 'reources': 'resources', 'adminitered': 'administered', 'fue': 'fuel', 'veg': 'vegitation', 'inlcude': 'include', 'notheast': 'northeast', 'adequat': 'adequate', 'impeads': 'impedes','earlt': 'early', 'logitical': 'logistical', 'runing': 'running', 'naural': 'natural', 'wiht': 'with','merced': 'merged', 'trimble': 'tremble', 'scatted': 'scatter', 'benificial': 'beneficial','benifit': 'benefit', 'dispered': 'dispersed','opportunites': 'opportunities', 'approximely': 'approximately','mefe': 'mere', 'reasses': 'reassess', 'reccomends': 'recommends', 'othere': 'other', 'immmediately': 'immediately', 'iven': 'even', 'remanins': 'remains', 'ocurring': 'occurring', 'spase': 'space', 'managemet': 'management', 'informaiton': 'information', 'exremely': 'extremely', 'lilve': 'live', 'asthe': 'as the', 'duraiton': 'duration', 'srpead': 'spread', 'concer': 'concur', 'conderns': 'concerns', 'percnet': 'percent', 'ttowards': 'towards', 'dependancy': 'dependency', 'corriders': 'corridors', 'contol': 'control', 'groth': 'growth', 'contorl': 'control', 'siginificant': 'significant', 'windier': 'winter', 'therefor': 'therefore', 'perenial': 'peroneal', 'inaccesssable': 'inaccessible', 'responsibities': 'responsibilities', 'immediatly': 'immediately', 'lcoal': 'local', 'withdrawel': 'withdrawal', 'regrouping': 'grouping', 'inaccesibility': 'inaccessibility', 'acessing': 'accessing', 'assinged': 'assigned', 'seasaonal': 'seasonal', 'adjoing': 'adjoining', 'restricitons': 'restrictions', 'nene': 'none', 'mulitiple': 'multiple', 'uncontolled': 'uncontrolled', 'sint': 'sent', 'issuess': 'issues', 'owenership': 'ownership', 'addded': 'added', 'contiue': 'continue', 'poil': 'soil', 'stragey': 'strategy', 'proably': 'probably', 'priortization': 'privatization', 'allign': 'align', 'heigth': 'height', 'exteme': 'extreme', 'opportunties': 'opportunities', 'strom': 'storm', 'wildnerness': 'wilderness', 'cummulative': 'cumulative', 'transportion': 'transportation', 'inhubit': 'inhibit', 'beinf': 'being', 'sitution': 'situation', 'minimumx': 'minimum', 'grider': 'rider', 'numeorus': 'numerous', 'auguest': 'august', 'seiad': 'said', 'communties': 'communities', 'widespead': 'widespread', 'concers': 'concerns', 'growith': 'growth', 'wil': 'will', 'milne': 'mine', 'annchor': 'anchor', 'allotements': 'allotments', 'larg': 'large', 'memeber': 'member', 'thouroughtly': 'thoroughly', 'archaelogical': 'archaeological', 'evacuaitons': 'evacuation','nort': 'north', 'patroled': 'patrolled', 'temperatrues': 'temperatures', 'platts': 'plates', 'wll': 'all', 'withing': 'within', 'steepest': 'deepest', 'wich': 'which', 'extreemly': 'extremely', 'issuse': 'issue', 'concren': 'concern', 'decions': 'decisions', 'rockey': 'rocky', 'opperations': 'operations', 'oppertunity': 'opportunity', 'overngiht': 'overnight', 'promitory': 'promissory', 'posses': 'possess', 'recommened': 'recommended', 'steen': 'seen', 'temp': 'temperature', 'utlizing': 'utilizing', 'properites': 'properties', 'grantie': 'granite', 'plumers': 'plumes', 'fules': 'fuels', 'foresta': 'forests', 'exisiting': 'existing', 'controll': 'control', 'curiousity': 'curiosity', 'greatley': 'greatly', 'diminshed': 'diminished', 'bodie': 'bodies', 'afer': 'after', 'swall': 'shall', 'abut': 'about', 'intertie': 'inertia', 'lignite': 'ignite', 'yuch': 'such', 'beign': 'being', 'potentiial': 'potential', 'barriars': 'barriers', 'withion': 'within', 'peat': 'seat', 'egde': 'edge', 'coas': 'coast', 'regen': 'regenerate', 'reletively': 'relatively', 'minimual': 'minimal', 'mor': 'for', 'smokeis': 'smokies', 'higth': 'high', 'militrary': 'military', 'sytems': 'systems', 'impavct': 'impact', 'personnell': 'personnel', 'availabliity': 'availability', 'searson': 'season', 'particulate': 'particular', 'manageing': 'managing','funtions': 'functions', 'mcrd': 'more', 'receiveing': 'receiving', 'politaical': 'political', 'standinng': 'standing', 'canoue': 'canoe', 'staats': 'starts', 'cancelled': 'canceled', 'humiditys': 'humidity', 'taketna': 'taken', 'hellicopter': 'helicopter', 'protoocls': 'protocols', 'tho': 'the', 'esspecially': 'especially', 'lmoderate': 'moderate', 'complecated': 'complicated', 'communit': 'community', 'deficets': 'defects', 'llittle': 'little', 'deterimental': 'detrimental', 'stuctures': 'structures', 'condisered': 'considered', 'funcutional': 'functional', 'minamul': 'minimum', 'regaurds': 'regards', 'oderate': 'moderate', 'tupper': 'upper', 'damped': 'dampened', 'csss': 'cases', 'withy': 'with', 'contianed': 'contained', 'benefical': 'beneficial', 'holden': 'holder', 'intensty': 'intensity', 'loaction': 'location', 'sliver': 'silver', 'boarder': 'border', 'heli': 'helicopter', 'throughs': 'through', 'coporation': 'cooperation', 'grasg': 'grass', 'publi': 'public', 'exclusiveley': 'exclusively', 'liklely': 'likely', 'densly': 'densely', 'erefuge': 'refuge', 'enp': 'end', 'terraine': 'terrain', 'ojectives': 'objectives', 'sructures': 'structures', 'ares': 'are', 'assesement': 'assessment', 'untypically': 'atypically', 'cascadel': 'cascade', 'misquite': 'mesquite', 'fule': 'fuel', 'mostures': 'moistures', 'avearge': 'average', 'utilyzing': 'utilizing', 'lasing': 'losing', 'begininng': 'beginning', 'wtihin': 'within', 'aqatics': 'aquatics', 'vicinty': 'vicinity', 'duzen': 'dozen', 'norte': 'north', 'mainstem': 'mainstream', 'particularily': 'particularly', 'balde': 'bald', 'cee': 'see', 'rescores': 'resources', 'truely': 'truly', 'alene': 'alone', 'aeneas': 'areas', 'tel': 'tell', 'swann': 'swan', 'coridoor': 'corridor', 'activi': 'active', 'stratigies': 'strategics', 'berray': 'betray', 'swick': 'sick', 'realtionships': 'relationships', 'oppurtunites': 'opportunities', 'assoicated': 'associated', 'sparsly': 'sparsely', 'exsist': 'exist', 'imediately':'immediately',
           'assetts': 'assets', 'normals': 'normal', 'aggrevated': 'aggravated', 'chineese': 'chinese', 'anticipateit': 'anticipate it', 'witin': 'within', 'acheological': 'archeological', 'siller': 'silver', 'secaped': 'escaped', 'solider': 'solid', 'proxamity': 'proximity', 'communiction': 'communication', 'curren': 'current', 'benefitting': 'benefiting', 'measureable': 'measurable', 'lemmon': 'lemon','nateral': 'natural','significants': 'significance', 'dificult': 'difficult', 'eastof': 'east', 'pubic': 'public', 'structuures': 'structures', 'antone': 'anyone', 'towads': 'towards', 'erractic': 'erratic','lighnting': 'lightning','severit': 'severity','lessor': 'lesser', 'threated': 'threatened', 'primarlily': 'primarily', 'tormorrow': 'tomorrow', 'eractic': 'erratic', 'extremly': 'extremely', 'straydog': 'stray dog', 'issuses': 'issues', 'primative': 'primitive', 'opportuiites': 'opportunities', 'acticity': 'activity', 'managament': 'management', 'conatined': 'contained', 'moisures': 'moistures', 'mles': 'miles', 'reprod': 'reproduce', 'sround': 'around', 'owerships': 'ownership', 'currnetly': 'currently', 'biggets': 'biggest', 'hootest': 'hottest', 'expexted': 'expected','occurrance': 'occurrence', 'arguement': 'argument', 'negativley': 'negatively','nusiance': 'nuisance', 'handleing': 'handling', 'suppresssed': 'suppressed', 'divison': 'division', 'sensitve': 'sensitive', 'jursidiction': 'jurisdiction', 'necessiatates': 'necessitates', 'potion': 'portion', 'effectivley': 'effectively', 'highwway': 'highway', 'bounday': 'boundary', 'benefitted': 'benefited', 'deminish': 'diminish', 'opposie': 'opposite', 'assessmnets': 'assessment', 'imprementation': 'implementation', 'prescibed': 'prescribed', 'cotinue': 'continue', 'tempertures': 'temperatures', 'unbered': 'entered', 'fron': 'from', 'effectivily': 'effectively', 'developped': 'developed', 'premptive': 'primitive', 'inclued': 'included', 'whch': 'which', 'religous': 'religious', 'curltural': 'cultural', 'reltive': 'relative', 'reflectred': 'reflected', 'deamed': 'deemed', 'determend': 'determined', 'duntil': 'until', 'remotness': 'remoteness', 'assesed': 'assessed', 'wold': 'would','altogtehr': 'altogether', 'elveation': 'elevation', 'intersate': 'interstate', 'trigging': 'rigging', 'impactive': 'impactful', 'imber': 'timber', 'fuctions': 'functions', 'incresed': 'increased', 'sseverity': 'severity', 'neww': 'new', 'comprimised': 'compromised', 'proximty': 'proximity', 'resourse': 'resource', 'redued': 'reduced', 'northeat': 'northeast', 'redforn': 'reform', 'johhny': 'johnny', 'geroge': 'george', 'northest': 'northeast', 'fromthe': 'from the','ofprivate': 'of private', 'avaiation': 'aviation', 'occurances': 'occurrences', 'precipt': 'precipitation','acrres': 'acres', 'buisness': 'business', 'seperate': 'separate', 'fedrally': 'federally', 'approximatly': 'approximately', 'conserns': 'concerns', 'otal': 'total', 'allision': 'allusion', 'misc': 'miscellaneous', 'entrancr': 'entrance', 'tansfer': 'transfer', 'bevahior': 'behavior', 'cumatively': 'cumulatively', 'immenint': 'imminent', 'availabilty': 'availability', 'scatered': 'scattered', 'proximae': 'proximate', 'vihicle': 'vehicle', 'undburned': 'unburned','systsem': 'system', 'bahavior': 'behavior', 'recored': 'recorded', 'infracstructure': 'infrastructure', 'instated': 'instead', 'untill': 'until', 'polution': 'pollution', 'moderat': 'moderate', 'comitted': 'committed', 'availbility': 'availability', 'tatics': 'tactics', 'smodering': 'smoldering', 'seasonaly': 'seasonally', 'minimze': 'minimize', 'direst': 'direct', 'montitor': 'monitor', 'particulary': 'particularly', 'activitiy': 'activity', 'stradegy': 'strategy', 'cotrolled': 'controlled', 'utilzed': 'utilized','extiguishes': 'extinguishes', 'cumlative': 'cumulative', 'routt': 'route', 'thurday': 'thursday', 'realy': 'really', 'ajacent': 'adjacent', 'resourses': 'resources', 'remaiing': 'remaining', 'statigic': 'strategic', 'immeditaly': 'immediately', 'diffuculty': 'difficulty','stategy': 'strategy', 'mitiate': 'mitigate','succeptible': 'susceptible', 'strucutre': 'structure', 'piont': 'point', 'priroirty': 'priority', 'prioirty': 'priority', 'controlle': 'controller', 'archeolgical': 'archaeological', 'closesness': 'closeness', 'coperators': 'cooperators', 'barrires': 'barriers', 'aremote': 'a remote', 'obseverd': 'observed', 'comminities': 'communities', 'obeectives': 'objectives', 'imped': 'limped', 'reamainder': 'remainder', 'boundaires': 'boundaries', 'infrastuctures': 'infrastructure', 'immanent': 'imminent', 'objecties': 'objectives', 'jursdiction': 'jurisdiction', 'burner': 'burned', 'undesired': 'desired', 'withiing': 'within', 'govenor': 'governor', 'strices': 'strikes', 'priorty': 'priority', 'impactng': 'impacted', 'miless': 'miles', 'desireable': 'desirable', 'empede': 'impede', 'possibl': 'possible', 'seaasonal': 'seasonal', 'tstorms': 'storms', 'intially': 'initially', 'adequete': 'adequate', 'limitted': 'limited', 'consiste': 'consists', 'nsoon': 'soon', 'threatended': 'threatened', 'widerness': 'wilderness', 'tospread': 'spread', 'progessing': 'progressing', 'probabilites': 'probabilities', 'lassen': 'lessen', 'thoughout': 'throughout', 'higway': 'highway', 'seasonl': 'seasonal', 'grainery': 'granary', 'contrained': 'contained', 'rual': 'rural', 'commiting': 'committing', 'inaccesible': 'inaccessible', 'prgnaization': 'organization','intemittant': 'intermittent', 'explainations': 'explanations', 'reccomended': 'recommended', 'animas': 'animal', 'recieves': 'receives', 'therre': 'there', 'fallowed': 'followed', 'objectivies': 'objectives', 'resourced': 'resources', 'apprised': 'appraised', 'activies': 'activity', 'thorougfare': 'thoroughfare', 'vecinity': 'vicinity', 'mosoon': 'monsoon', 'significately': 'significantly', 'significate': 'significance', 'suffient': 'sufficient', 'prehistorical': 'prehistoric', 'realitively': 'relatively', 'fuell': 'fuel', 'tactis': 'tactics', 'modeate': 'moderate', 'visinity': 'vicinity', 'policitcal': 'political', 'moutain': 'mountain', 'apprise': 'appraise', 'intermittant': 'intermittent', 'politcal': 'political', 'possibilty': 'possibility', 'rivate': 'private', 'unlikey': 'unlikely', 'escaper': 'escape', 'serval': 'several', 'strucutures': 'structures', 'entercom': 'intercom', 'lfuel': 'fuel', 'growt': 'growth', 'factilities': 'facilities', 'livr': 'live', 'straitforward': 'straightforward', 'relitive': 'relative', 'potintial': 'potential', 'renforces': 'reinforces', 'fier': 'fire', 'burring': 'burning', 'poplulated': 'populated', 'hav': 'have', 'unlikley': 'unlikely', 'strructures': 'structures', 'tthe': 'the', 'othe': 'other', 'unsuppressed': 'suppressed', 'pressence': 'presence', 'maintenence': 'maintenance', 'mosture': 'moisture', 'properity': 'property', 'opproaches': 'approaches', 'repotred': 'reported', 'continuos': 'continuous', 'notifiing': 'notifying', 'storey': 'story', 'lelevation': 'elevation', 'exerice': 'exercise', 'commision': 'commission', 'overal': 'overall', 'dens': 'dense', 'manaagement': 'management', 'moitor': 'monitor', 'larios': 'various', 'wisherd': 'wished', 'perosnnel': 'personnel', 'sesonal': 'seasonal', 'cultral': 'cultural', 'quailty': 'quality', 'brcomes': 'becomes', 'occouring': 'occuring', 'bottem': 'bottom', 'raido': 'raidio', 'extreem': 'extreme', 'competiton': 'competition', 'tiome': 'time', 'rodside': 'roadside', 'coopative': 'cooperative', 'mires': 'miles', 'anlysis': 'analysis', 'exsposed': 'exposed', 'beave': 'beaver', 'logicical': 'logical', 'conjuction': 'conjunction', 'indcident': 'incident', 'wildersness': 'wilderness', 'origion': 'origin', 'escalte': 'escalate', 'lessoned': 'lessened', 'lowere': 'lower', 'vegitative': 'vegetative', 'sturcture': 'structure', 'twards': 'towards', 'historc': 'history', 'esily': 'easily', 'developent': 'development', 'sverity': 'severity', 'sginificant': 'significant', 'rockerfeller': 'rockefeller', 'compelted': 'compelled', 'inaccessability': 'inaccessibility', 'routte': 'route', 'reintroduce': 'introduce', 'splits': 'splints', 'kortes': 'cortes', 'petersen': 'peterson', 'pahaska': 'alaska', 'continuosly': 'continuously', 'potnetial': 'potential', 'corrently': 'currently', 'adjecent': 'adjacent', 'cuirrent': 'current', 'tempuratures': 'temperatures', 'moonson': 'monsoon', 'aproximately': 'approximately', 'thunderstoms': 'thunderstorms', 'manged': 'managed', 'nataional': 'national', 'lengthes': 'lengths', 'invovled': 'involved', 'behvavior': 'behavior', 'meetin': 'meeting', 'gaging': 'gauging', 'mositures': 'moisture', 'interrest': 'interest', 'caching': 'cacheing', 'orgainization': 'organization', 'assement': 'assesment', 'approximalty': 'approximately', 'signifcantly': 'significantly', 'invovlement': 'involvement', 'contintue': 'continue', 'boudary': 'boundary', 'moisturers': 'moisture', 'terain': 'terrain', 'rolloing': 'rolling', 'paneled': 'panelled', 'inaccessbile': 'inaccessible', 'tthere': 'there', 'transiting': 'transition', 'expsore': 'explore', 'outhside': 'southside', 'orgainziation': 'organization', 'usuall': 'usually', 'terrine': 'terraine', 'visition': 'visitation', 'activily': 'activity', 'shold': 'should', 'figher': 'fighter', 'ecpected': 'expected', 'moistrures': 'moisture', 'invintoried': 'inventoried', 'expreienced': 'experienced', 'significatly': 'significantly', 'highd': 'high', 'asigned': 'assigned', 'frezing': 'freezing', 'activited': 'activities', 'suffiecient': 'sufficient', 'calibrated': 'celebrated', 'mnt': 'met', 'rmimt': 'mist', 'tracker': 'trace', 'supai': 'spain', 'aztca': 'attica', 'swasey': 'swayed', 'condusice': 'condusive', 'clsoe': 'close', 'nevaa': 'nevada', 'thiry': 'third', 'hisorically': 'historically', 'bunred': 'burned', 'organiaztion': 'organization', 'orgainizational': 'organizational',
            'orgainzation': 'organization', 'occurng': 'occuring', 'labeled': 'labelled', 'estimeate': 'estimate', 'conceans': 'concerns', 'iout': 'out', 'galss': 'glass', 'quailified': 'qualified', 'adivsor': 'advisor', 'terranin': 'terrain', 'minmal': 'minimal', 'ight': 'right', 'severley': 'severely', 'prosses': 'process', 'montata': 'montana', 'areass': 'areas',  'stratigic': 'strategic', 'suppresss': 'suppress', 'lenths': 'lengths', 'durration': 'duration', 'ncident': 'incident', 'limted': 'limited', 'laterial': 'material', 'seperating': 'separating', 'thorugh': 'thorough', 'expierince': 'experience', 'severaty': 'severity', 'exteranl': 'external', 'functioan': 'function', 'reseptive': 'receptive', 'uper': 'upper', 'visability': 'visibility', 'nightime': 'nighttime', 'feul': 'fuel', 'conopy': 'canopy', 'preventative': 'preventive', 'probab': 'probably','recieving': 'receiving','cayon': 'canyon', 'apprx': 'approx', 'sulfer': 'sulfur', 'potentital': 'potential', 'shitiike': 'shitike', 'beaty': 'beauty', 'percintile': 'percentile', 'hiway': 'highway', 'probabililty': 'probability', 'eding': 'ending', 'deptartment': 'department', 'withhin': 'within', 'exhibted': 'exhibited', 'protecct': 'protect', 'likliehood': 'likelihood', 'orgaization': 'organization', 'implament': 'implement', 'contnue': 'continue', 'astringent': 'stringent', 'cascasde': 'cascade', 'averge': 'average', 'potentiallly': 'potentially', 'musick': 'music', 'experianced': 'experienced', 'organzational': 'organizational', 'countyr': 'country', 'proir': 'prior', 'mutliple': 'multiple', 'norther': 'northern', 'olene': 'olean', 'delated': 'related', 'extracation': 'extraction', 'utilizied': 'utilized', 'plannaing': 'planning','complexitiy': 'complexity', 'downd': 'downed', 'reconnaisance': 'reconnaissance', 'synomous': 'synonymous', 'ilustrate': 'illustrate', 'chsracterized': 'characterized', 'chage': 'change', 'permieter': 'perimeter','nusuiance': 'nuisance', 'previos': 'previous', 'similer': 'similar', 'wpould': 'would', 'ther': 'the', 'inthe': 'the', 'wildernes': 'wilderness', 'mangaement': 'management', 'implementations': 'implementation', 'limitied': 'limited', 'gravellys': 'gravelly', 'hinters': 'hunters','operationaly': 'operational', 'juristictions': 'jurisdiction', 'approximetly': 'approximately', 'cultrual': 'cultural', 'arent': 'are not', 'plannining': 'planning', 'actionis': 'actions', 'strucures': 'structures', 'choosen': 'chosen', 'directy': 'direct', 'significnat': 'significant', 'adequateor': 'adequate', 'burnde': 'burned', 'counrty': 'country', 'availibility': 'availability', 'searsk': 'sears', 'willb': 'will', 'esuccessful': 'successful', 'htis': 'this', 'acess': 'access', 'acitive': 'active', 'efferts': 'efforts', 'inflences': 'influences', 'moinsture': 'moisture', 'relitively': 'relatively','parshall': 'partial', 'grandby': 'grandly', 'vrain': 'brain','poetential': 'potential', 'potentil': 'potential', 'substantialy': 'substantial', 'redetermination': 'determination', 'todate': 'to date','setember': 'september', 'oving': 'moving',  'holston': 'houston', 'responsibilty': 'responsibility', 'dependance': 'dependence', 'componemnts': 'components', 'averagre': 'average', 'chaimberlain': 'chamberlain', 'ancticipated': 'anticipated', 'imact': 'impact', 'managememt': 'management', 'concrens': 'concerns', 'burend': 'burned', 'stragegies': 'stratetegies', 'strategi': 'strategic', 'pivate': 'private', 'receeding': 'receding', 'aproaching': 'approaching', 'proximately': 'approximately', 'precent': 'present', 'burntside': 'burnt side', 'treatened': 'threatened', 'homogenous': 'homogeneous', 'corrior': 'corridor', 'firre': 'fire', 'xcel': 'excel', 'manely': 'mainly', 'interring': 'entering', 'tranee': 'trainee', 'severeity': 'severity', 'contious': 'conscious', 'avalnche': 'avalanche', 'impovements': 'improvements', 'onthe': 'on the', 'classifed': 'classified', 'availalbe': 'available', 'overestimate': 'overestimated', 'misture': 'mixture','argicultual': 'agricultural', 'structuresw': 'structures', 'sensativity': 'sensitivity','preforming': 'performing', 'dought': 'drought', 'ablility': 'ability', 'xpected': 'expected', 'extream': 'extreme', 'therse': 'there', 'bering': 'being', 'investsms': 'investors', 'routs': 'routes', 'lotts': 'lots', 'operationa': 'operation', 'culutral': 'cultural', 'throughfare': 'thoroughfare', 'currnt': 'current', 'climatic': 'climactic', 'alothough': 'although', 'vacinity': 'vicinity', 'availaibility': 'availability', 'unusally': 'unusually', 'cureent': 'current', 'divde': 'divide', 'acitvities': 'activities', 'shite': 'white', 'confire': 'confirm', 'hasnt': 'has not', 'moverating': 'moderating', 'sprerad': 'spread', 'hud': 'had', 'specialis': 'specialist', 'contiues': 'continues', 'loggin': 'logging', 'evern': 'even', 'addiitonal': 'additional','storrie': 'stories', 'wexpected': 'we expected', 'coodination': 'coordination', 'sucessful': 'successful', 'previious': 'previous', 'inyou': 'in you', 'embeded': 'embedded', 'reletive': 'relative', 'solstace': 'solstice', 'wrangel': 'wrangle', 'supresion': 'suppression','straegy': 'strategy', 'fores': 'forest', 'litlle': 'little', 'featurs': 'features', 'stafffed': 'staffed', 'proximitiy': 'proximity', 'natio': 'nation', 'roadl': 'road', 'lire': 'like', 'tennent': 'tenant', 'unease': 'uneasy',
            'serveral': 'several', 'flasy': 'flash', 'inlcudes': 'includes', 'lovwer': 'lower', 'pubilc': 'public', 'infrastructue': 'infrastructure', 'cosumnes': 'consumes', 'proximiatey': 'proximity', 'hase': 'has', 'recoverd': 'recovered', 'representive': 'representative', 'yrds': 'yards', 'discontinious': 'discontinuous', 'accordint': 'according', 'preparness': 'prepares', 'exteremely': 'extremely', 'potental': 'potential', 'unitl': 'until', 'extremal': 'extremelyl', 'managenet': 'management', 'exited': 'excited', 'easliy': 'easily', 'outhern': 'southern', 'exclates': 'escalates', 'vegetationin': 'vegetation', 'metropolitain': 'metropolitan','execrate': 'execute', 'asimple': 'simple', 'recoginized': 'recognized', 'multuple': 'multiple', 'albuqueque': 'albuquerque','isnt': 'is not', 'enity': 'entity', 'bandelier': 'bandolier', 'remians': 'remains', 'perctentage': 'percentage', 'incidcate': 'indicate', 'operaitons': 'operations', 'pertinent': 'pertinent', 'ouray': 'our', 'commitments': 'commitment', 'etimated': 'estimated', 'itent': 'tent', 'surounded': 'surrounded', 'cutltural': 'cultural', 'starr': 'start', 'onoging': 'ongoing', 'communcations': 'communications', 'exisited': 'existed', 'terrrain': 'terrain', 'preceeding': 'preceding', 'suday': 'sunday', 'withi': 'with', 'greates': 'greatest', 'curranty': 'current', 'pleanty': 'plenty', 'moniotring': 'monitoring', 'reserach': 'research', 'consits': 'consists', 'multipe': 'multiple', 'preditced': 'predicted', 'modelled': 'modeled', 'unknow': 'unknown', 'intermitent': 'intermittent', 'artic': 'attic','eagar': 'eager', 'transcept': 'transept', 'incluences': 'influences', 'bariers': 'barriers', 'paralell': 'parallel', 'scarrs': 'scars', 'jurisdicition': 'jurisdiction', 'fthe': 'the', 'completly': 'completely', 'considerated': 'considerate', 'humities': 'humilities', 'mplementation': 'implementation', 'continumity': 'continuity', 'barreirs': 'barriers', 'wuth': 'with', 'recretaion': 'recreation', 'whil': 'while', 'politicial': 'political', 'prediciton': 'prediction', 'idnicate': 'indicate', 'accomodate': 'accommodate', 'currrent': 'current', 'morman': 'mormon', 'proceding': 'preceding', 'continity': 'continuity', 'equiptment': 'equipment', 'litte': 'little', 'exisitng': 'existing', 'resrvoir': 'reservoir', 'pesent': 'present', 'magement': 'management', 'speads': 'spreads', 'overlaying': 'overlying', 'montan': 'mountain', 'bitterrot': 'bitteroot', 'inculding': 'including', 'continunity': 'continuity', 'weakend': 'weakened', 'tothe': 'the', 'suppresssion': 'suppression', 'contineous': 'continuous', 'mechancial': 'mechanical','consititutes': 'constitutes', 'wether': 'whether', 'putlooks': 'outlooks', 'commiunicated': 'communicated', 'conatain': 'contain', 'isstill': 'still', 'trafficed': 'trafficked', 'bourn': 'burn', 'incoporated': 'incorporated', 'exsisting': 'existing', 'forcecast': 'forecast', 'resulated': 'resulted', 'simpified': 'simplified', 'recration': 'recreation', 'equipement': 'equipment', 'mitaged': 'mitigated', 'achive': 'archive', 'lenghts': 'lengths', 'barrieris': 'barriers', 'inproved': 'improved', 'saffing': 'staffing', 'requirments': 'requirements', 'andr': 'and', 'anticipaed': 'anticipated', 'suppressin': 'suppression', 'bcause': 'because', 'alterned': 'altered', 'exibited': 'exhibited', 'alond': 'along', 'currnet': 'current', 'kimbel': 'kimble', 'kimble': 'nimble', 'exsists': 'exists', 'signifacantly': 'significantly', 'thupdate': 'update', 'elgible': 'eligible', 'politicol': 'political', 'heoghted': 'heighted', 'weatrher': 'weather', 'ahould': 'should', 'transtion': 'transition', 'foreward': 'forward', 'resorces': 'resources', 'mountian': 'mountain', 'threre': 'there', 'prebbles': 'pebbles', 'cource': 'course', 'tempatures': 'temperatures', 'adjecnt': 'adjacent', 'cources': 'sources', 'confier': 'conifer', 'cary': 'carry', 'countinous': 'continuous', 'objecvtive': 'objective', 'loacl': 'local', 'comminications': 'communications', 'prehsitroic': 'prehistoric', 'asssure': 'assure', 'influenes': 'influences', 'imapacts': 'impacts', 'consturced': 'constructed', 'resoures': 'resources', 'moustures': 'moisture', 'sevice': 'service', 'apahce': 'apache', 'sufficant': 'sufficient', 'residul': 'residual', 'thoough': 'through', 'surppression': 'suppression', 'sacrad': 'sacred', 'vegitatively': 'vegitatively', 'occassionally': 'occasionally', 'degrades': 'degraded', 'implemtation': 'implementation', 'advacned': 'advanced', 'foilure': 'failure', 'inaccessabilty': 'inaccessibility', 'attetempts': 'attempts', 'wth': 'with', 'referrenced': 'reference', 'attck': 'attack', 'difficlult': 'difficult', 'subdivisons': 'subdivisions', 'develope': 'develop', 'langmuir': 'languid', 'allignment': 'alignment', 'duraton': 'duration', 'barrirers': 'barriers', 'copetition': 'competition', 'elswhere': 'elsewhere', 'poetial': 'poetical', 'relativel': 'relative', 'standar': 'standard', 'thruogh': 'through', 'operaters': 'operators', 'exacuations': 'evacuations', 'oustide': 'outside', 'beig': 'being', 'trhrough': 'through', 'protimity': 'proximity', 'indiactes': 'indicates', 'remaim': 'remain', 'traild': 'trails', 'iterest': 'interest', 'confimed': 'confirmed', 'emediate': 'immediate', 'recived': 'received', 'fallm': 'fall', 'requrements': 'requirements', 'arge': 'are', 'adaquate': 'adequate', 'gurad': 'guard', 'happend': 'happened', 'colateral': 'collateral', 'flage': 'flag', 'accompained': 'accompanied','decreeased': 'decreased', 'transistion': 'transition', 'witihn': 'within', 'signifcnat': 'significant', 'protions': 'portions', 'numeroud': 'numerous', 'lcation': 'location', 'seaonal': 'seasonal', 'asssessment': 'assessment', 'addtional': 'additional', 'oragnization': 'organization', 'closests': 'closest', 'efforst': 'effort', 'organaztional': 'organizational', 'analyisis': 'analysis', 'estabilshed': 'established', 'pottential': 'potential', 'necesarrily': 'necessarily', 'immeadiate': 'immediate', 'avaliable': 'available', 'cummlus': 'cumulus', 'qualifed': 'qualified', 'altough': 'although', 'moniotoring': 'monitoring', 'releif': 'relief', 'moisturre': 'moisture', 'fireis': 'fires', 'comercial': 'commercial', 'geneeral': 'general', 'unble': 'unable', 'organiztion': 'organization', 'resjponse': 'response', 'doesnts': 'does not', 'sourthern': 'southern', 'cosidered': 'considered', 'togather': 'together', 'emmiter': 'emitter', 'midigated': 'mitigate', 'contuiue': 'continue', 'groupt': 'group', 'probabaility': 'probability', 'resourcces': 'resources', 'specilaized': 'specialized', 'suport': 'support', 'rappid': 'rapid', 'habbitat': 'habitat', 'meritt': 'merit', 'moderaty': 'moderate', 'pelative': 'relative', 'orgaizational': 'organizational', 'contunues': 'continues', 'berriers': 'barriers', 'adminstered': 'administered', 'ocurred': 'occurred', 'avaialable': 'available', 'adequite': 'adequate', 'assessent': 'assessment', 'coutry': 'country', 'decandent': 'decadent', 'bariiers': 'barriers', 'prodcution': 'production', 'anticiapted': 'anticipated', 'cultual': 'cultural', 'hazardious': 'hazardous', 'traffice': 'traffic', 'subivision': 'subdivision', 'outbuilings': 'outbuildings', 'scatterred': 'scattered', 'homogenuous': 'homogeneous', 'esitmated': 'estimated', 'stratagy': 'strategy', 'suffecient': 'sufficient', 'perinial': 'peroneal', 'prodominant': 'predominant', 'uninterupted': 'uninterrupted', 'suceptible': 'susceptible', 'interprise': 'enterprise', 'anticpate': 'anticipate', 'recend': 'recent', 'currentluy': 'currently', 'littlre': 'little', 'invloved': 'involved', 'conatinment': 'containment', 'droped': 'dropped', 'rehabed': 'rehabbed', 'recending': 'resending', 'flucuating': 'fluctuating', 'potentiol': 'potential', 'facilatate': 'facilitate', 'meeding': 'meeting', 'accoplishing': 'accomplishing', 'locak': 'local', 'destert': 'desert', 'contributeing': 'contributing', 'alhtough': 'although', 'liklehood': 'likelihood', 'threrat': 'threat', 'identifed': 'identified', 'raking': 'taking', 'aveage': 'average', 'humidy': 'humidity', 'incient': 'incident', 'verticle': 'vertical', 'nprocess': 'process', 'containement': 'containment', 'reudced': 'reduced', 'demenished': 'diminished', 'propogates': 'propagated', 'propagates': 'propagated', 'relliance': 'reliance', 'indictors': 'indicator', 'consideably': 'considerably', 'chruch': 'church', 'activty': 'activity', 'highlighted': 'highlight', 'operationl': 'operation', 'consideable': 'considerable', 'currenty': 'current', 'industrials': 'industrial', 'cutural': 'cultural', 'funcitional': 'functional', 'unlimted': 'unlimited', 'operatons': 'operations', 'rollong': 'rolling', 'surgace': 'surface', 'dtich': 'ditch', 'efficiencies': 'efficiencies', 'elevational': 'elevation', 'mimized': 'minimized', 'preperation': 'preparation', 'faclitate': 'facilitate', 'responsiblity': 'responsibility', 'montior': 'monitor', 'confiine': 'confine', 'thelong': 'long', 'evacutions': 'evacuation', 'warrents': 'warrants', 'basion': 'basin', 'conrtol': 'control', 'futre': 'future', 'cocnern': 'concern', 'optiosn': 'option', 'mainatin': 'maintain', 'postiven': 'positive', 'municiple': 'municipal', 'incuding': 'including', 'combinatin': 'combination', 'succes': 'success', 'oreder': 'order', 'integreates': 'integrated', 'amisture': 'mixture', 'inadeguatly': 'inadequately', 'occational': 'occasional', 'conersn': 'concern', 'aread': 'area', 'availiblity': 'availability', 'sporadice': 'sporadic','potentialy': 'potential', 'firel': 'fire', 'entrys': 'entries', 'bevavior': 'behavior', 'structrues': 'structures', 'iginition': 'ignition', 'residental': 'residential', 'proggessing': 'progressing', 'saied': 'said', 'infastraucture': 'infrastructure', 'unforseen': 'unforeseen', 'incorperated': 'incorporated', 'appox': 'approx', 'continuty': 'continuity', 'fom': 'for', 'experincing': 'experiencing', 'avaibility': 'amiability', 'restulting': 'resulting',
            'unaccessible': 'inaccessible', 'patroling': 'patrolling', 'sttep': 'step', 'vire': 'fire', 'helicpoters': 'helicopter', 'orginazation': 'organization', 'shouls': 'should', 'resistent': 'resistant', 'behavour': 'behaviour', 'throughtout': 'throughout', 'threathened': 'threatened', 'behavor': 'behavior','frnchmans': 'frenchman', 'visble': 'visible', 'adquate': 'adequate', 'accentuate': 'accentuated', 'depated': 'departed', 'witht': 'with', 'existance': 'existence', 'distnace': 'distance', 'stuructures': 'structures', 'minimaly': 'minimal', 'suficient': 'sufficient', 'ariel': 'aerial', 'areial': 'aerial', 'smioe': 'smile', 'previouly': 'previously','distrurbed': 'disturbed', 'eutopia': 'utopia', 'publice': 'public', 'creeding': 'creeping', 'natuural': 'natural', 'influnence': 'influence', 'strcutures': 'structures', 'coodinate': 'coordinate', 'infrastructual': 'infrastructure', 'representitive': 'representative', 'contitions': 'conditions', 'adminisitative': 'administrative', 'incindents': 'incidents', 'discontious': 'discontinue', 'progess': 'process', 'condions': 'conditions', 'sperad': 'spread', 'valuan': 'value', 'depts': 'departments', 'recreaion': 'recreation', 'cccupied': 'occupied', 'occcupied': 'occupied', 'decisional': 'decision', 'tosafety': 'safety', 'remail': 'remain', 'histrocial': 'historical', 'roacks': 'rocks', 'vegetaion': 'vegetation', 'seasson': 'season', 'soicial': 'social', 'esaily': 'easily', 'opeation': 'operation', 'poweder': 'powder', 'easly': 'early', 'isolatied': 'isolated', 'acton': 'action', 'probaility': 'probability', 'barrell': 'barrel', 'burnig': 'burning', 'occaissonal': 'occasional', 'inderest': 'interest', 'reqeater': 'repeated', 'establigh': 'establish', 'helicoter': 'helicopter', 'neightbors': 'neighbors', 'oredered': 'ordered', 'oppertunitys': 'opportunity', 'categorize': 'categories', 'decicion': 'decision', 'implemintation': 'implementation', 'polital': 'political', 'hight': 'height', 'humidies': 'humidity', 'tactive': 'active', 'corse': 'course', 'stratigey': 'strategy', 'ocassional': 'occasional', 'appropriatly': 'appropriately', 'surounding': 'surrounding', 'adequatley': 'adequately', 'icrease': 'increase', 'inholings': 'inholdings', 'opprtunities': 'opportunities', 'evacutations': 'evacuation', 'interst': 'interest', 'secures': 'secure', 'coverd': 'covered', 'asessment': 'assessment', 'arrier': 'barrier', 'yds': 'yards', 'highligh': 'highlight', 'sevaral': 'several', 'thurough': 'through', 'concen': 'concern', 'dirunal': 'diurnal', 'assment': 'assessment', 'subistence': 'subsistence', 'appalacian': 'appalachian', 'seend': 'send', 'vlaues': 'values', 'funtion': 'function', 'remaiin': 'remain', 'jurisdition': 'jurisdiction', 'thundershwers' : 'thundershowers'
            
            
    }

# Update misspellings
misspellings.update(badspell)


# In[ ]:




