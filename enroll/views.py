from django.http.response import HttpResponseNotModified
from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate ,login, logout 
from django.shortcuts import render;
import json # will be needed for saving preprocessing details
import numpy as np # for data manipulation
import pandas as pd # for data manipulation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.template.response import TemplateResponse
from django.views.decorators.csrf import csrf_protect
from django import forms
from django.core.files.storage import FileSystemStorage
from django.views.generic import TemplateView
from collections import Counter
import re
import nltk
import spacy
from nltk.corpus import stopwords
import string
from tika import parser
from nltk.tokenize import word_tokenize 
from django.http import HttpResponseRedirect
import json
from django.core.paginator import Paginator
import csv
import PyPDF2
from pdf2docx import parse ,Converter
import docx2txt
from pdfminer.high_level import extract_text
from spacy.matcher import Matcher
import locationtagger
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)
data=[]
@csrf_protect
def predict(request):
	if request.user.is_authenticated:
		return render(request,"enroll/predict.html")
	else:
		messages.info(request,"Please Login In Order To Access the Features")
		return redirect('home')
"""-----------------------------------------------------------------------------"""
def result(request):
	if request.method=='POST' and request.POST.get('nata'):
		df=pd.read_csv(r"C:\Users\admin\Desktop\Finalyear\data\12-ARCH-final-done.csv")
		count = CountVectorizer(stop_words='english')
		count_matrix = count.fit_transform(df[['AllIndia']])
		cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
		indices = pd.Series(df.index, index=df['college_name'])
		clgs = [df['AllIndia'][i] for i in range(len(df['AllIndia']))]
		score=float(request.POST.get('nata',None))
		cosine_sim = cosine_similarity(count_matrix, count_matrix)
		idx=df.loc[df.AllIndia<=score,['college_img','college_name','college_loc','college_course','college_fees','AllIndia','Open','Minority']]
		result_final = idx
		result_final=result_final.to_json(orient='records')
		data=[]
		data=json.loads(result_final)	
		return render(request,'enroll/predict.html',{'data':data})
	if request.method=='POST' and request.POST.get('ct'):
		df=pd.read_csv(r"D:\Users\sahil\Desktop\Final_year_project\Finalyear-project-master\data\sample.csv")
		count = CountVectorizer(stop_words='english')
		count_matrix = count.fit_transform(df[['AllIndia']])
		cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
		indices = pd.Series(df.index, index=df['college_name'])
		clgs = [df['AllIndia'][i] for i in range(len(df['AllIndia']))]
		score=float(request.POST.get('ct',''))
		cosine_sim = cosine_similarity(count_matrix, count_matrix)
		idx=df.loc[df.AllIndia<=score,['college_img','college_name','college_fees','AllIndia','Open','Minority']]
		idx=idx[:10]
		result_final = idx
		ig=[]
		names = []
		fees = []
		Ind = []
		opn = []
		minr = []
		for i in range(len(result_final)):
			ig.append(result_final.iloc[i][0])
			names.append(result_final.iloc[i][1])
			fees.append(result_final.iloc[i][2])
			Ind.append(result_final.iloc[i][3])
			opn.append(result_final.iloc[i][4])
			minr.append(result_final.iloc[i][5])
				
		
		return render(request,'enroll/predict.html',{"college_img":ig,"college_name":names,"college_fees":fees,"AllIndia":Ind,"Open":opn,"Minority":minr})
"""-------------------------------------------------------------------------------------------------------------"""
def job_search(request):
	if request.user.is_authenticated:
		if request.method == 'POST' and request.FILES['myfile']:
			myfile = request.FILES['myfile']
			fs = FileSystemStorage()
			filename = fs.save(myfile.name, myfile)
			uploaded_file_path = fs.path(filename)
			uploaded_file_url = fs.url(filename)
			skillDataset = pd.read_csv(r"D:\Users\sahil\Desktop\Final_year_project\Finalyear-project-master\data\companies_data.csv")
			skills = list(skillDataset['comp_skills'])
			cleanedskillList = [x for x in skills if str(x) != 'nan']
			cleanedskillList = [i.split()[0] for i in skills]
			skillsList=cleanedskillList
			newResumeTxtFile = open('sample.txt', 'w',encoding='utf-8')
			resumeFile =uploaded_file_path
			resumeFileData = parser.from_file(resumeFile)
			fileContent = resumeFileData['content']
			newResumeTxtFile.write(fileContent)
			obtainedResumeText = fileContent
			firstLetterCapitalizedObtainedResumeText = []
			firstLetterCapitalizedText,obtainedResumeTextLowerCase,obtainedResumeTextUpperCase = CapitalizeFirstLetter(obtainedResumeText)
			obtainedResumeText = obtainedResumeTextLowerCase + obtainedResumeTextUpperCase + firstLetterCapitalizedText
			obtainedResumeText = obtainedResumeText.translate(str.maketrans('','',string.punctuation))
			filteredTextForSkillExtraction = stopWordRemoval(obtainedResumeText)
			resumeTechnicalSkillSpecificationList = {'Skill':skillsList}
			technicalSkillScore , technicalSkillExtracted = ResumeSkillExtractor(resumeTechnicalSkillSpecificationList,filteredTextForSkillExtraction)
			dataList = {"candi_skills":technicalSkillExtracted}#,'Company_name':compname,'Job_role':comprole,'Job_loc':comploc}
			softwareDevelopemtTechnicalSkills = pd.DataFrame(dataList)
			df=softwareDevelopemtTechnicalSkills.explode('candi_skills')
			df.drop_duplicates(keep='first',inplace=True)
			df1 = pd.read_csv(r"D:\Users\sahil\Desktop\Final_year_project\Finalyear-project-master\data\companies_data.csv")
			df1['comp_skills'] = df1['comp_skills'].str.split()
			df1['matchedName'] = df1['comp_skills'].apply(lambda x: [item for item in x if item in df['candi_skills'].tolist()])
			df1['mskills'] = [','.join(map(str, l)) for l in df1['matchedName']]
			df1.drop(['matchedName'],axis=1,inplace=True)
			df1['mskills'].replace('',np.nan,inplace=True)
			df1=df1.dropna()
			df1['cmp_skills'] = [','.join(map(str, l)) for l in df1['comp_skills']]
			df1.drop(['comp_skills'],axis=1,inplace=True)
			df1.drop_duplicates(keep='first',inplace=True)
			dfz = df1.reset_index(drop=True)
			dfc=dfz.to_csv("D:\\Users\\sahil\\Desktop\\Final_year_project\\Finalyear-project-master\\data\\search_file.csv",index=False)
			result_final = dfz
			result_final=result_final.to_json(orient='records')
			data=[]
			data=json.loads(result_final)
			rd=pd.read_csv(r"search_file.csv")
			jobs = dfz.to_dict(orient='records')
			jobs = rd.to_dict(orient='records')
			job_paginator = Paginator(jobs,20)
			page_num = request.GET.get('page')
			page = job_paginator.get_page(page_num) 
			return render(request,'enroll/job_search.html',{'dd':data,'uploaded_file_url': uploaded_file_url,'count' : job_paginator.count,
				'page' : page})			
	else:
		messages.info(request,"Please Login In Order To Access the Features")
		return redirect('home')		
	return render(request,'enroll/job_search.html')
"""-----------------------------------------------------------------------------"""
def loadSkillDataset():
	skillDataset = pd.read_csv(r"D:\Users\sahil\Desktop\Final_year_project\Finalyear-project-master\data\companies_data.csv")
	skills = list(skillDataset['comp_skills'])
	name = list(skillDataset['comp_name'])
	role = list(skillDataset['comp_role'])
	loc=list(skillDataset['comp_loc'])
	cleanedskillList = [x for x in skills if str(x) != 'nan']
	cleanednameList = [x for x in name if str(x) != 'nan']
	cleanedroleList = [x for x in role if str(x) != 'nan']
	cleanedlocList= [x for x in loc if str(x) != 'nan']
	return cleanedskillList , cleanednameList, cleanedroleList,cleanedlocList
"""-----------------------------------------------------------------------------"""
skillsList , nameList, roleList, locList = loadSkillDataset()
obtainedResumeText = ''
firstLetterCapitalizedObtainedResumeText = []
"""-----------------------------------------------------------------------------"""
def CapitalizeFirstLetter(obtainedResumeText):
	capitalizingString = " "
	obtainedResumeTextLowerCase = obtainedResumeText.lower()
	obtainedResumeTextUpperCase = obtainedResumeText.upper()
	splitListOfObtainedResumeText = obtainedResumeText.split()
	for i in splitListOfObtainedResumeText:
		firstLetterCapitalizedObtainedResumeText.append(i.capitalize())        
	return (capitalizingString.join(firstLetterCapitalizedObtainedResumeText),obtainedResumeTextLowerCase,obtainedResumeTextUpperCase)
firstLetterCapitalizedText,obtainedResumeTextLowerCase,obtainedResumeTextUpperCase = CapitalizeFirstLetter(obtainedResumeText)
"""-----------------------------------------------------------------------------"""
def stopWordRemoval(obtainedResumeText):
	stop_words = set(stopwords.words('english')) 
	word_tokens = word_tokenize(obtainedResumeText) 
	filtered_sentence = [w for w in word_tokens if not w in stop_words] 

	filtered_sentence = [] 
	joinEmptyString = " "
	for w in word_tokens: 
		if w not in stop_words: 
			filtered_sentence.append(w)
	return(joinEmptyString.join(filtered_sentence))
"""-----------------------------------------------------------------------------"""
filteredTextForSkillExtraction = stopWordRemoval(obtainedResumeText)
resumeTechnicalSkillSpecificationList = {'Skill':skillsList,
			'Company name':nameList}
"""-----------------------------------------------------------------------------"""
def ResumeSkillExtractor(resumeTechnicalSkillSpecificationList,filteredTextForSkillExtraction):
	skill = 0
	skillScores = []
	skillExtracted = []

	# Obtain the scores for each area
	for area in resumeTechnicalSkillSpecificationList.keys():

		if area == 'Skill':
			skillWord = []
			for word in resumeTechnicalSkillSpecificationList[area]:
				if word in filteredTextForSkillExtraction:
					skill += 1
					skillWord.append(word)
			skillExtracted.append(skillWord)
			skillScores.append(skill)
	return skillScores,skillExtracted
"""-----------------------------------------------------------------------------"""
dfc=pd.read_csv(r"search_file.csv")
regex=skillsList
def search(request):
	if request.method == 'POST' and request.POST.get('search'):
		ip=str(request.POST.get('search',None))
		textlikes = dfc.select_dtypes(include=[object, "string"])
		bs=textlikes.apply(
				lambda column: column.str.contains(ip,regex=True,case=False,na=False)
			).any(axis=1)
		result_final = dfc[bs]
		result_final=result_final.reset_index().to_json(orient='records')
		data=[]
		data=json.loads(result_final)

		job_paginator = Paginator(data, 20)

		page_num = request.GET.get('page')

		page = job_paginator.get_page(page_num)
		return render(request,'enroll/search.html',{'d':data,'name':ip,'page':page})
	return render(request,'enroll/search.html')
"""-----------------------------------------------------------------------------"""
def pagination(request):
	rd=pd.read_csv(r"search_file.csv")
	jobs = rd.to_dict(orient='records')

	job_paginator = Paginator(jobs, 20)

	page_num = request.GET.get('page')

	page = job_paginator.get_page(page_num)

	context = {
		'count' : job_paginator.count,
		'page' : page
	}
	return render(request, 'enroll/pagination.html', context)
"""-----------------------------------------------------------------------------"""
def sea_pag(request):
	rd=pd.read_csv(r"search_file.csv")
	jobs = rd.to_dict(orient='records')

	job_paginator = Paginator(jobs, 20)

	page_num = request.GET.get('page')

	page = job_paginator.get_page(page_num)

	context = {
		'count' : job_paginator.count,
		'page' : page
	}
	return render(request, 'enroll/pagination.html', context)
def home(request):
    return render(request,"enroll/index.html")
def rsm_a(request):
    return render(request,"enroll/rsm_a.html")
"""-----------------------------------------------------------------------------"""
def signup(request):
    if request.method == "POST":
        username = request.POST['username']
        fname = request.POST['fname']
        lname = request.POST['lname']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']

        if User.objects.filter(username=username):
            messages.error(request,"user already exist")
            return redirect('signup')
        if User.objects.filter(email=email):
            messages.error(request,"Email is already registered")
            return redirect('signup')
        if len(username)>10:
            messages.error(request,"username should be less then 10 character")
            return redirect('signup')
        if pass1 != pass2:
            messages.error(request,"Password and confirm password did not match")
            return redirect('signup')
        if not username.isalnum():
            messages.error(request,"Username must me alpha numeric!")
            return redirect('signup')


        myuser = User.objects.create_user(username,email,pass1)
        myuser.first_name = fname
        myuser.last_name = lname
        myuser.save()
        messages.success(request,"Your Account has been successfully Created")
        return redirect('signin')

    return render(request,"enroll/signup.html")
"""-----------------------------------------------------------------------------"""
def signin(request):
    if request.method =="POST":
        username = request.POST['username']
        pass1 = request.POST['pass1']
        user = authenticate(username=username,password=pass1)
        if user is not None:
            login(request,user)
            fname=user.first_name
            messages.success(request,"Hello "+fname)
            return render(request,"enroll/index.html",{'fname':fname})
        else:
            messages.error(request,"Bad Creadentials!")
            return redirect('signin')
    return render(request,"enroll/signin.html")
"""-----------------------------------------------------------------------------"""
def signout(request):
    logout(request)
    messages.success(request,"Logged Out succesfully")
    return redirect('home')
"""-----------------------------------------------------------------------------"""
def rsm_a(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name,myfile)
        uploaded_file_path = fs.path(filename)
        uploaded_file_url = fs.url(filename)
        extension=filename.split(".")[-1]
        urls=extract_urls(uploaded_file_path)
        if extension=='docx':
            text=extract_text_from_docx(uploaded_file_path)
        elif extension == 'pdf':
            text=extract_text_from_pdf(uploaded_file_path)
            convert_pdf_to_docx(uploaded_file_path)
            resume_text=extract_text_from_docx('./demo.docx')    
        else:
            pass
        place_entity = locationtagger.find_locations(text = resume_text)
        cities=place_entity.cities
        if((text != '' ) and (resume_text != '')):
            """-----------0------------"""
            name=proper_name(resume_text)
            if(name==''):
                data.append(None)
            else:
                data.append(name)
            """-----------1-------------"""
            phone_number=extract_phone_number(text)
            if(phone_number==''):
                data.append(None)
            else:
                data.append(phone_number)
            """-----------2-------------"""
            emails=extract_emails(text)
            if len(emails):
                data.append(emails[0])
            else:
                data.append(None)
            """-----------3-------------"""
            skills_list=list(extract_skills(text))
            data.append(skills_list)
            """-----------4-------------"""
            skills_score=extract_skills_score(text)
            data.append(skills_score)
            """-----------5-------------"""
            linkedin_urls=extract_linkedin(urls)
            if(linkedin_urls==''):
                data.append(None)
            else:
                data.append(linkedin_urls)
            """-----------6-------------"""
            Github_urls=extract_Github(urls)
            if(Github_urls==''):
                data.append(None)
            else:
                data.append(Github_urls)
            """-----------7-------------"""
            education_score=Validation_education(text)
            data.append(education_score)
            "--------------8---------------"
            experience_score=Validation_experience(resume_text)
            data.append(experience_score)       
            "---------------9--------------"
            project_score=Validate_Projects(resume_text)
            data.append(project_score)
            "---------------10--------------"
            if len(cities):
                data.append(cities)
            else:
                data.append(None)
            return redirect('display')
        else:
            return messages.info(request,'Blank Document')
            
    return render(request, 'enroll/rsm_a.html')

def display(request):
    name=data[0]
    phone_number=data[1]
    email=data[2]
    skills_list=data[3]
    skills_score=data[4]
    linkedin_url=data[5]
    Github_url=data[6]
    education_score=data[7]
    experience_score=data[8]
    project_score=data[9]
    cities=data[10]
    data.clear()
    "5+5+10+30+20+20+5+5="
    if(phone_number):
        phone_marks=5
    else:
        phone_marks=0
    if(email):
        email_marks=5
    else:
        email_marks=0   
    if(linkedin_url):
        linkedin_marks=10
    else:
        linkedin_marks=0
    if(skills_score < 8):
        skills_marks=10
    elif(skills_score > 8 and skills_score < 15):
        skills_marks=15
    else:
        skills_marks=30
    if(education_score !=0):
        education_marks=20
    else:
        education_marks=0
    if((Github_url != None) or (project_score != 0)):
        project_marks=20
    else:
        project_marks=0
    if(experience_score != 0):
        experience_marks=5
    else:
        experience_marks=0
    if(cities):
        location_marks=5
    else:
        location_marks=0
    labels=['phone', 'email', 'linkedin', 'skills','education','project','Experience','Location']
    data1=[phone_marks,email_marks, linkedin_marks, skills_marks, education_marks,project_marks,experience_marks,location_marks]
    totMarks=0
    for j in data1:
        totMarks=totMarks+j

    return render(request,'enroll/display.html',{
        'labels': labels,
        'data': data1,
        'totMarks':totMarks,
        
    })  

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

"""----------------------------------------------------------------------"""
def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        return txt.replace('\t',' ')
    return None
"""----------------------------------------------------------------------"""
def convert_pdf_to_docx(pdf_path):
        docx_path='./demo.docx'
        cv=Converter(pdf_path)
        cv.convert(docx_path,start=0,end=None)
        cv.close()


"""----------------------------------------------------------------------"""
#Extracting Name Method 1
def extract_name(resume_text):
    nlp_text = nlp(resume_text)
    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'},{'POS': 'PROPN'},{'POS': 'PROPN'}]
    matcher.add('NAME',[pattern])
    matches = matcher(nlp_text)
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text
#Cleaning and Comparing and returning name from Resume      
def proper_name(resume_text):
    if(resume_text != ''):
        name1=extract_name(resume_text)
        if(name1 != None):
            tokenize_name1=name1.split()
            first_name1=tokenize_name1[0]
            last_name1=tokenize_name1[1]
        resume_text=resume_text.split()
        first_name=resume_text[0]
        last_name=resume_text[1]
        full_name=first_name+' '+last_name
        if(name1 == None):
            return full_name
        elif(first_name1==first_name and last_name1==last_name):
            return name1
        else:
            return full_name

"""----------------------------------------------------------------------"""
#Mobile Number Extracting
PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
def extract_phone_number(resume_text):
    phone = re.findall(PHONE_REG, resume_text)

    if phone:
        number = ''.join(phone[0])

        if resume_text.find(number) >= 0 and len(number) < 12:
            return number
    return None
"""----------------------------------------------------------------------"""
#Email Extracting
EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')
def extract_emails(resume_text):
    return re.findall(EMAIL_REG, resume_text)
"""----------------------------------------------------------------------"""
#Linked Extracting
def extract_linkedin(urls):
    for element in urls:
        if((re.search('linkedin.com/in/', element)!= None) or (re.search('https://linkedin.com/in/', element)!= None)):
            return element
"""----------------------------------------------------------------------"""
#Github Extracting
def extract_Github(urls):
    for el in urls:
        if((re.search('github.com/', el)!= None) or (re.search('https://github.com/', el)!= None)):
            return el
"""----------------------------------------------------------------------"""
#Education Extraction
STOPWORDS = set(stopwords.words('english'))
EDUCATION =["BACHELOR","MASTERS","DIPLOMA","HIGHER","SECONDARY","BCA","MCA","BSC","BE","B.E",'B.COM','M.COM'
            "ME","M.E", "MS", "M.S", "BCS","B.C.S" ,"B.E.","M.E.","M.S.","B.C.S.","C.A","CA",'C.A.', 'MBA','PHD'
            "B.TECH", "M.TECH", "BA","B.A","BS","B.S"
            "SSC", "HSC", "CBSE", "ICSE", "X", "XII"]
def extract_education(resume_text):
    nlp_text = nlp(resume_text)
    # Sentence Tokenizer
    nlp_text = [sent.text.strip() for sent in nlp_text.sents]
    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        #print(index, text), print('-'*50)
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.upper() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index + 1]
                print(edu.keys())
"""----------------------------------------------------------------------"""
#Valdating That Education Section is Present OR NOT
newData =[]
def Validation_education(resume_text):
    resume_text = resume_text.strip()
    resume_text = resume_text.split()
    for i in range(len(resume_text)):
        resume_text[i]=resume_text[i].upper()
    score = 0
    for i in EDUCATION:
        for j in resume_text:
            if(i==j):
                score += 1
    return score
"""----------------------------------------------------------------------"""
def extract_institute(input_text):
    data=pd.read_csv("./data/schools.csv")
    school_DB=list(data.columns.values)



"""----------------------------------------------------------------------"""

#Extracting Skills and Skills Count From skills and experience Section
def extract_skills(input_text):
    length_of_list=0
    data = pd.read_csv("./data/skills.csv") 
    SKILLS_DB = list(data.columns.values)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(input_text)
 
    # remove the stop words
    filtered_tokens = [w for w in word_tokens if w not in stop_words]
 
    # remove the punctuation
    filtered_tokens = [w for w in word_tokens if w.isalpha()]
 
    # generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))
 
    # we create a set to keep the results in.
    found_skills = set()
 
    # we search for each token in our skills database
    for token in filtered_tokens:
        if token.lower() in SKILLS_DB:
            found_skills.add(token)
 
    # we search for each bigram and trigram in our skills database
    for ngram in bigrams_trigrams:
        if ngram.lower() in SKILLS_DB:
            found_skills.add(ngram)
    return found_skills
"""----------------------------------------------------------------------"""
#Skills Score 
def extract_skills_score(resume_text):
    skills_list=extract_skills(resume_text)
    length_of_list=len(skills_list)
    return length_of_list

"""----------------------------------------------------------------------"""
def extract_urls(pdf_path):
    PDFFile = open(pdf_path,'rb')
    PDF = PyPDF2.PdfFileReader(PDFFile)
    pages = PDF.getNumPages()
    key = '/Annots'
    uri = '/URI'
    ank = '/A'
    urls=[]
    for page in range(pages):
        print("Current Page: {}".format(page))
        pageSliced = PDF.getPage(page)
        pageObject = pageSliced.getObject()
        if key in pageObject.keys():
            ann = pageObject[key]
            for a in ann:
                u = a.getObject()
                if uri in u[ank].keys():
                    urls.append(u[ank][uri])
    return urls

Experience = ['accomplishments','experience','professional experience','leadership','companies','worked','publications']

def Validation_experience(resume_text):
    resume_text = resume_text.strip()
    resume_text = resume_text.split()
    for i in range(len(resume_text)):
        resume_text[i]=resume_text[i].lower()
    score = 0
    for i in Experience:
        for j in resume_text:
            if(i==j):
                score += 1
    return score

Projects = ['projects','publications','certifications',]
"--------------------------------------------------------------------------------------"
def Validate_Projects(resume_text):
    resume_text = resume_text.strip()
    resume_text = resume_text.split()
    for i in range(len(resume_text)):
        resume_text[i]=resume_text[i].lower()
    score = 0
    for i in Projects:
        for j in resume_text:
            if(i==j):
                score += 1
    return score