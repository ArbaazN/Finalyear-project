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
from nltk.corpus import stopwords
import string
from tika import parser
from nltk.tokenize import word_tokenize 
from django.http import HttpResponseRedirect
import json
from django.core.paginator import Paginator
import csv
@csrf_protect

def predict(request):
	return render(request,"enroll/predict.html")
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
		# ig=[]
		# names = []
		# course=[]
		# locc=[]
		# fees = []
		# Ind = []
		# opn = []
		# minr = []
		# for i in range(len(result_final)):
		# 	ig.append(result_final.iloc[i][0])
		# 	names.append(result_final.iloc[i][1])
		# 	locc.append(result_final.iloc[i][2])
		# 	course.append(result_final.iloc[i][3])
		# 	fees.append(result_final.iloc[i][4])
		# 	Ind.append(result_final.iloc[i][5])
		# 	opn.append(result_final.iloc[i][6])
		# 	minr.append(result_final.iloc[i][7])
		result_final=result_final.to_json(orient='records')
		data=[]
		data=json.loads(result_final)		
		return render(request,'enroll/predict.html',{'data':data})

	if request.method=='POST' and request.POST.get('ct'):
		df=pd.read_csv(r"C:\Users\admin\Desktop\Finalyear\data\sample.csv")

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
	#return render(request,'enroll/predict.html')
def job_search(request):
	if request.method == 'POST' and request.FILES['myfile']:
		myfile = request.FILES['myfile']
		fs = FileSystemStorage()
		filename = fs.save(myfile.name, myfile)
		uploaded_file_path = fs.path(filename)
		uploaded_file_url = fs.url(filename)
		skillDataset = pd.read_csv(r"C:\Users\admin\Desktop\companies_data.csv")
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
		# # Removing numbers from text file
		# obtainedResumeText = re.sub(r'\d+','',obtainedResumeText)
		 # Remove punctuation from the text files
		obtainedResumeText = obtainedResumeText.translate(str.maketrans('','',string.punctuation))

		filteredTextForSkillExtraction = stopWordRemoval(obtainedResumeText)
		resumeTechnicalSkillSpecificationList = {'Skill':skillsList}
		technicalSkillScore , technicalSkillExtracted = ResumeSkillExtractor(resumeTechnicalSkillSpecificationList,filteredTextForSkillExtraction)
		
		dataList = {"candi_skills":technicalSkillExtracted}#,'Company_name':compname,'Job_role':comprole,'Job_loc':comploc}
		softwareDevelopemtTechnicalSkills = pd.DataFrame(dataList)

		df=softwareDevelopemtTechnicalSkills.explode('candi_skills')

		df.drop_duplicates(keep='first',inplace=True)

		df1 = pd.read_csv(r"C:\Users\admin\Desktop\companies_data.csv")

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
		dfc=dfz.to_csv("C:\\Users\\admin\\Desktop\\Finalyear\\search_file.csv",index=False)

		result_final = dfz
		# name=[]
		# role = []
		# exp = []
		# loc = []
		# desc = []
		# for i in range(len(result_final)):
		# 	name.append(result_final.iloc[i][1])
		# 	role.append(result_final.iloc[i][2])
		# 	exp.append(result_final.iloc[i][3])
		# 	loc.append(result_final.iloc[i][4])
		# 	desc.append(result_final.iloc[i][5])
		result_final=result_final.to_json(orient='records')
		data=[]
		data=json.loads(result_final)

		# for i in range(len(dfz['comp_name'])):
		# 	# Insert in the database
		# 	srs.objects.create(comp_name = dfz['comp_name'][i], comp_role = dfz['comp_role'][i], comp_exp = dfz['comp_exp'][i], comp_loc = dfz['comp_loc'][i])
		# try:
		#     obj = srs.objects.get(comp_name = dfz['comp_name'][i], comp_role = dfz['comp_role'][i], comp_exp = dfz['comp_exp'][i], comp_loc = dfz['comp_loc'][i])
		#     for key, value in defaults.items():
		#         setattr(obj, key, value)
		#     obj.save()
		# except srs.DoesNotExist:
		#     new_values = {'first_name': 'John', 'last_name': 'Lennon'}
		#     new_values.update(defaults)
		#     obj = srs(**new_values)
		#     obj.save()
		rd=pd.read_csv(r"search_file.csv")
	
		jobs = dfz.to_dict(orient='records')
		jobs = rd.to_dict(orient='records')
		job_paginator = Paginator(jobs,20)

		page_num = request.GET.get('page')

		page = job_paginator.get_page(page_num) 

		return render(request,'enroll/job_search.html',{'dd':data,'uploaded_file_url': uploaded_file_url,'count' : job_paginator.count,
	        'page' : page})
		#{'comp_name':name,'comp_role':role,'comp_exp':exp,'comp_loc':loc,'comp_desc':desc}	'dd':data,	
	return render(request,'enroll/job_search.html')

def loadSkillDataset():
	skillDataset = pd.read_csv(r"C:\Users\admin\Desktop\companies_data.csv")
	skills = list(skillDataset['comp_skills'])
	name = list(skillDataset['comp_name'])
	role = list(skillDataset['comp_role'])
	loc=list(skillDataset['comp_loc'])
	cleanedskillList = [x for x in skills if str(x) != 'nan']
	cleanednameList = [x for x in name if str(x) != 'nan']
	cleanedroleList = [x for x in role if str(x) != 'nan']
	cleanedlocList= [x for x in loc if str(x) != 'nan']
	return cleanedskillList , cleanednameList, cleanedroleList,cleanedlocList

skillsList , nameList, roleList, locList = loadSkillDataset()

obtainedResumeText = ''

firstLetterCapitalizedObtainedResumeText = []
def CapitalizeFirstLetter(obtainedResumeText):
	capitalizingString = " "
	obtainedResumeTextLowerCase = obtainedResumeText.lower()
	obtainedResumeTextUpperCase = obtainedResumeText.upper()
	splitListOfObtainedResumeText = obtainedResumeText.split()
	for i in splitListOfObtainedResumeText:
		firstLetterCapitalizedObtainedResumeText.append(i.capitalize())        
	return (capitalizingString.join(firstLetterCapitalizedObtainedResumeText),obtainedResumeTextLowerCase,obtainedResumeTextUpperCase)
firstLetterCapitalizedText,obtainedResumeTextLowerCase,obtainedResumeTextUpperCase = CapitalizeFirstLetter(obtainedResumeText)

# obtainedResumeText = obtainedResumeTextLowerCase + obtainedResumeTextUpperCase + firstLetterCapitalizedText
# # Removing numbers from text file
# obtainedResumeText = re.sub(r'\d+','',obtainedResumeText)
# # Remove punctuation from the text files
# obtainedResumeText = obtainedResumeText.translate(str.maketrans('','',string.punctuation))

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
	
filteredTextForSkillExtraction = stopWordRemoval(obtainedResumeText)

resumeTechnicalSkillSpecificationList = {'Skill':skillsList,
			'Company name':nameList}

def ResumeSkillExtractor(resumeTechnicalSkillSpecificationList,filteredTextForSkillExtraction):
	skill = 0
	
	# Create an empty list where the scores will be stored
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
	return skillScores,skillExtracted#Compname,Comprole,Comploc

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
		# for i in range(len(result_final)):
		# 	name.append(result_final.iloc[i][1])
		return render(request,'enroll/search.html',{'d':data,'name':ip})
	return render(request,'enroll/search.html')

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
def home(request):
    return render(request,"enroll/index.html")

# def job_search(request):
#     return render(request,"enroll/job_search.html")
 



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

def signin(request):
    if request.method =="POST":
        username = request.POST['username']
        pass1 = request.POST['pass1']
        user = authenticate(username=username,password=pass1)
        if user is not None:
            login(request,user)
            fname=user.first_name
            messages.info(request,"Hello "+fname)
            return render(request,"enroll/index.html",{'fname':fname})
        else:
            messages.error(request,"Bad Creadentials!")
            return redirect('signin')
    return render(request,"enroll/signin.html")


def signout(request):
    logout(request)
    messages.success(request,"Logged Out succesfully")
    return redirect('home')
