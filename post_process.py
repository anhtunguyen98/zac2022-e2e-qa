import dateparser
import json
from thefuzz import fuzz
from dateparser.search import search_dates
import os
from tqdm import tqdm
from datetime import datetime

def format_date(question, answer):
    question = question.lower()
    # if 'ngày' in question or 'tháng' in question or 'năm' in question:

    strings = search_dates(answer,languages=['vi'])
    if strings is not None:
        # print(strings)
        date = strings[0][1]
        if date is not None:
            if 'ngày' in question:
                if 'tháng' in question and 'năm' in question:
                    if date.day.numerator == datetime.today().day.numerator and date.month.numerator == datetime.today().month.numerator :
                        answer = f"năm {date.year.numerator}"
                    elif date.year.numerator == datetime.today().year.numerator:
                        answer = f"ngày {date.day.numerator} tháng {date.month.numerator}"
                    else:
                        answer = f"ngày {date.day.numerator} tháng {date.month.numerator} năm {date.year.numerator}"
                elif 'năm' in question:
                    answer = f"năm {date.year.numerator}"
                else:
                    if date.day.numerator == datetime.today().day.numerator and date.month.numerator == datetime.today().month.numerator :
                        answer = f"năm {date.year.numerator}"
                    elif date.year.numerator == datetime.today().year.numerator:
                        answer = f"ngày {date.day.numerator} tháng {date.month.numerator}"
                    else:
                        answer = f"ngày {date.day.numerator} tháng {date.month.numerator} năm {date.year.numerator}"
                return answer, True
            elif 'năm' in question:
                answer = f"năm {date.year.numerator}"
                return answer, True
            
            elif 'thời gian' in question or 'lúc' in question or 'khi' in question:
                if date.day.numerator == datetime.today().day.numerator and date.month.numerator == datetime.today().month.numerator :
                    answer = f"năm {date.year.numerator}"
                elif date.year.numerator == datetime.today().year.numerator:
                    answer = f"ngày {date.day.numerator} tháng {date.month.numerator}"
                else:
                    answer = f"ngày {date.day.numerator} tháng {date.month.numerator} năm {date.year.numerator}"
                return answer, True
            else:
                return answer, False
        else:
            if 'năm' in answer.lower():
                
                return answer.lower(), True
            else:     
                return answer, False
    else:
        if 'năm' in answer.lower():
            
            return answer.lower(), True
        else:
            return answer, False
        
def search_answer(answer, entities, title):

    best_score = 0
    best_key = ''
    if title in entities:
        for key in entities[title].keys():
            score = fuzz.token_sort_ratio(answer, key)
            if score > best_score:
                best_score = score
                best_key = key
        if best_key != '':
            score = fuzz.token_sort_ratio(answer.lower(), title.lower())
            if score > best_score:
                result = 'wiki/' + title.replace(' ', '_')
            else:
                result = 'wiki/'+ entities[title][best_key].replace(' ','_')

        else:
            result = None
    else:
        score = fuzz.token_sort_ratio(answer.lower(), title.lower())
        if score > 50:
            result = 'wiki/' + title.replace(' ', '_')
            best_score = score
        else:
            result = None
    return result, best_score

def process_answer(sample,entities, all_titles):
        
        result = {
            'id': sample['id'],
            'question': sample['question']
        }
        if sample['answer'].lower() == sample['title'].lower():
            answer = 'wiki/'+ sample['title'].replace(' ', '_')
        else:
            # check date
            try:
                answer, check_date = format_date(sample['question'], sample['answer'])
            except:
                print(sample['answer'])
                print(format_date(sample['question'], sample['answer']))
            if check_date is False:
                if not answer.isdigit():

                    if answer.lower() in all_titles:
                        answer = 'wiki/'+ all_titles[answer.lower()].replace(' ', '_')
                    else:
                        answer, score = search_answer(answer,entities, sample['title'])
                        # title, score_title = search_title(answer,all_titles)
                        # if score > score_title:
                        #     answer = answer_tmp
                        # else:
                        #     answer = title

            # search answer


        
        result['answer'] = answer
        
        return result


