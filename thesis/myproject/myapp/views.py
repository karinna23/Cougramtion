from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from .models import *
from .forms import PredictionForm
import joblib
import numpy as np
import openai
from django.conf import settings



# ----- WEASYPRINT PDF -----
from django.http import HttpResponse
from django.template import loader
from weasyprint import HTML
import datetime



def index(request):
    return render(request, "index.html")


def Courses(request):
    return render(request, "courses.html")


def Recommend(request):
    messages = []

    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Get input values
            first_name = form.cleaned_data['first_name']
            last_name = form.cleaned_data['last_name']
            sex = form.cleaned_data['sex']
            cet = form.cleaned_data['cet']
            gpa = form.cleaned_data['gpa']
            strand = form.cleaned_data['strand']

            # Map strand to text
            strand_mapping = {
                '1': 'General Academic Strand (GAS)',
                '2': 'Humanities and Social Sciences (HUMSS)',
                '4': 'Science, Technology, Engineering, and Mathematics (STEM)',
                '5': 'Technology Vocational and Livelihood (TVL)',
                '0': 'Accountancy, Business, and Management (ABM)',
                '6': 'Arts & Design (AD)',
                '3': 'Sports Track (SP)',
            }
            strand_text = strand_mapping.get(strand)

            # Prepare input data for prediction
            input_data = [[cet, gpa, strand]]

            # Use your model to make predictions
            model = joblib.load(r"C:\Users\acer\Desktop\thesis\myproject\model.pkl")
            decision_function_scores = model.decision_function(input_data)

            # Calculate percentages
            percentages = np.exp(decision_function_scores) / np.sum(np.exp(decision_function_scores), axis=1, keepdims=True)

            # Get the top 3 predicted courses
            top_3_courses_indices = decision_function_scores[0].argsort()[-3:][::-1]
            top_3_predicted_classes = model.classes_[top_3_courses_indices]

            course_mapping = {
                0: "BACHELOR OF ARTS IN ASIAN STUDIES",
                1: "BACHELOR OF ARTS IN BROADCASTING",
                2: "BACHELOR OF ARTS IN HISTORY",
                3: "BACHELOR OF ARTS IN ISLAMIC STUDIES",
                4: "BACHELOR OF ARTS IN JOURNALISM",
                5: "BACHELOR OF ARTS IN POLITICAL SCIENCE",
                6: "BACHELOR OF CULTURE AND ARTS EDUCATION",
                7: "BACHELOR OF EARLY CHILDHOOD EDUCATION",
                8: "BACHELOR OF ELEMENTARY EDUCATION",
                9: "BACHELOR OF PHYSICAL EDUCATION",
                10: "BACHELOR OF PUBLIC ADMINISTRATION",
                11: "BACHELOR OF SCIENCE IN ACCOUNTANCY",
                12: "BACHELOR OF SCIENCE IN ARCHITECTURE",
                13: "BACHELOR OF SCIENCE IN BIOLOGY",
                14: "BACHELOR OF SCIENCE IN CIVIL ENGINEERING",
                15: "BACHELOR OF SCIENCE IN COMMUNITY DEVELOPMENT",
                16: "BACHELOR OF SCIENCE IN COMPUTER ENGINEERING",
                17: "BACHELOR OF SCIENCE IN COMPUTER SCIENCE",
                18: "BACHELOR OF SCIENCE IN CRIMINOLOGY",
                19: "BACHELOR OF SCIENCE IN ECONOMICS",
                20: "BACHELOR OF SCIENCE IN ENVIRONMENTAL ENGINEERING",
                21: "BACHELOR OF SCIENCE IN EXERCISE AND SPORTS SCIENCES",
                22: "BACHELOR OF SCIENCE IN GEODETIC ENGINEERING",
                23: "BACHELOR OF SCIENCE IN HOME ECONOMICS",
                24: "BACHELOR OF SCIENCE IN HOSPITALITY MANAGEMENT",
                25: "BACHELOR OF SCIENCE IN INDUSTRIAL ENGINEERING",
                26: "BACHELOR OF SCIENCE IN INFORMATION TECHNOLOGY",
                27: "BACHELOR OF SCIENCE IN MECHANICAL ENGINEERING",
                28: "BACHELOR OF SCIENCE IN NURSING",
                29: "BACHELOR OF SCIENCE IN PSYCHOLOGY",
                30: "BACHELOR OF SCIENCE IN SANITARY ENGINEERING",
                31: "BACHELOR OF SCIENCE IN SOCIAL WORK",
                32: "BACHELOR OF SECONDARY EDUCATION",
                33: "BACHELOR OF SPECIAL NEED EDUCATION",
                34: "BATSILYER NG SINING SA FILIPINO",
            }

            # Calculate percentages for the top 3 courses
            top_3_percentages = percentages[0, top_3_courses_indices]

            # Determine the label for the course with the highest percentage
            highest_percentage_index = np.argmax(top_3_percentages)
            labels = [
                f"Highly Recommended! ({int(percentage * 100)}%)" if i == highest_percentage_index
                else f"({int(percentage * 100)}%)"
                for i, percentage in enumerate(top_3_percentages)
            ]

            top_3_predicted_courses_with_description = [
                (course_mapping[course], f"Based on your CET {cet}, GPA {gpa}, and SHS Strand {strand_text}", label)
                for course, label in zip(top_3_predicted_classes, labels)
            ]

            top_3_predicted_classes = [course_mapping[course_num] for course_num in top_3_predicted_classes]

            course_container = ""
            for course in top_3_predicted_classes:
                course_container += course + '|'

            courses = course_container[:-1].strip().split('|')

            pred_result = PredResults(
                first_name=first_name,
                last_name=last_name,
                sex=sex,
                cet=cet,
                gpa=gpa,
                strand=strand_text,
                recommended_course=courses,
            )
            pred_result.save()

            

            # Use OpenAI GPT-3 to generate analysis
            openai.api_key = settings.OPENAI_API_KEY

            analyses = []
            for course in top_3_predicted_classes:
                prompt_for_course = f"{input_data} Give an analysis. As result of your Academic background and CET OAPR it shows this {course} has {labels} shown a pattern of students having the similar gpa, cet, and strand, hence the {strand_text} input aligns with the course. Explain in 2-3 sentences why this course is suitable base on the user inputs of their CET, GPA, and SHS Strand and short description of the course."

                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt_for_course,
                    max_tokens=250,
                    temperature=0.5,
                )

                # Append each analysis inside the loop
                analyses.append(response['choices'][0]['text'])

            course_index = top_3_predicted_classes.index(course)
            course_description = top_3_predicted_courses_with_description[course_index][1]
            percentage = top_3_predicted_courses_with_description[course_index][2]
            
            for (course, description, percentage), analysis in zip(top_3_predicted_courses_with_description, analyses):
                recommended_course = RecommendedCourse(
                    prediction_id=pred_result,
                    course=course,
                    percentage=percentage,
                    description=description,
                    analysis=analysis,  # Save the analysis in the analysis field
                )
                recommended_course.save()

            # Pass the variables to the template
            return render(request, 'result.html', {
            'analysis': analyses,
            'recommended_courses_with_description': top_3_predicted_courses_with_description,
            'first_name': first_name,
            'last_name': last_name,
            'sex': sex,
            'cet': cet,
            'gpa': gpa,
            'strand': strand_text,
            'prediction_id': pred_result.id,
            'title': 'Result',
            'messages': messages,
        })

    else:
        form = PredictionForm()

    return render(request, 'recommend.html', {'form': form, 'title': 'Recommend'})



def pdf(request, id):

    current_date = datetime.datetime.now().strftime('%B %d, %Y')
    
    # Assuming your HTML file is stored in the 'templates' directory
    template_path = 'pdf_template.html'

    # Get the required data from the database or wherever it's stored
    prediction = PredResults.objects.get(id=id)
    recommended = RecommendedCourse.objects.filter(prediction_id=prediction)

    # Render the template with context data if needed
    context = {'prediction': prediction, 'recommendeds': recommended, 'current_date': current_date}

    # Create a WeasyPrint HTML object
    html = HTML(string=render(request, template_path, context).content)

    # Generate PDF
    pdf_file = html.write_pdf()

    # Create a Django HttpResponse with the PDF content
    response = HttpResponse(pdf_file, content_type='application/pdf')

    # Optionally, you can set the Content-Disposition header to force download
    response['Content-Disposition'] = 'filename="output.pdf"'

    return response