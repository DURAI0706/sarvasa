from urllib.parse import quote_plus
from flask import Flask, render_template, request, redirect, flash, session, url_for
from pymongo import MongoClient
from bson.objectid import ObjectId
from transformers import PreTrainedTokenizerFast, TFBertForSequenceClassification
import os
import bcrypt  
import joblib
from transformers import AutoTokenizer, TFAutoModel
from flask import jsonify
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.models import Model
from flask import render_template, redirect, session, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from collections import Counter
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta, timezone
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
import skfuzzy as fuzz
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from keras.utils import to_categorical
import matplotlib.pyplot as plt 
from datetime import timedelta
app = Flask(__name__)
app.secret_key = 'sawq#@21'




# Connect to MongoDB with the correct database name
connection_string = f"mongodb+srv://hackers:hackers123@psg.kmis61j.mongodb.net/"
# Initialize the MongoClient with the connection string
client = MongoClient(connection_string)
db = client['gamification']
student_collection = db['student']
recommendation_collection = db['recommendation'] 
teacher_collection= db['teacher']# Create a new collection for storing recommendations
courses_collection=db['courses']
recommendation_collection=db['recommendation']
mycourses_collection=db['mycourses']
quiz_collection=db['quiz']
assignment_collection=db['assignment']
student_test_collection= db['quizsubmission']
app.permanent_session_lifetime = timedelta(minutes=30)
# Routes


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/role', methods=['GET', 'POST'])
def role():
    if request.method == 'POST':
        selected_role = request.form['role']
        session['selected_role'] = selected_role
        if selected_role == 'student':
            return redirect('/studentlogin')  # Redirect to student registration
        elif selected_role == 'teacher':
            return redirect('/teacherlogin')
    return render_template('role.html')

@app.route('/studentregister', methods=['GET', 'POST'])
def student_register():
    if request.method == 'POST':
        name = request.form.get('name')
        register_number = request.form.get('register_number')
        roll_number = request.form.get('roll_number')
        department = request.form.get('department')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validate email format
        if not email.endswith('student.tce.edu'):
            flash('Invalid email format')
            return redirect('/studentregister')

        # Check if the email already exists
        existing_user = student_collection.find_one({'email': email})
        if existing_user:
            flash('Email already exists')
            return redirect('/studentregister')

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match')
            return redirect('/studentregister')

        # Hash the password before storing in the database
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Prepare user data
        user_data = {
            'name': name,
            'register_number': register_number,
            'roll_number': roll_number,
            'department': department,
            'email': email,
            'password': hashed_password,
            'role': 'student',
            'total_coins':500,
            'total_keys' :0,
            'total_heart' :0
        }

        # Insert user data into the database
        result = student_collection.insert_one(user_data)

        # Check if registration is successful
                # Check if registration is successful
        if result.inserted_id:
            session['email'] = email
        return redirect('/recommendation')
    return render_template('studentregister.html')

@app.route('/studentlogin', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Find the student with the provided email and role
        student = student_collection.find_one({'email': email, 'role': 'student'})
        
        # Check if the student exists and the password is correct
        if student and bcrypt.checkpw(password.encode('utf-8'), student['password']):
            # Store user ID in the session
            session['user_id'] = str(student['_id'])
            
            # Fetch user data from the student collection
            student_data = student_collection.find_one({'email': email})

            if student_data:
                print("Total Coins:", student_data.get('total_coins', 'Not Found'))
                
                # Prepare template context
                template_context = {
                    'name': student_data['name'],
                    'role': student_data['role'],
                    'email': student_data['email'],
                    'department': student_data['department'],
                    'roll_number': student_data['roll_number'],
                    'register_number': student_data['register_number'],
                    'total_coins': student_data.get('total_coins', 0),
                    'total_keys': student_data.get('total_keys', 0),
                    'total_heart': student_data.get('total_heart', 0)
                }
                
                # Set session['student_email'] to the email variable
                session['student_email'] = email
                
                # Fetch recommendation result from the recommendation collection
                recommendation_data = recommendation_collection.find_one({'user_id': email})
                recommendation_result = recommendation_data.get('recommendation') if recommendation_data else None

                if recommendation_result:
                    # You can use the recommendation result as needed
                    print(f'Recommendation for {email}: {recommendation_result}')
                    if recommendation_result == 'based on above we recommend you horror theme to nourish and to grow':
                        return render_template('horror_dashboard.html', student_data=template_context)
                    elif recommendation_result == 'based on above we recommend you nature content theme to nourish and to grow':
                        return render_template('natural_dashboard.html', student_data=template_context)
                    else:
                        return render_template('fantasy_dashboard.html', student_data=template_context)

            return redirect('/studentlogin')  # Redirect to student dashboard
        
    return render_template('studentlogin.html')


def map_fn(raw_values):
    column_mapping = {
        'Gender': 'Gender',
        'I typically play videogames /online games': 'a',
        'I prefer the following way of playing video/online games:': 'b',
        'Exploring hidden treasures': 'c',
        "Finding what's behind a locked door": 'd',
        'Giving Hint to others': 'e',
        'Picking up every single collectible in an area': 'f',
        'Thirst to crack more challenges': 'g',
        'Improve myself by self competing': 'h',
        'Playing in a group': 'i',
        'Cooperating with strangers': 'j',
        'Share your success with others': 'k',
        'Taking on a strong opponent when playing one to one match': 'l',
        'Getting 100% (completing everything in a game)': 'm',
        'Giving more importance to achieving goals': 'n',
        'I understand something better after I': 'o',
        'When I have to work on a group project, I first want to': 'p',
        'I more easily remember': 'q',
        'When I am doing long calculations,': 'r',
        'When I am reading for enjoyment, I like writers to': 's',
        'When I have to perform a task, I prefer to': 't',
        'When someone is showing me data, I prefer': 'u',
        'When I am learning a new subject, I prefer to': 'v',
        'When considering a body of information, I am more likely to': 'w',
        'When I solve math problems': 'x',
        'My attitude to videogame stories is:': 'y',
        'Do you like to do time-bound challenges?': 'z',
        'Choose a theme you like the most': 'aa',
        'Which design of points you like the most?': 'bb',
        'If you cross an important milestone in a game, Which design of Badges you like the most?': 'cc',
        'If you want a leaderboard for your game scores, which of the designs you like the most?': 'dd',
        'When you want to monitor your progress in a game and unlock the next level, Which design of levels you like the most?': 'ee',
        'Which Appreciation Animation that you would like to have when you complete a task?': 'ff',
        'Do you love it, if you see these many game elements into your learning Management system?': 'gg'
        # Add other column mappings as needed
    }

    preferences_mappings = {
        'Gender': {'Male': 1, 'Female': 2},
        'a': {'Every day': 1, 'Every week': 2, 'Occasionally': 3, 'Rarely': 4, 'Never': 5},
        'b': {'Single Player Alone': 1, 'Single Player with Other Player Helping': 2, 'Multiplayer': 3},
        'c': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'd': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'e': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'f': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'g': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'h': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'i': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'j': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'k': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'l': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'm': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'n': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'o': {'Try It Out': 1, 'Think It Through': 2},
        'p': {"Have 'group brainstorming' where everyone contributes ideas.": 1, 'Brainstorm individually and then come together as a group to compare ideas': 2},
        'q': {'Something I have done.': 1, 'Something I have thought a lot about': 2},
        'r': {'I tend to repeat all my steps and check my work carefully.': 1, 'I find checking my work tiresome and have to force myself to do it': 2},
        's': {'Clearly say what they mean.': 1, 'Say things in creative, interesting ways': 2},
        't': {'Master one way of doing it.': 1, 'Come up with new ways of doing it': 2},
        'u': {'Charts or graphs.': 1, 'Text summarizing the results': 2},
        'v': {'Stay focused on that subject, learning as much about it as I can.': 1, 'Try to make connections between that subject and related subjects': 2},
        'w': {'Focus on details and miss the big picture.': 1, 'Try to understand the big picture before getting into the details': 2},
        'x': {'I usually work my way to the solutions one step at a time.': 1, 'I often just see the solutions but then have to struggle to figure out the steps to get to them': 2},
        'y': {'Stories help me enjoy a videogame.': 1, 'Stories are not important to me in videogames': 2, 'I prefer videogames without stories': 3},
        'z': {'Yes': 1, 'No': 0},
        'aa': {'Alien': 1, 'Living location': 2, 'Nature connected': 3, 'Life style': 4, 'Fantasy': 5, 'Horror': 6},
        'bb': {'1': 1, '2': 2, '3': 3, '4': 4},
        'cc': {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, 'I do not wish for any badges': 0},
        'dd': {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'I do not need a leaderboard': 0},
        'ee': {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, 'I do not want any special indication for levels': 0},
        'ff': {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6},
        'gg': {'Yes': 1, 'No': 2},
    }

    mapped_values = {}
    for column, value in raw_values.items():
        if column in column_mapping:
            mapped_values[column_mapping[column]] = value

    for column, value in mapped_values.items():
        if column in preferences_mappings:
            mapped_values[column] = preferences_mappings[column].get(value, 0)  # Use 0 for unknown values

    return mapped_values


@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if 'email' in session:
        email = session['email']
        personal_details = student_collection.find_one({'email': email})

        if personal_details:
            if request.method == 'POST':
                # Get the user-entered values from the recommendation form
                user_preferences = {
                    'Gender': request.form.get('Gender'),
                    'a': request.form.get('PlayFrequency'),
                    'b': request.form.get('PlayPreference'),
                    'c': request.form.get('ExploringHiddenTreasures'),
                    'd': request.form.get('FindingBehindLockedDoor'),
                    'e': request.form.get('GivingHintToOthers'),
                    'f': request.form.get('PickingUpCollectibles'),
                    'g': request.form.get('ThirstToCrackChallenges'),
                    'h': request.form.get('ImproveMyselfSelfCompeting'),
                    'i': request.form.get('PlayingInAGroup'),
                    'j': request.form.get('CooperatingWithStrangers'),
                    'k': request.form.get('ShareYourSuccess'),
                    'l': request.form.get('TakingOnStrongOpponent'),
                    'm': request.form.get('Getting100'),
                    'n': request.form.get('Givingimportance'),
                    'o': request.form.get('UnderstandBetter'),
                    'p': request.form.get('GroupProject'),
                    'q': request.form.get('EasilyRemember'),
                    'r': request.form.get('LongCalculations'),
                    's': request.form.get('ReadingPreference'),
                    't': request.form.get('TaskPerformance'),
                    'u': request.form.get('DataPresentation'),
                    'v': request.form.get('LearningPreference'),
                    'w': request.form.get('InformationConsideration'),
                    'x': request.form.get('MathProblemSolving'),
                    'y': request.form.get('VideogameStories'),
                    'z': request.form.get('TimeBoundChallenges'),
                    'aa': request.form.get('FavoriteTheme'),
                    'bb': request.form.get('PointsDesign'),
                    'cc': request.form.get('BadgeDesign'),
                    'dd': request.form.get('LeaderboardDesign'),
                    'ee': request.form.get('LevelDesign'),
                    'ff': request.form.get('AppreciationAnimation'),
                    'gg': request.form.get('GameElementsInLMS'),
                    # Add more mappings as needed
                }                
                cursor = recommendation_collection.find(
                    {"Your suggestion for designing an immersive gaming experience": {"$ne": "No idea"}},
                    {"_id": 0, "Your suggestion for designing an immersive gaming experience": 0}
                ).limit(208)

                # Fetching the data as a list
                entries_from_database = list(cursor)
                
                mapped_entries = [map_fn(entry) for entry in entries_from_database]
                
                # Call the recommendation function with user preferences
                recommendation_result = perform_recommendation(user_preferences, mapped_entries)
                
                save_recommendation_to_db(email, user_preferences, recommendation_result)
            
                return redirect(url_for('show_recommendation', recommendation_result=recommendation_result))

                
                
                session['recommendation_result'] = recommendation_result
               
                
                
                # Save the recommendation result to the session
                
                           
            # Render the recommendation form
            return render_template('recommendation.html', personal_details=personal_details)

    return redirect('/login')



# Your recommendation logic function
def perform_recommendation(user_preferences, mapped_entries):
    # Create DataFrames for user preferences and mapped entries
    df_user = pd.DataFrame([user_preferences])

    # Construct a DataFrame using mapped entries
    preferences_mappings = {
        'Gender': {'Male': 1, 'Female': 2},
        'a': {'Every day': 1, 'Every week': 2, 'Occasionally': 3, 'Rarely': 4, 'Never': 5},
        'b': {'Single Player Alone': 1, 'Single Player with Other Player Helping': 2, 'Multiplayer': 3},
        'c': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'd': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'e': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'f': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'g': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'h': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'i': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'j': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'k': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'l': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'm': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'n': {'Love It': 1, 'Like It': 2, 'Dislike It': 3, 'Hate It': 4},
        'o': {'Try It Out': 1, 'Think It Through': 2},
        'p': {"Have 'group brainstorming' where everyone contributes ideas.": 1, 'Brainstorm individually and then come together as a group to compare ideas': 2},
        'q': {'Something I have done.': 1, 'Something I have thought a lot about': 2},
        'r': {'I tend to repeat all my steps and check my work carefully.': 1, 'I find checking my work tiresome and have to force myself to do it.': 2},
        's': {'Clearly say what they mean.': 1, 'Say things in creative, interesting ways.': 2},
        't': {'Master one way of doing it.': 1, 'Come up with new ways of doing it.': 2},
        'u': {'Charts or graphs.': 1, 'Text summarizing the results.': 2},
        'v': {'Stay focused on that subject, learning as much about it as I can.': 1, 'Try to make connections between that subject and related subjects.': 2},
        'w': {'Focus on details and miss the big picture.': 1, 'Try to understand the big picture before getting into the details.': 2},
        'x': {'I usually work my way to the solutions one step at a time.': 1, 'I often just see the solutions but then have to struggle to figure out the steps to get to them.': 2},
        'y': {'Stories help me enjoy a videogame.': 1, 'Stories are not important to me in videogames.': 2, 'I prefer videogames without stories.': 3},
        'z': {'Yes': 1, 'No': 0},
        'aa': {'Alien': 1, 'Living location': 2, 'Nature connected': 3, 'Life style': 4, 'Fantasy': 5, 'Horror': 6},
        'bb': {'1': 1, '2': 2, '3': 3, '4': 4},
        'cc': {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, 'I do not wish for any badges': 0},
        'dd': {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'I do not need a leaderboard': 0},
        'ee': {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, 'I do not want any special indication for levels': 0},
        'ff': {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6},
        'gg': {'Yes': 1, 'No':2},        
    }

    # Map the values in the DataFrame
    for column, mapping in preferences_mappings.items():
        df_user[column] = df_user[column].map(mapping)
    
    df_entries = pd.DataFrame(mapped_entries)
    
    df_combined = pd.concat([df_user, df_entries], ignore_index=True)

    df_combined.fillna(0, inplace=True)

    # Select the input variables (features)
    input_columns = df_combined.columns[1:]  # Exclude the 'Gender' column

    # Standardize the data (important for K-means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_combined[input_columns])

    try:
        # Apply Fuzzy C-Means clustering
        n_clusters = 4
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_scaled.transpose(), c=n_clusters, m=4, error=0.005, maxiter=1000, init=None)

        # Assign the cluster with the highest membership to the user
        user_cluster = np.argmax(u, axis=0)[0]

        # Print cluster memberships
               # Return the recommendation based on the user's cluster
        if user_cluster == 4:
            recommendation = "based on your obs we recommend you fantasy theme to nourish and to grow"
        elif user_cluster == 3:
            recommendation = "based on above we recommend you nature content theme to nourish and to grow"
            
        else:
            recommendation = "based on above we recommend you horror theme to nourish and to grow"
        
        print(recommendation)
        return recommendation

    except Exception as e:
        print(f"Error during clustering: {e}")
        return "Unable to provide a recommendation at the moment. Please check your preferences and try again."




def save_recommendation_to_db(email, user_preferences, recommendation):
    # Save the user's input and the recommendation to the database
    recommendation_data = {
        'user_id': email,
        'user_preferences': user_preferences,
        'recommendation': recommendation,
           }
    recommendation_collection.insert_one(recommendation_data)

@app.route('/show_recommendation')
def show_recommendation():
    # Retrieve the recommendation result from the URL parameter
    recommendation_result = request.args.get('recommendation_result')
    session.pop('recommendation_result', None)

    # Render a template to display the recommendation result
    return render_template('recommendation_result.html', recommendation_result=recommendation_result)

@app.route('/horror_dashboard')
def horror_dashboard():
    if 'email' in session:
        # Retrieve additional information from students_collection based on the email
        student_data = student_collection.find_one({'email': session['email']})
        if student_data:
            print("Student Data:", student_data)
            print("Total Coins:", student_data.get('total_coins', 'Not Found'))
            template_context = {
                'name': student_data['name'],
                'role': student_data['role'],
                'email': student_data['email'],
                'department': student_data['department'],
                'roll_number': student_data['roll_number'],
                'register_number': student_data['register_number'],
                'total_coins': student_data.get('total_coins', 0),
                'total_keys': student_data.get('total_keys', 0),
                'total_heart': student_data.get('total_heart', 0)
            }
            return render_template('horror_dashboard.html', student_data=template_context)
    return redirect(url_for('studentlogin'))

@app.route('/natural_dashboard')
def natural_dashboard():
    if 'email' in session:
        # Retrieve additional information from students_collection based on the email
        student_data = student_collection.find_one({'email': session['email']})
        if student_data:
            # Map the retrieved data to the template context
            template_context = {
                'name': student_data['name'],
                'role': student_data['role'],
                'email': student_data['email'],
                'department': student_data['department'],
                'roll_number': student_data['roll_number'],
                'register_number': student_data['register_number'],
                'total_coins': student_data.get('total_coins', 0),
                'total_keys': student_data.get('total_keys', 0),
                'total_heart': student_data.get('total_heart', 0)
            }
            return render_template('natural_dashboard.html', student_data=template_context)
    return redirect(url_for('studentlogin'))
    

@app.route('/fantasy_dashboard')
def fantasy_dashboard():
    if 'email' in session:
        # Retrieve additional information from students_collection based on the email
        student_data = student_collection.find_one({'email': session['email']})
        if student_data:
            print("Total Coins:", student_data.get('total_coins', 'Not Found'))
            template_context = {
                'name': student_data['name'],
                'role': student_data['role'],
                'email': student_data['email'],
                'department': student_data['department'],
                'roll_number': student_data['roll_number'],
                'register_number': student_data['register_number'],
                'total_coins': student_data.get('total_coins', 0),
                'total_keys': student_data.get('total_keys', 0),
                'total_heart': student_data.get('total_heart', 0)
            }
            return render_template('fantasy_dashboard.html', student_data=template_context)
        

    return redirect(url_for('studentlogin'))

# Your other imports and route handlers

@app.route('/mycourses/student', methods=['GET', 'POST'])
def student_courses():
    # Check if 'email' and 'student_email' are in the session
    if 'email' in session and 'student_email' in session:
        # Both emails are present, handle accordingly
        email = session['email']
        student_email = session['student_email']
        # You may want to log a warning or raise an exception here,
        # or you could choose to use both emails in your logic

    elif 'email' in session:
        # Retrieve the user email from the session
        email = session['email']
        student_email = None

    elif 'student_email' in session:
        # Retrieve the student email from the session
        email = None
        student_email = session['student_email']

    else:
        # Redirect to login if none of the emails is in the session
        return redirect('/role')

    # Fetch student details from the MongoDB collection
    student_details = student_collection.find_one({'email': email or student_email})
    if not student_details:
        # Handle the case where student details are not found
        return redirect('/role')

    # Extract department information from student details
    department = student_details.get('department')

    # Fetch all courses from the MongoDB collection
    all_courses = list(courses_collection.find())

    # Fetch all courses joined by the user from the MongoDB collection
    joined_courses = mycourses_collection.find({'user_email': email or student_email})

    # Extract course names from joined courses
    joined_course_names = [joined_course['course_name'] for joined_course in joined_courses]

    # Separate joined and other courses in the same department
    my_courses = [course for course in all_courses if course['name'] in joined_course_names and course['department'] == department]
    other_courses = [course for course in all_courses if course['name'] not in joined_course_names and course['department'] == department]

    # Render the template with joined and other courses
    return render_template('mycoursehorror.html', my_courses=my_courses, other_courses=other_courses)




    
@app.route('/verify_join', methods=['POST'])
def verify_join():
    # Retrieve the entered password and course name from the form
    entered_password = request.form.get('enteredPassword')
    course_name = request.form.get('courseName')

    # Retrieve the user email from the session
    user_email = session.get('email') or session.get('student_email')

    # Print session and form variables for debugging
    print("Session Variables in verify_join:", session)
    print("Form Variables - Password:", entered_password)
    print("Form Variables - Course Name:", course_name)
    print("Form Variables - User Email:", user_email)

    # Check if the entered password, course name, and user email are available
    if entered_password and course_name and user_email:
        # Debug statement to check control flow
        print("Inside if condition")

        # Retrieve the course details from the MongoDB collection based on the course name
        course = courses_collection.find_one({'name': course_name})

        # Check if the entered password matches the course password
        if course and entered_password == course.get('password'):
            # Store the course in the student database using insert_one
            mycourses_collection.insert_one({
                'user_email': user_email,
                'course_name': course_name,
                'created_by': course.get('createdBy'),
                'description': course.get('description')
            })

            # Debug statement to check control flow
            print("Redirecting to mycourses/student")

            # Redirect to the student courses page
            return redirect('/mycourses/student')

    # Debug statement to check control flow
    print("Redirecting to login page")

    # Render an error template or redirect to an appropriate route
    # For example, you can redirect to the login page or render an error template
    return redirect(url_for('role'))


@app.route('/course/<course_name>', methods=['GET'])
def individual_course(course_name):
    # Retrieve the user email from the session
    user_email = session.get('email') or session.get('student_email')

    # Check if the user is registered for the specified course
    joined_course = mycourses_collection.find_one({
        'user_email': user_email,
        'course_name': course_name
    })

    if not joined_course:
        # Redirect to the main courses page or display an error message
        return redirect(url_for('student_courses'))

    # Retrieve the recommendation result for the user
    recommendation_data = recommendation_collection.find_one({'user_id': user_email})
    recommendation_result = recommendation_data.get('recommendation') if recommendation_data else None

    # Render the individual course page based on the recommendation result
    if recommendation_result == 'based on above we recommend you horror theme to nourish and to grow':
        return render_template('individual_course_horror.html', course=joined_course)
    elif recommendation_result == 'based on above we recommend you nature content theme to nourish and to grow':
        return render_template('individual_course_nature.html', course=joined_course)
    else:
        return render_template('individual_course_fantasy.html', course=joined_course)

def get_quizzes(course_name):
    quiz_collection = db['quiz']
    return list(quiz_collection.find({'course_name': course_name}))

def get_quizzes(course_name):
    quiz_collection = db['quiz']
    return list(quiz_collection.find({'course_name': course_name}))

@app.route('/see_quizzes/<course_name>', methods=['GET'])
def see_quizzes(course_name):
    # Retrieve quizzes for the specified course from MongoDB
    course_quizzes = get_quizzes(course_name)

    # Retrieve the recommendation result for the user from the database
    user_email = session.get('email') or session.get('student_email')
    recommendation_data = recommendation_collection.find_one({'user_id': user_email})
    recommendation_result = recommendation_data.get('recommendation') if recommendation_data else None

    # Check if the user has attended each quiz and categorize them
    completed_tests = []
    non_completed_tests = []

    # Retrieve user email from the session
    user_email = session.get('email') or session.get('student_email')

    for quiz in course_quizzes:
        # Check if the user has submitted a quiz
        quiz_submission = student_test_collection.find_one({
            'user_email': user_email,
            'course_name': course_name,
            'quiz_name': quiz['quiz_name']
        })

        if quiz_submission:
            # Check if the user's score is greater than the conditional marks
            if quiz_submission['marks'] >= quiz['condition_marks']:
                completed_tests.append({'quiz_name': quiz['quiz_name'], 'quiz_submission': quiz_submission})
            else:
                non_completed_tests.append({'quiz_name': quiz['quiz_name'], 'quiz_submission': quiz_submission})
        else:
            # User has not attended the quiz
            non_completed_tests.append({'quiz_name': quiz['quiz_name'], 'quiz_submission': None})

    print("Completed Tests:", completed_tests)
    print("Non-Completed Tests:", non_completed_tests)

    # Render the template based on the recommendation result
    if recommendation_result == 'based on above we recommend you horror theme to nourish and to grow':
        return render_template('student_quizzes_horror.html', course_name=course_name, 
                               completed_tests=completed_tests, non_completed_tests=non_completed_tests)
    elif recommendation_result == 'based on above we recommend you nature content theme to nourish and to grow':
        return render_template('student_quizzes_nature.html', course_name=course_name, 
                               completed_tests=completed_tests, non_completed_tests=non_completed_tests)
    else:
        return render_template('student_quizzes_fantasy.html', course_name=course_name, 
                               completed_tests=completed_tests, non_completed_tests=non_completed_tests)


        
@app.route('/see_assignments/<course_name>', methods=['GET'])
def see_assignments(course_name):

    return render_template(template_name, course_name=course_name, quizzes=course_quizzes)


@app.route('/see_materials/<course_name>', methods=['GET'])
def see_materials(course_name):

    return render_template(template_name, course_name=course_name, quizzes=course_quizzes)

@app.route('/start_quiz/<course_name>/<quiz_name>', methods=['POST'])
def start_quiz(course_name, quiz_name):
    # Retrieve user email from the session
    user_email = session.get('email') or session.get('student_email')
    

    return render_template('quiz_page.html', course_name=course_name, quiz_name=quiz_name, questions=quiz_questions)

    
@app.route('/check_pin', methods=['POST'])
def check_pin():
    entered_pin = request.form.get('pin')
    # Check if the entered PIN is correct (you can replace '123456' with your actual PIN)
    if entered_pin == '123456':
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid PIN'})

# Update the teacher_register route
@app.route('/teacherregister', methods=['GET', 'POST'])
def teacher_register():
    if request.method == 'POST':
        name = request.form.get('name')
        register_number = request.form.get('register_number')
        roll_number = request.form.get('roll_number')
        department = request.form.get('department')
        email_id = request.form.get('email_id')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate email format
        if not email_id.endswith('tce.edu'):
            flash('Invalid email format')
            return redirect('/teacherregister')
        
        # Check if the email already exists
        existing_user = teacher_collection.find_one({'email_id': email_id})
        if existing_user:
            flash('Email already exists')
            return redirect('/teacherregister')
        
        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match')
            return redirect('/teacherregister')
        
        # Hash the password before storing in the database
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Prepare user data
        user_data = {
            'name': name,
            'register_number': register_number,
            'roll_number': roll_number,
            'department': department,
            'email_id': email_id,
            'password': hashed_password,
            'role': 'teacher',
        }
        
        # Insert user data into the database
        result = teacher_collection.insert_one(user_data)
        
        # Check if registration is successful
        if result.inserted_id:
            session['email_id'] = email_id
            session['department'] = department
            # Flash a success message
            flash('Registration successful! You can now log in.')
            
            # Redirect to the teacher dashboard page with the success message
            return redirect('/teacherdashboard')
    
    # Render the teacher registration template for GET requests
    return render_template('teacherregister.html')

@app.route('/teacherlogin', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Debug print
        print(f"Received login request for email: {email}")

        # Update the field name to 'email_id' in the database query
        teacher = teacher_collection.find_one({'email_id': email, 'role': 'teacher'})

        # Debug print

        if teacher and bcrypt.checkpw(password.encode('utf-8'), teacher['password']):
            session['teacher_email'] = email

            # Debug print
            print("Login successful. Redirecting to teacherdashboard.")

            return redirect('/teacherdashboard')

    return render_template('teacherlogin.html')

@app.route('/teacherdashboard')
def teacherdashboard():
    # Check if either 'email_id' or 'teacher_email' is in the session
    if 'email_id' in session or 'teacher_email' in session:
        # Create a list of emails by combining both 'email_id' and 'teacher_email'
        emails_list = [session.get('email_id'), session.get('teacher_email')]

        # Iterate through the list of emails
        for email in emails_list:
            # Check if the current email is not None
            if email:
                # Fetch teacher data from the teacher_collection based on email_id
                teacher_data = teacher_collection.find_one({'email_id': email})
                
                # Check if the teacher data is found
                if teacher_data:    
                    template_context = {
                        'name': teacher_data['name'],
                        'role': teacher_data['role'],
                        'email_id': teacher_data['email_id'],
                        'department': teacher_data['department'],
                    }
                    return render_template('teacherdashboard.html', teacher_data=template_context)

    # Redirect to the login page if neither 'email_id' nor 'teacher_email' is in the session
    return redirect(url_for('teacher_login'))

@app.route('/mycourses')
def my_courses():
    # Check if either 'email' or 'student_email' is in the session
    if 'email_id' in session or 'teacher_email' in session:
        # Create a list of emails by combining both 'email' and 'student_email'
        emails_list = [session.get('email_id'), session.get('teacher_email')]

        # Iterate through the list of emails
        for teacher_email in emails_list:
            # Check if the current email is not None
            if teacher_email:
                # Fetch department from teacher_collection based on email_id
                teacher_data = teacher_collection.find_one({'email_id': teacher_email})
                
                # Check if the teacher data is found
                if teacher_data:
                    department = teacher_data['department']
                    
                    # Fetch courses from the MongoDB collection based on the department and createdBy
                    teacher_courses = list(courses_collection.find({'createdBy': teacher_email, 'department': department}))
                    department_courses = list(courses_collection.find({
                        'department': department,
                        'createdBy': {'$ne': teacher_email}
                    }))
                    
                    # Render the template with the relevant data
                    return render_template('mycourses.html', teacher_courses=teacher_courses, department_courses=department_courses)

    # Redirect to login if the user is not logged in or if neither 'email' nor 'student_email' is in the session
    return redirect(url_for('teacherlogin'))

@app.route('/course_attributes/<course_name>')
def course_attributes(course_name):
    # Check if the user is logged in as a teacher
    if 'teacher_email' in session:
        teacher_email = session.get('teacher_email')

        # Fetch course details from the MongoDB collection based on name and createdBy
        course = courses_collection.find_one({'name': course_name, 'createdBy': teacher_email})

        if course:
            # Render the course attributes page with the relevant data
            return render_template('course_attributes.html', course=course)
        else:
            # Redirect to mycourses page if the course does not belong to the teacher
            return redirect(url_for('mycourses'))

    # Redirect to login if the user is not logged in as a teacher
    return redirect(url_for('teacherlogin'))

@app.route('/view_quiz/<course_name>')
def view_quiz(course_name):
    # Assuming quiz documents have a field 'course_name' that matches the course_name parameter
    quizzes = quiz_collection.find({'course_name': course_name})

    # Pass the quiz details to the template
    return render_template('view_quiz.html', course_name=course_name, quizzes=quizzes)


@app.route('/edit_quiz/<quiz_name>', methods=['GET', 'POST'])
def edit_quiz(quiz_name):
    # Fetch existing quiz details from the MongoDB collection based on quiz_name
    quiz = quiz_collection.find_one({'quiz_name': quiz_name})

    if request.method == 'POST':
        print(request.form)
        print(quiz_name) 

        # Update quiz details based on the form data
        updated_data = {
            "quiz_name": request.form['quiz_name'],
            "timer": int(request.form['timer']),
            "condition_marks": int(request.form['condition_marks']),
            "badges": request.form['badges'],
            "questions": []
        }

        # Extract dynamic question details from form data
        question_keys = [key for key in request.form.keys() if key.startswith('question_')]
        for key in question_keys:
            question_index = key.split('_')[-1]
            question = {
                "question": request.form[f'question_{question_index}'],
                "choices": request.form.getlist(f'choices_{question_index}[]'),
                "correct_answer": request.form[f'correctAnswer_{question_index}'],
                "hint": request.form[f'hint_{question_index}'],
                "type": request.form[f'type_{question_index}']
            }
            updated_data["questions"].append(question)

        # Update the quiz in the database
        result = quiz_collection.update_one({'quiz_name': quiz_name}, {'$set': updated_data})

        if result.modified_count > 0:
            # Quiz updated successfully, redirect to course_attributes page
            return redirect(url_for('course_attributes', course_name=quiz['course_name']))

    # Render the edit_quiz template with the existing quiz data
    return render_template('edit_quiz.html', quiz=quiz)

@app.route('/show_quiz/<course_name>')
def show_quiz(course_name):
    # Fetch quiz names for the specified course from the database
    # Assume quiz_collection is your MongoDB collection for quizzes
    quizzes = quiz_collection.find({'course_name': course_name}, {'quiz_name': 1, '_id': 0})

    # Render the template with the list of quiz names
    return render_template('show_quiz.html', course_name=course_name, quizzes=quizzes)


@app.route('/view_assignment/<course_name>/<quiz_name>')
def view_assignment(course_name, quiz_name):
    # Fetch the assignment details for the provided course_name and quiz_name
    assignments = assignment_collection.find({'course_name': course_name, 'quiz_name': quiz_name})

    # Pass the assignment details to the template
    return render_template('view_assignment.html', assignments=assignments)



@app.route('/edit_assignment/<assignment_name>', methods=['GET', 'POST'])
def edit_assignment(assignment_name):
    # Fetch existing assignment details from the MongoDB collection based on assignment_name
    assignment = assignment_collection.find_one({'assignment_name': assignment_name})

    if request.method == 'POST':
        # Update assignment details based on the form data
        updated_data = {
            "assignment_name": request.form['assignmentName'],
            "questions": []
        }

        # Extract dynamic question details from form data
        question_keys = [key for key in request.form.keys() if key.startswith('question_')]
        for key in question_keys:
            question_index = key.split('_')[-1]
            question = {
                "question": request.form[f'question_{question_index}']
            }
            updated_data["questions"].append(question)

        # Update the assignment in the database
        result = assignment_collection.update_one({'assignment_name': assignment_name}, {'$set': updated_data})

        if result.modified_count > 0:
            # Assignment updated successfully, redirect to course_attributes page
            return redirect(url_for('course_attributes', course_name=assignment['course_name']))
    
    # Render the edit_assignment template with the existing assignment data
    return render_template('edit_assignment.html', assignment=assignment)



@app.route('/createcourse', methods=['GET', 'POST'])
def create_course():
    # Check if either 'email_id' or 'teacher_email' is in the session
    if 'email_id' in session or 'teacher_email' in session:
        # Create a list of emails by combining both 'email_id' and 'teacher_email'
        emails_list = [session.get('email_id'), session.get('teacher_email')]

        # Iterate through the list of emails
        for teacher_email in emails_list:
            # Check if the current email is not None
            if teacher_email:
                # Fetch department from teacher_collection based on email_id
                teacher_data = teacher_collection.find_one({'email_id': teacher_email})
                
                # Check if the department is found
                if teacher_data and 'department' in teacher_data:
                    department = teacher_data['department']

                    if request.method == 'POST':
                        # Get course details from the form
                        name = request.form['title']
                        description = request.form['description']
                        code= request.form['code']
                        print('title',name)
                        print('description',description)
                        print('password',code)
                        print('createdBy',teacher_email)
                        print('department',department)
                        # Create a new course document
                        
                        new_course = {
                            'name': name,
                            'department': department,
                            'createdBy': teacher_email,
                            'description': description,
                            'password': code
                        }
                        session['name'] = name
                        print('session',session)
                        
                        # Insert the new course into the MongoDB collection
                        result = courses_collection.insert_one(new_course)

                        # Check if the insertion was successful
                        if result.inserted_id:
                            return redirect('/coursepage')  # Redirect to the courses page or any other page
                        else:
                            return render_template('createcourse.html', message='Failed to create the course')

                    return render_template('createcourse.html')  # Render the form to create a new course

        # Redirect to login if the user is not logged in or if neither 'email_id' nor 'teacher_email' is in the session
        return redirect(url_for('teacherlogin'))

    # Redirect to login if neither 'email_id' nor 'teacher_email' is in the session
    return redirect(url_for('teacherlogin'))

@app.route('/coursepage',methods=['GET', 'POST'])
def coursepage():
    if request.method == 'POST':
        course_type = request.form.get('course_type')

        # Redirect to the appropriate route based on the course type
        if course_type == 'assignment':
            return redirect('/submit_assignment')
        elif course_type == 'quiz':
            return redirect('/submit_quiz')
        else:
            # Handle the case where course_type is neither 'assignment' nor 'test'
            return render_template('createcourse.html', message='Invalid course type')
    
    return render_template('coursepage.html')

@app.route('/submit_quiz', methods=['GET', 'POST'])
def submit_quiz():
    success_message = None

    if request.method == 'POST':
        if 'email_id' in session or 'teacher_email' in session or 'name' in session:
            # Create a list of emails by combining both 'email_id' and 'teacher_email'
            emails_list = [session.get('email_id'), session.get('teacher_email')]
            name = session.get('name')

            # Iterate through the list of emails
            for teacher_email in emails_list:
                # Check if the current email is not None
                if teacher_email:
                    # Fetch department from teacher_collection based on email_id
                    teacher_data = teacher_collection.find_one({'email_id': teacher_email})

                    # Proceed with quiz data insertion into MongoDB collection
                    quiz_name = request.form['quizName']
                    quiz_data = {
                        "course_name": name,
                        "quiz_name": quiz_name,
                        "timer": int(request.form['timer']),
                        "condition_marks": int(request.form['mark']),
                        "badges": request.form['badges'],
                        "questions": []
                    }
                    session['quiz_name'] = quiz_name

                    # Extract dynamic question details from form data
                    question_keys = [key for key in request.form.keys() if key.startswith('question_')]
                    for key in question_keys:
                        question_index = key.split('_')[-1]
                        question = {
                            "question": request.form[f'question_{question_index}'],
                            "choices": request.form.getlist(f'choices_{question_index}[]'),
                            "correct_answer": request.form[f'correctAnswer_{question_index}'],
                            "hint": request.form[f'hint_{question_index}'],
                            "type": request.form[f'type_{question_index}']
                        }
                        quiz_data["questions"].append(question)

                    # Insert quiz data into MongoDB collection
                    result = quiz_collection.insert_one(quiz_data)

                    if result.inserted_id:
                        # Quiz successfully inserted, set success message
                        success_message = 'Quiz successfully inserted'
                    else:
                        # Insertion failed, set appropriate error message
                        success_message = 'Failed to insert the quiz'

    # Render the form for creating a new quiz with success message
    return render_template('submit_quiz.html', success_message=success_message)
    
@app.route('/submit_assignment', methods=['GET', 'POST'])
def submit_assignment():
    success_message = None

    if request.method == 'POST':
        if 'email_id' in session or 'teacher_email' in session or 'name' in session or 'quiz_name' in session:
            emails_list = [session.get('email_id'), session.get('teacher_email')]
            name = session.get('name')
            quiz_name = session.get('quiz_name')

            for teacher_email in emails_list:
                if teacher_email:
                    teacher_data = teacher_collection.find_one({'email_id': teacher_email})

                    assignment_data = {
                        "course_name": name,
                        "quiz_name": quiz_name,
                        "assignment_name": request.form.get('assignmentName'),  # Fix here
                        "questions": []
                    }

                    question_keys = [key for key in request.form.keys() if key.startswith('question_')]
                    for key in question_keys:
                        question_index = key.split('_')[-1]
                        question = {
                            "question": request.form[f'question_{question_index}']
                        }
                        assignment_data["questions"].append(question)

                    result = assignment_collection.insert_one(assignment_data)

                    if result.inserted_id:
                        # Assignment submitted successfully
                        success_message = 'Assignment successfully submitted'
                    else:
                        # Insertion failed, set appropriate error message
                        success_message = 'Failed to insert the quiz'


    # Render the form for submitting an assignment with success message if available
    return render_template('submit_assignment.html', success_message=success_message)

if __name__ == '__main__':
    app.run()
