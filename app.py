#################### IMPORT PYTHON PACKAGES ################################
from datetime import datetime, timedelta
import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import Ridge
import tensorflow as tf
from requests.auth import HTTPBasicAuth
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Base2,Base3,Base4,Entry, Result,Race,Runner,Runners,Racecard,Distance,Horse,Time, MergedData
from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from io import BytesIO
import seaborn as sns
from sklearn.model_selection import train_test_split
from joblib import load
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix, average_precision_score
from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import plot_precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import json
import pandas
from datetime import datetime, timedelta
import time
import csv
from requests.auth import HTTPBasicAuth
import re
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func, Column, Integer, String, Date, Float, UniqueConstraint, and_
from sqlalchemy.ext.declarative import declarative_base
import requests
from datetime import datetime


#################### API CRED & DATABASE SETUP ##############################
# Define API credentials
user = "mFQGORO0a0k8upIywXtNUvTO"
password = "eqzz8leY4tYY0PpBqlR6mxZ9"

# Initialize database
DATABASE_URL = "sqlite:///north_american.db"
engine = create_engine(DATABASE_URL)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

DATABASE_URL3 = "sqlite:///race_cards.db"
engine3 = create_engine(DATABASE_URL3)
Base3.metadata.create_all(engine3)  
Session3 = sessionmaker(bind=engine3)
session3 = Session3()

DATABASE_URL2 = "sqlite:///uk.db"
engine2 = create_engine(DATABASE_URL2)
Base2.metadata.create_all(engine2)  
Session2 = sessionmaker(bind=engine2)
session2 = Session2()


DATABASE_URL4 = "sqlite:///horse_analysis.db"
engine4 = create_engine(DATABASE_URL4)
Base4.metadata.create_all(engine4)  
Session4 = sessionmaker(bind=engine4)
session4 = Session4()


######################### FUNCTIONS FOR "RESULTS" API ############
def save_to_database(data, session):
    from datetime import datetime

    url = "https://api.theracingapi.com/v1/courses"

    # Get the course data
    response = requests.get(url, auth=HTTPBasicAuth(user, password))
    courses_data = response.json()
  

    # Filter courses by region 'GB'
    gb_courses = {course['id']: course['region'] for course in courses_data['courses'] if course['region_code'] == 'gb'}
    
    for race in data['results']:
        course_id = race.get('course_id', 'Unknown')
        if course_id not in gb_courses:
            # print(f"Skipping race_id {race.get('race_id')} as course_id {course_id} is not in GB region.")
            continue
        existing_race = session.query(Race).filter_by(race_id=race.get('race_id')).first()
        if existing_race:
            continue  # Skip if race already exists
        try:
            race_date = datetime.strptime(race.get('date', "1970-01-01"), "%Y-%m-%d")  # Parse date with format
        except ValueError:
            race_date = datetime(1970, 1, 1)  # Default date in case of error
        # Create a new race entry
     
        race_entry = Race(
            race_id=race.get('race_id'),  
            date=race_date,  
            region='GB',  
            course=race.get('course', 'Unknown'),  
            course_id=race.get('course_id', 'Unknown'),  
            off=race.get('off', 'Unknown'),  
            off_dt=race.get('off_dt','Unknown'),
            race_name=race.get('race_name', 'Unknown'),  
            race_type=race.get('type', 'Unknown'),  
            race_class=race.get('class', 'Unknown'),  
            pattern=race.get('pattern', 'Unknown'), 
            rating_band=race.get('rating_band', 'Unknown'),  
            age_band=race.get('age_band', 'Unknown'),  
            sex_rest=race.get('sex_rest', 'Unknown'),  
            dist=race.get('dist', 'Unknown'),
            dist_f=race.get('dist_f','Unknown'), 
            dist_y=race.get('dist_y','Unknown'),
            dist_m=race.get('dist_m', 'Unknown'),
            jumps=race.get('jumps','Unknown'),
            going=race.get('going', 'Unknown'),  
            non_runners=race.get('non_runners','Unknown'),
            winning_time_detail=race.get('winning_time_detail', 'Unknown'),  
            tote_win=race.get('tote_win', 'Unknown'),  
            tote_pl=race.get('tote_pl', 'Unknown'),  
            tote_ex=race.get('tote_ex', 'Unknown'),  
            tote_csf=race.get('tote_csf', 'Unknown'),  
            tote_tricast=race.get('tote_tricast', 'Unknown'), 
            tote_trifecta=race.get('tote_trifecta', 'Unknown'),
            comments=race.get('comments', 'No comment')
        )
        
        session.add(race_entry)  # Add race entry first to get race_id
        
        for runner in race.get('runners', []):  # Using get() to handle missing 'runners'
            runner_entry = Runner(
                horse_id=runner.get('horse_id', 'Unknown'),  # Default if missing
                horse=runner.get('horse', 'Unknown'),  # Default if missing
                sp=runner.get('sp', 'Unknown'),  # Changed to float for consistency
                sp_dec=runner.get('sp_dec', 'Unknown'),  # Kept as string for consistency
                number=runner.get('number', 'Unknown'),  # Changed to integer for consistency
                position=runner.get('position', 'Unknown'),  # Changed to integer for consistency
                draw=runner.get('draw', 'Unknown'),  # Changed to integer for consistency
                btn=runner.get('btn', 'Unknown'),  # Default if missing
                ovr_btn=runner.get('ovr_btn', 'Unknown'),  # Default if missing
                age=runner.get('age', 'Unknown'),  # Changed to integer for consistency
                sex=runner.get('sex', 'Unknown'),  # Default if missing
                weight=runner.get('weight', 'Unknown'),  # Changed to float for consistency
                weight_lbs=runner.get('weight_lbs', 'Unknown'),  # Changed to float for consistency
                headgear=runner.get('headgear', 'None'),  # Default if missing
                time=runner.get('time', 'Unknown'),  # Changed to string for consistency
                or_rating=runner.get('or','Unknown' ),  # Changed to float for consistency
                rpr=runner.get('rpr', 'Unknown'),  # Changed to float for consistency
                tsr=runner.get('tsr', 'Unknown'),  # Changed to float for consistency
                prize=runner.get('prize', 'Unknown'),  # Changed to float for consistency
                jockey=runner.get('jockey', 'Unknown'),  # Default if missing
                jockey_id=runner.get('jockey_id', 'Unknown'),  # Default if missing
                jockey_claim_lbs=runner.get('jockey_claim_lbs', 'Unknown'),
                trainer=runner.get('trainer', 'Unknown'),  # Default if missing
                trainer_id=runner.get('trainer_id', 'Unknown'),  # Default if missing
                owner=runner.get('owner', 'Unknown'),  # Default if missing
                owner_id=runner.get('owner_id', 'Unknown'),  # Default if missing
                sire=runner.get('sire', 'Unknown'),  # Default if missing
                sire_id=runner.get('sire_id', 'Unknown'),  # Default if missing
                dam=runner.get('dam', 'Unknown'),  # Default if missing
                dam_id=runner.get('dam_id', 'Unknown'),  # Default if missing
                damsire=runner.get('damsire', 'Unknown'),  # Default if missing
                damsire_id=runner.get('damsire_id', 'Unknown'),  # Default if missing  # Default if missing
                silk_url=runner.get('silk_url', 'Unknown'),  # Default if missing
                race_id=race_entry.race_id  # Set the foreign key
            )
            session.add(runner_entry)  # Add each runner entry

    session.commit()  # Commit the session after adding all entries

def fetch_data(limit, skip, start_date=None, end_date=None):
    url = "https://api.theracingapi.com/v1/results"
    params = {
        "limit": limit,
        "skip": skip,
    }
    time.sleep(0.5)
    # Add start_date and end_date to params if they are provided
    if start_date and end_date:
        params['start_date'] = start_date
        params['end_date'] = end_date
    resp = requests.get(url, params=params,
                                auth=HTTPBasicAuth(user, password))
    resp.raise_for_status()  # Raise an exception for HTTP errors
    return resp.json()
def process_uk_data(limit=50, start_date=None, end_date=None):
    skip = 0
    total_records = None
    while total_records is None or skip < total_records:
        data = fetch_data(limit, skip, start_date, end_date)
        total_records = data.get('total')
        # st.write("total records is",total_records)
        if total_records is None:
            print("No 'total' key in response. Exiting.")
            break
        if total_records == 0:
            print("No records found for the given date range, exiting.")
            break
        save_to_database(data, session2)
        skip += limit  
    session2.close()


# import aiohttp
# import asyncio
# from sqlalchemy.orm import sessionmaker
# from requests.auth import HTTPBasicAuth

# # Replace synchronous fetch_data with async version
# async def fetch_data_async(limit, skip, start_date=None, end_date=None):
#     url = "https://api.theracingapi.com/v1/results"
#     params = {
#         "limit": limit,
#         "skip": skip,
#     }
#     # Add start_date and end_date to params if they are provided
#     if start_date and end_date:
#         params['start_date'] = start_date
#         params['end_date'] = end_date

#     async with aiohttp.ClientSession(auth=aiohttp.BasicAuth(user, password)) as session:
#         async with session.get(url, params=params) as response:
#             response.raise_for_status()
#             return await response.json()

# # Asynchronous function to process the data
# async def process_uk_data_async(limit=50, start_date=None, end_date=None):
#     skip = 0
#     total_records = None
#     tasks = []  # List to store async tasks
#     while total_records is None or skip < total_records:
#         data = await fetch_data_async(limit, skip, start_date, end_date)
#         total_records = data.get('total')
        
#         if total_records is None:
#             print("No 'total' key in response. Exiting.")
#             break
#         if total_records == 0:
#             print("No records found for the given date range, exiting.")
#             break
        
#         # Instead of saving directly, create tasks for saving to DB asynchronously
#         tasks.append(asyncio.to_thread(save_to_database, data, session2))
#         skip += limit

#     await asyncio.gather(*tasks)  # Run all save_to_database tasks concurrently

# async def main():
#     # Run the async processing of UK data
#     await process_uk_data_async(limit=50, start_date="2018-01-01", end_date="2024-10-17")




def query_and_merge_RESULTS_data():
    # Fetch entries and results into pandas DataFrames
    entries_df = pd.read_sql_table('races', con=engine2,)
    results_df = pd.read_sql_table('runners', con=engine2)
    # Merge entries and results DataFrames using pd.merge()
    merged_df = pd.merge(entries_df, results_df, on=['race_id'], how='outer')
    return merged_df

################# FUNCTONS FOR NEW RACECARD API ########################

# def save_to_database_racecards(data, session):
#     from datetime import datetime
#     url = "https://api.theracingapi.com/v1/courses"
#     response = requests.get(url, auth=HTTPBasicAuth(user, password))
#     courses_data = response.json()

#     gb_courses = {course['id']: course['region'] for course in courses_data['courses'] if course['region_code'] == 'gb'}
    
#     for race in data.get('pro',[]):
#         course_id = race.get('course_id', 'Unknown')
#         if course_id not in gb_courses:
#             # print(f"Skipping race_id {race.get('race_id')} as course_id {course_id} is not in GB region.")
#             continue
#         existing_race = session.query(Race).filter_by(race_id=race.get('race_id')).first()
#         if existing_race:
#             continue  # Skip if race already exists
#         try:
#             race_date = datetime.strptime(race.get('date', "1970-01-01"), "%Y-%m-%d")  # Parse date with format
#         except ValueError:
#             race_date = datetime(1970, 1, 1)  # Default date in case of error
#         # Create a new race entry
     
#         race_entry = Racecard(
#             race_id=race.get('race_id', "Unknown"),
#             date=race_date,
#             course=race.get('course', "Unknown"),
#             course_id=course_id,
#             off_time=race.get('off_time', "Unknown"),
#             off_dt=race.get('off_dt', "Unknown"),
#             race_name=race.get('race_name', "Unknown"),
#             distance_round=race.get('distance_round', "Unknown"),
#             distance=race.get('distance', "Unknown"),
#             distance_f=race.get('distance_f', "Unknown"),
#             region=race.get('region', "Unknown"),
#             pattern=race.get('pattern', "Unknown"),
#             race_class=race.get('race_class', "Unknown"),
#             race_type=race.get('type', "Unknown"),
#             age_band=race.get('age_band', "Unknown"),
#             rating_band=race.get('rating_band', "Unknown"),
#             prize=race.get('prize', "Unknown"),
#             field_size=race.get('field_size', "Unknown"),
#             going_detailed=race.get('going_detailed', "Unknown"),
#             rail_movements=race.get('rail_movements', "Unknown"),
#             stalls=race.get('stalls', "Unknown"),
#             weather=race.get('weather', "Unknown"),
#             going=race.get('going', "Unknown"),
#             surface=race.get('surface', "Unknown"),
#             jumps=race.get('jumps', "Unknown"),
#             big_race=race.get('big_race', False),
#             is_abandoned=race.get('is_abandoned', False)
#         )
#         session.add(race_entry)
#         for runner in race.get('runners', []):  # Using get() to handle missing 'runners'
#             runner_entry = Runners(
#                 horse_id=runner.get('horse_id', "Unknown"),
#                 horse=runner.get('horse', "Unknown"),
#                 dob=runner.get('dob', "Unknown"),
#                 age=runner.get('age', "Unknown"),
#                 sex=runner.get('sex', "Unknown"),
#                 sex_code=runner.get('sex_code', "Unknown"),
#                 colour=runner.get('colour', "Unknown"),
#                 region=runner.get('region', "Unknown"),
#                 breeder=runner.get('breeder', "Unknown"),
#                 dam=runner.get('dam', "Unknown"),
#                 dam_id=runner.get('dam_id', "Unknown"),
#                 dam_region=runner.get('dam_region', "Unknown"),
#                 sire=runner.get('sire', "Unknown"),
#                 sire_id=runner.get('sire_id', "Unknown"),
#                 sire_region=runner.get('sire_region', "Unknown"),
#                 damsire=runner.get('damsire', "Unknown"),
#                 damsire_id=runner.get('damsire_id', "Unknown"),
#                 damsire_region=runner.get('damsire_region', "Unknown"),
#                 trainer=runner.get('trainer', "Unknown"),
#                 trainer_id=runner.get('trainer_id', "Unknown"),
#                 trainer_location=runner.get('trainer_location', "Unknown"),
#                 trainer_14_days=json.dumps(runner.get('trainer_14_days', "Unknown")),  # Use JSON to store complex structures
#                 owner=runner.get('owner', "Unknown"),
#                 owner_id=runner.get('owner_id', "Unknown"),
#                 comment=runner.get('comment', "Unknown"),
#                 spotlight=runner.get('spotlight', "Unknown"),
#                 number=runner.get('number', "Unknown"),
#                 draw=runner.get('draw', "Unknown"),
#                 headgear=runner.get('headgear', "Unknown"),
#                 headgear_run=runner.get('headgear_run', "Unknown"),
#                 wind_surgery=runner.get('wind_surgery', "Unknown"),
#                 wind_surgery_run=runner.get('wind_surgery_run', "Unknown"),
#                 lbs=runner.get('lbs', "Unknown"),
#                 ofr=runner.get('ofr', "Unknown"),
#                 rpr=runner.get('rpr', "Unknown"),
#                 ts=runner.get('ts', "Unknown"),
#                 jockey=runner.get('jockey', "Unknown"),
#                 jockey_id=runner.get('jockey_id', "Unknown"),
#                 silk_url=runner.get('silk_url', "Unknown"),
#                 last_run=runner.get('last_run', "Unknown"),
#                 form=runner.get('form', "Unknown"),
#                 trainer_rtf=runner.get('trainer_rtf', "Unknown"),
#                 race_id=race_entry.race_id

#             )
#             session.add(runner_entry)  # Add each runner entry
#     session.commit()  # Commit the session after adding all entries

# def fetch_data_racecards(limit, skip, date=None, region_codes=None, course_ids=None):
#     url = "https://api.theracingapi.com/v1/racecards/pro"
#     params = {
#         "limit": limit,
#         "skip": skip,
#     }
    
#     # If a date is provided, add it to the parameters
#     if date:
#         params["date"] = date  # API expects a single date in YYYY-MM-DD format
    
#     # If region_codes are provided, add them to the params
#     if region_codes:
#         params["region_codes"] = region_codes  # Should be a list like ['gb']
    
#     # If course_ids are provided, add them to the params
#     if course_ids:
#         params["course_ids"] = course_ids  # Should be a list like ['1', '2']
    
#     # Add a small delay to prevent hitting API rate limits
#     time.sleep(0.5)
    
#     # Send the GET request with the parameters
#     try:
#         resp = requests.get(url, params=params, auth=HTTPBasicAuth(user, password))
#         resp.raise_for_status()  # Raise an exception for HTTP errors
#         return resp.json()
#     except requests.exceptions.HTTPError as err:
#         print(f"HTTPError: {err}")
#         print("Response content:", resp.content)  # Print the error response from the API
#         raise

# def process_uk_racecards(limit=50, start_date=None, end_date=None, region_codes=None, course_ids=None):
#     skip = 0
#     total_records = None

#     # If a start_date and end_date are provided, generate the date range
#     if start_date and end_date:
#         # Assuming get_dates is defined elsewhere to generate the date range
#         date_range = get_dates(start_date, end_date)  # Generate a list of dates in YYYY-MM-DD format
#     elif start_date:
#         date_range = [start_date]  # Use a single date if only start_date is provided
#     else:
#         date_range = []  # If no date range is provided, use an empty list (or handle as needed)

#     # Loop through each date in the range
#     for date in date_range:
#         while total_records is None or skip < total_records:
#             data = fetch_data_racecards(limit, skip, date=date, region_codes=region_codes, course_ids=course_ids)
#             total_records = data.get('total')
#             if total_records is None:
#                 print("No 'total' key in response. Exiting.")
#                 break
#             if total_records == 0:
#                 print("No records found for the given date range, exiting.")
#                 break
#             save_to_database_racecards(data, session3)
#             skip += limit  # Increment skip to fetch the next set of data

#     session3.close()  # Close the session after processing





################## FUNCTIONS FOR RACECARD API ############################
from sqlalchemy.exc import IntegrityError
from datetime import datetime



def write_racecards_to_db(data):
    try:
        # Check if race_id already exists
        existing_race = session3.query(Racecard).filter_by(race_id=data.get('race_id')).first()

        if existing_race:
            # Update the existing record
            existing_race.date = datetime.strptime(data.get('date'), '%Y-%m-%d').date()  # Use .date() if using Date column
            existing_race.course = data.get('course')
            existing_race.course_id = data.get('course_id')
            existing_race.off_time = data.get('off_time')
            existing_race.off_dt = data.get('off_dt')
            existing_race.race_name = data.get('race_name')
            existing_race.distance_round = data.get('distance_round')
            existing_race.distance = data.get('distance')
            existing_race.distance_f = data.get('distance_f')
            existing_race.region = data.get('region')
            existing_race.pattern = data.get('pattern')
            existing_race.race_class = data.get('race_class')
            existing_race.type = data.get('type')
            existing_race.age_band = data.get('age_band')
            existing_race.rating_band = data.get('rating_band')
            existing_race.prize = data.get('prize')
            existing_race.field_size = data.get('field_size')
            existing_race.going_detailed = data.get('going_detailed')
            existing_race.rail_movements = data.get('rail_movements')
            existing_race.stalls = data.get('stalls')
            existing_race.weather = data.get('weather')
            existing_race.going = data.get('going')
            existing_race.surface = data.get('surface')
            existing_race.jumps = data.get('jumps')
        else:
            # Create a new record
            race = Racecard(
                race_id=data.get('race_id'),
                date=datetime.strptime(data.get('date'), '%Y-%m-%d').date(),  # Use .date() if using Date column
                course=data.get('course'),
                course_id=data.get('course_id'),
                off_time=data.get('off_time'),
                off_dt=data.get('off_dt'),
                race_name=data.get('race_name'),
                distance_round=data.get('distance_round'),
                distance=data.get('distance'),
                distance_f=data.get('distance_f'),
                region=data.get('region'),
                pattern=data.get('pattern'),
                race_class=data.get('race_class'),
                type=data.get('type'),
                age_band=data.get('age_band'),
                rating_band=data.get('rating_band'),
                prize=data.get('prize'),
                field_size=data.get('field_size'),
                going_detailed=data.get('going_detailed'),
                rail_movements=data.get('rail_movements'),
                stalls=data.get('stalls'),
                weather=data.get('weather'),
                going=data.get('going'),
                surface=data.get('surface'),
                jumps=data.get('jumps')
            )
            session3.add(race)
            session3.commit()  # Commit after adding the race
            
            # Retrieve the newly created race's ID
            existing_race = race
        # Process runners
        for runner in data.get('runners', []):
            runner_entry = Runners(
                horse_id=runner.get('horse_id'),
                horse=runner.get('horse'),
                dob=runner.get('dob'),
                age=runner.get('age'),
                sex=runner.get('sex'),
                sex_code=runner.get('sex_code'),
                colour=runner.get('colour'),
                region=runner.get('region'),
                breeder=runner.get('breeder'),
                dam=runner.get('dam'),
                dam_id=runner.get('dam_id'),
                dam_region=runner.get('dam_region'),
                sire=runner.get('sire'),
                sire_id=runner.get('sire_id'),
                sire_region=runner.get('sire_region'),
                damsire=runner.get('damsire'),
                damsire_id=runner.get('damsire_id'),
                damsire_region=runner.get('damsire_region'),
                trainer=runner.get('trainer'),
                trainer_id=runner.get('trainer_id'),
                trainer_location=runner.get('trainer_location'),
                trainer_14_days=json.dumps(runner.get('trainer_14_days')),  # Use JSON to store complex structures
                owner=runner.get('owner'),
                owner_id=runner.get('owner_id'),
                comment=runner.get('comment'),
                spotlight=runner.get('spotlight'),
                number=runner.get('number'),
                draw=runner.get('draw'),
                headgear=runner.get('headgear'),
                headgear_run=runner.get('headgear_run'),
                wind_surgery=runner.get('wind_surgery'),
                wind_surgery_run=runner.get('wind_surgery_run'),
                lbs=runner.get('lbs'),
                ofr=runner.get('ofr'),
                rpr=runner.get('rpr'),
                ts=runner.get('ts'),
                jockey=runner.get('jockey'),
                jockey_id=runner.get('jockey_id'),
                silk_url=runner.get('silk_url'),
                last_run=runner.get('last_run'),
                form=runner.get('form'),
                trainer_rtf=runner.get('trainer_rtf'),
                race_id=existing_race.race_id
            )
            session3.add(runner_entry)
        
        session3.commit()  # Commit after adding/updating runners

    except IntegrityError as e:
        session3.rollback()  # Rollback the session on error
        print(f"IntegrityError: {e}")
    except Exception as e:
        session3.rollback()
        print(f"An error occurred: {e}")

import requests
from requests.auth import HTTPBasicAuth
import time

def fetch_and_write_racecards_db(dates, region='GB'):
    for date in dates:
        params = {"date": date}
        time.sleep(2)  # Adjust sleep time if necessary
        
        try:
            resp = requests.get("https://api.theracingapi.com/v1/racecards/pro", 
                                auth=HTTPBasicAuth(user, password), params=params)
            resp.raise_for_status()  # Raise an exception for HTTP errors
            racecards_data = resp.json()
            # print("racecards data is:", racecards_data)

            # Check if the key 'racecards' exists and is a list
            if 'racecards' not in racecards_data or not isinstance(racecards_data['racecards'], list):
                print("Unexpected data format:", racecards_data)
                continue

            for race in racecards_data.get("racecards", []):
                race_region = race.get("region")
                print("Processing race with region:", race_region)
                
                if region and race_region != region:
                    print("Skipping race due to region mismatch")
                    continue
                
                time.sleep(1)  # Throttle requests
                # print("Writing race to DB:", race)
                write_racecards_to_db(race)
                
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except Exception as err:
            print(f"Other error occurred: {err}")



from datetime import datetime

def write_racecards_to_csv(race_data, runners_data, races_writer, runners_writer):
    try:
        # Write race data
        races_writer.writerow([
            race_data.get('race_id'),
            datetime.strptime(race_data.get('date'), '%Y-%m-%d').strftime('%Y-%m-%d'),
            race_data.get('region'),
            race_data.get('course'),
            race_data.get('course_id'),
            race_data.get('off_time'),
            race_data.get('off_dt'),
            race_data.get('race_name'),
            race_data.get('distance'),
            race_data.get('distance_f'),
            race_data.get('pattern'),
            race_data.get('race_class'),
            race_data.get('type'),
            race_data.get('age_band'),
            race_data.get('rating_band'),
            race_data.get('prize'),
            race_data.get('field_size'),
            race_data.get('going_detailed'),
            race_data.get('rail_movements'),
            race_data.get('stalls'),
            race_data.get('weather'),
            race_data.get('going'),
            race_data.get('jumps')
        ])

        # Write runner data
        for runner in runners_data:
            runners_writer.writerow([
                runner.get('horse_id'),
                runner.get('horse'),
                runner.get('dob'),
                runner.get('age'),
                runner.get('sex'),
                runner.get('sex_code'),
                runner.get('colour'),
                runner.get('region'),
                runner.get('breeder'),
                runner.get('dam'),
                runner.get('dam_id'),
                runner.get('dam_region'),
                runner.get('sire'),
                runner.get('sire_id'),
                runner.get('sire_region'),
                runner.get('damsire'),
                runner.get('damsire_id'),
                runner.get('damsire_region'),
                runner.get('trainer'),
                runner.get('trainer_id'),
                runner.get('trainer_location'),
                runner.get('trainer_14_days'),
                runner.get('owner'),
                runner.get('owner_id'),
                runner.get('prev_trainers'),
                runner.get('prev_owners'),
                runner.get('comment'),
                runner.get('spotlight'),
                runner.get('quotes'),
                runner.get('stable_tour'),
                runner.get('medical'),
                runner.get('number'),
                runner.get('draw'),
                runner.get('headgear'),
                runner.get('headgear_run'),
                runner.get('wind_surgery'),
                runner.get('wind_surgery_run'),
                runner.get('past_results_flags'),
                runner.get('lbs'),
                runner.get('ofr'),
                runner.get('rpr'),
                runner.get('ts'),
                runner.get('jockey'),
                runner.get('jockey_id'),
                runner.get('silk_url'),
                runner.get('last_run'),
                runner.get('form'),
                runner.get('trainer_rtf'),
                runner.get('odds'),
                race_data.get('race_id')
            ])

    except Exception as e:
        print(f"An error occurred: {e}")


def fetch_and_write_racecards_csv(dates, region='GB'):
    # Open CSV files for writing
    with open('./data/racecards_pred.csv', mode='w', newline='') as file1, \
         open('./data/racecards_runners_pred.csv', mode='w', newline='') as file2:
        races_writer = csv.writer(file1)
        runners_writer = csv.writer(file2)
        
        # Write headers (only once)
        races_writer.writerow([
            'race_id', 'date', 'region', 'course', 'course_id', 'off_time',
            'off_dt', 'race_name', 'distance', 'distance_f', 'pattern',
            'race_class', 'type', 'age_band', 'rating_band', 'prize',
            'field_size', 'going_detailed', 'rail_movements', 'stalls',
            'weather', 'going', 'jumps'
        ])
        runners_writer.writerow([
            'horse_id', 'horse_name', 'dob', 'age', 'sex', 'sex_code',
            'colour', 'region', 'breeder', 'dam', 'dam_id', 'dam_region',
            'sire', 'sire_id', 'sire_region', 'damsire', 'damsire_id',
            'damsire_region', 'trainer', 'trainer_id', 'trainer_location',
            'trainer_14_days', 'owner', 'owner_id', 'prev_trainers',
            'prev_owners', 'comment', 'spotlight', 'quotes', 'stable_tour',
            'medical', 'number', 'draw', 'headgear', 'headgear_run',
            'wind_surgery', 'wind_surgery_run', 'past_results_flags', 'lbs',
            'ofr', 'rpr', 'ts', 'jockey', 'jockey_id', 'silk_url',
            'last_run', 'form', 'trainer_rtf', 'odds', 'race_id'
        ])
        
        # Fetch and write data
        for date in dates:
            params = {"date": date}
            time.sleep(0.5) 
            resp = requests.get("https://api.theracingapi.com/v1/racecards/pro", auth=HTTPBasicAuth(user, password), params=params)
            
            if resp.status_code == 200:
                racecards_data = resp.json()
                for racecard in racecards_data.get("racecards", []):
                    race_region = racecard.get("region")
                    if region and race_region != region:
                        continue  # Skip if the race's region does not match the specified region
    
                    runners_data = racecard.get("runners", [])
                    
                    write_racecards_to_csv(racecard, runners_data, races_writer, runners_writer)
            else:
                print("Error fetching racecards:", resp.status_code)


def convert_column_to_string(df, column_name):
    if column_name in df.columns:
        try:
            # Convert column to string
            df[column_name] = df[column_name].astype(str)
        except Exception as e:
            print(f"Error converting column {column_name}: {e}")


def export_tables_to_csv():
    try:
        # Load the 'races' table into a DataFrame
        df_races = pd.read_sql_table('races', con=engine2)
        print("Races table loaded.")
        
        # Inspect data (optional)
        print("Inspecting races table data:")
        print(df_races.head())  # Print first few rows for inspection
        
        # Convert necessary columns to strings
        convert_column_to_string(df_races, 'race_id')
        
        # Export the 'races' DataFrame to CSV
        df_races.to_csv('races.csv', index=False)
        print("The 'races' table has been exported to 'races.csv'.")

        try:
            # Define the SQL query to read the 'runners' table
            query = "SELECT * FROM runners;"
            
            # Load the 'runners' table into a DataFrame
            df_runners = pd.read_sql_query(query, con=engine2)
            
            # Print the first few rows to inspect
            print("Inspecting 'runners' table data:")
            print(df_runners.head())
            
            # Convert 'race_id' to string
            df_runners['race_id'] = df_runners['race_id'].astype(str)
            
            # Export to CSV
            df_runners.to_csv('runners.csv', index=False)
            print("The 'runners' table has been exported to 'runners.csv'.")
    
        except Exception as e:
            print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def export_racecards_tables_to_csv():
    try:
        # Load the 'races' table into a DataFrame
        df_races = pd.read_sql_table('racecards', con=engine3)
        print("Races table loaded.")
        
        # Inspect data (optional)
        print("Inspecting races table data:")
        print(df_races.head())  # Print first few rows for inspection
        
        # Convert necessary columns to strings
        convert_column_to_string(df_races, 'race_id')
        
        # Export the 'races' DataFrame to CSV
        df_races.to_csv('racecards.csv', index=False)
        print("The 'races' table has been exported to 'racecards.csv'.")

        try:
            # Define the SQL query to read the 'runners' table
            query = "SELECT * FROM runners2;"
            
            # Load the 'runners' table into a DataFrame
            df_runners = pd.read_sql_query(query, con=engine3)
            
            # Print the first few rows to inspect
            print("Inspecting 'runners2' table data:")
            print(df_runners.head())
            
            # Convert 'race_id' to string
            df_runners['race_id'] = df_runners['race_id'].astype(str)
            
            # Export to CSV
            df_runners.to_csv('runners_rc.csv', index=False)
            print("The 'runners' table has been exported to 'runners_rc.csv'.")
    
        except Exception as e:
            print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")





#################### FUNCTIONS FOR NORTH AMERICAN API #####################

def query_and_merge_data():
    # Fetch entries and results into pandas DataFrames
    entries_df = pd.read_sql_table('entries', con=engine)
    results_df = pd.read_sql_table('results', con=engine)

    # Merge entries and results DataFrames using pd.merge()
    merged_df = pd.merge(entries_df, results_df, on=['meet_id', 'track_name', 'race_key_race_number', 'date', 'runners_horse_name'], how='outer')
    merged_df['race_id'] = merged_df['meet_id'] + '_' + merged_df['race_key_race_number'].astype(str)
    return merged_df

# Function to convert fractional odds to decimal odds
def convert_fractional_to_decimal(fractional_odds):
    if fractional_odds.lower() == "nan":
        return None
    try:
        numerator, denominator = map(int, fractional_odds.split("-"))
        return round(float(numerator) / denominator + 1, 1)  # Round to the nearest 10th
    except ValueError:
        return None

# Function to combine first name and last name
def combine_names(first_name, last_name):
    if first_name:  # Check if first_name is not empty
        # Extracting the first initial of the first name
        first_initial = first_name[0].upper()
    else:
        first_initial = ""  # If first_name is empty, set first_initial to an empty string
    # Combining the first initial with the last name with a space in between
    full_name = f"{first_initial} {last_name}"
    return full_name
from sqlalchemy.exc import IntegrityError
# Function to write entries to database
# Assuming `session` is already created and `Entry` class is defined

def write_entries_to_csv(meet, writer):
    meet_id = meet.get("meet_id")
    races = meet.get("races", [])
    date=meet.get("date")
    track_name=meet.get("track_name")
    weather=meet.get("weather",[])
    forecast_weather = weather.get("forecast_weather_description", "nan")
    for race in races:
        race_key_race_number = race.get("race_key", {}).get("race_number")
        distance_value_str = race.get("distance_value")
        course_type_str=race.get("course_type")
        course_type_class_str=race.get("course_type_class")
        surface_description_str=race.get("surface_description")
        track_condition_str=race.get("track_condition")
        purse_str=race.get("purse")
        if distance_value_str and distance_value_str != "NaN":  # Check if distance_value is not empty and not "NaN"
            distance_value = float(distance_value_str)
        else:
            distance_value = None  # Or any other appropriate value or handling
        # Process the other race details (course type, surface, track condition)
        course_type = course_type_str if course_type_str else "nan"
        course_type_class = course_type_class_str if course_type_class_str else "nan"
        surface_description = surface_description_str if surface_description_str else "nan"
        track_condition = track_condition_str if track_condition_str else "nan"
        purse = purse_str if purse_str else "nan"
        for runner in race.get("runners", []):
            horse_data_pools = runner.get("horse_data_pools","nan")
            if horse_data_pools:
                pool_type=horse_data_pools[0].get("pool_type_name", "nan")
                amount=horse_data_pools[0].get("amount", "nan")
                fractional_odds = horse_data_pools[0].get("fractional_odds", "nan")
                decimal_odds = convert_fractional_to_decimal(fractional_odds)
            else:
                fractional_odds = "nan"
                decimal_odds='nan'
                pool_type='nan'
                amount='nan'
            runners_horse_name = runner.get("horse_name")
            jockeys = runner.get("jockey")
            if jockeys:
              jockey_f_name=jockeys.get("first_name")
              jockey_l_name=jockeys.get("last_name")
            else:
              jockey_f_name=None
              jockey_l_name=None
            jockey_full=combine_names(jockey_f_name,jockey_l_name)
            live_odds=runner.get("live_odds")
            morning_line_odds=runner.get("morning_line_odds")
            if live_odds:
              live_odds=runner.get("live_odds")
              live_odds1=convert_fractional_to_decimal(live_odds)
            else:
              live_odds=None
              live_odds1=None
            if morning_line_odds:
              morning_line_odds=runner.get("morning_line_odds")
              morning_line_odds1=convert_fractional_to_decimal(morning_line_odds)
            else:
              morning_line_odds=None
              morning_line_odds1=None
            trainers=runner.get("trainer")
            if trainers:
              trainer_f_name=trainers.get("first_name")
              trainer_l_name=trainers.get("last_name")
            else:
              trainer_f_name=None
              trainer_l_name=None
            trainer_full=combine_names(trainer_f_name,trainer_l_name)
            runners_post_pos = runner.get("post_pos")
            runners_program_numb=runner.get("program_number")
            runners_weight = runner.get("weight")
        
            writer.writerow([meet_id,track_name,race_key_race_number, date,  distance_value, decimal_odds,morning_line_odds1,live_odds1, runners_horse_name, jockey_f_name, jockey_l_name, jockey_full, trainer_f_name, trainer_l_name,trainer_full, runners_post_pos,runners_program_numb, runners_weight,amount,pool_type,course_type,course_type_class,surface_description,track_condition,purse,forecast_weather])

def write_results_to_csv(meet, writer):
    meet_id = meet.get("meet_id")
    date=meet.get("date")
    track_id=meet.get("track_id")
    track_name=meet.get("track_name")
    races = meet.get("races", [])
    for race in races:
        race_key_race_number = race.get("race_key", {}).get("race_number")
        runners = race.get("runners", [])
        print("Number of runners for race {}: {}".format(race_key_race_number, len(runners)))
        # Handle placed runners
        placed_runners_count = len(runners)
        for count, runner in enumerate(runners):
            runners_horse_name = runner.get("horse_name")
            finish_position = count + 1
            place_payoff = runner.get("place_payoff")
            show_payoff = runner.get("show_payoff")
            win_payoff = runner.get("win_payoff")
            owner_f_name = runner.get("owner_first_name")
            owner_l_name = runner.get("owner_last_name")
            owner_full=combine_names(owner_f_name,owner_l_name)
            writer.writerow([meet_id, owner_f_name,owner_l_name,owner_full,track_name,race_key_race_number,date, runners_horse_name, finish_position, place_payoff, show_payoff, win_payoff])
        # Handle additional runners
        also_ran_value = race.get("also_ran", None)
        also_ran = []
        if also_ran_value:
            if isinstance(also_ran_value, str):
                also_ran_list = also_ran_value.replace('  and  ', ',').split(',')
                for runner in also_ran_list:
                    also_ran.append(runner.strip())
            if isinstance(also_ran_value, list):
                also_ran = also_ran_value
        for count, runner_name in enumerate(also_ran):
            runners_horse_name = runner_name
            finish_position = placed_runners_count + count + 1
            place_payoff = "0"
            show_payoff = "0"
            win_payoff = "0"
            owner_f_name = ""
            owner_l_name = ""
            owner_full=""
            writer.writerow([meet_id, owner_f_name,owner_l_name,owner_full,track_name,race_key_race_number,date, runners_horse_name, finish_position, place_payoff, show_payoff, win_payoff])

def write_entries_to_db(meet):
    meet_id = meet.get("meet_id")
    races = meet.get("races", [])
    date = datetime.strptime(meet.get("date"), "%Y-%m-%d")
    track_name = meet.get("track_name")
    for race in races:
        race_key_race_number = race.get("race_key", {}).get("race_number")
        distance_value_str = race.get("distance_value")
        distance_value = float(distance_value_str) if distance_value_str and distance_value_str != "NaN" else None
        for runner in race.get("runners", []):
            horse_data_pools = runner.get("horse_data_pools", "nan")
            fractional_odds = horse_data_pools[0].get("fractional_odds", "nan") if horse_data_pools else "nan"
            decimal_odds = convert_fractional_to_decimal(fractional_odds)
            runners_horse_name = runner.get("horse_name")
            jockeys = runner.get("jockey")
            jockey_f_name = jockeys.get("first_name") if jockeys else None
            jockey_l_name = jockeys.get("last_name") if jockeys else None
            jockey_full = combine_names(jockey_f_name, jockey_l_name)
            live_odds = convert_fractional_to_decimal(runner.get("live_odds")) if runner.get("live_odds") else None
            morning_line_odds = convert_fractional_to_decimal(runner.get("morning_line_odds")) if runner.get("morning_line_odds") else None
            trainers = runner.get("trainer")
            trainer_f_name = trainers.get("first_name") if trainers else None
            trainer_l_name = trainers.get("last_name") if trainers else None
            trainer_full = combine_names(trainer_f_name, trainer_l_name)
            runners_post_pos = runner.get("post_pos")
            runners_program_numb = runner.get("program_number")
            runners_weight = runner.get("weight")
            entry = Entry(
                meet_id=meet_id,
                track_name=track_name,
                race_key_race_number=race_key_race_number,
                date=date,
                distance_value=distance_value,
                decimal_odds=decimal_odds,
                morning_line_odds=morning_line_odds,
                live_odds=live_odds,
                runners_horse_name=runners_horse_name,
                jockey_f_name=jockey_f_name,
                jockey_l_name=jockey_l_name,
                jockey_full=jockey_full,
                trainer_f_name=trainer_f_name,
                trainer_l_name=trainer_l_name,
                trainer_full=trainer_full,
                runners_post_pos=runners_post_pos,
                runners_program_numb=runners_program_numb,
                runners_weight=runners_weight
            )
            try:
                session.add(entry)
                session.commit()  # Try to commit the transaction
            except IntegrityError:
                session.rollback()  # Rollback the transaction if a duplicate is detected
                print(f"Duplicate entry detected for horse: {runners_horse_name}. Skipping to next entry.")
                
# Function to write results to database
def write_results_to_db(meet):
    meet_id = meet.get("meet_id")
    date = datetime.strptime(meet.get("date"), "%Y-%m-%d")
    track_name = meet.get("track_name")
    races = meet.get("races", [])

    if not races:
        print(f"'races' key not found or None for meet: {meet_id}")
        return

    for race in races:
        if race is None:
            print("Encountered None race in meet:", meet_id)
            continue

        race_key_race_number = race.get("race_key", {}).get("race_number")
        runners = race.get("runners", [])
        
        for count, runner in enumerate(runners):
            runners_horse_name = runner.get("horse_name")
            finish_position = count + 1
            place_payoff = runner.get("place_payoff")
            show_payoff = runner.get("show_payoff")
            win_payoff = runner.get("win_payoff")
            owner_first_name = runner.get("owner_first_name")
            owner_last_name = runner.get("owner_last_name")
            owner_full=combine_names(owner_first_name,owner_last_name)
            result = Result(
                meet_id=meet_id,
                track_name=track_name,
                race_key_race_number=race_key_race_number,
                date=date,
                owner_f_name=owner_first_name,
                owner_l_name=owner_last_name,
                owner_full=owner_full,
                runners_horse_name=runners_horse_name,
                finish_position=finish_position,
                place_payoff=place_payoff,
                show_payoff=show_payoff,
                win_payoff=win_payoff
            )
            try:
                session.add(result)
                session.commit()  # Try to commit the transaction
            except IntegrityError:
                session.rollback()  # Rollback the transaction if a duplicate is detected
                print(f"Duplicate result detected for horse: {runners_horse_name}. Skipping to next result.")
                continue  # Move to the next result
        also_ran_value = race.get("also_ran", None)
        also_ran = []

        if also_ran_value:
            if isinstance(also_ran_value, str):
                also_ran = [runner.strip() for runner in also_ran_value.replace('  and  ', ',').split(',')]
            elif isinstance(also_ran_value, list):
                also_ran = also_ran_value

        for count, runner_name in enumerate(also_ran):
            finish_position = len(runners) + count + 1
            place_payoff = 0.0
            show_payoff = 0.0
            win_payoff = 0.0
            owner_first_name=""
            owner_last_name=""
            owner_full=""
          
            
            result = Result(
                meet_id=meet_id,
                track_name=track_name,
                race_key_race_number=race_key_race_number,
                date=date,
                owner_f_name=owner_first_name,
                owner_l_name=owner_last_name,
                owner_full=owner_full,
                runners_horse_name=runner_name,
                finish_position=finish_position,
                place_payoff=place_payoff,
                show_payoff=show_payoff,
                win_payoff=win_payoff
            )

            try:
                session.add(result)
                session.commit()  # Try to commit the transaction
            except IntegrityError:
                session.rollback()  # Rollback the transaction if a duplicate is detected
                print(f"Duplicate result detected for also-ran horse: {runner_name}. Skipping to next result.")
                continue  # Move to the next result

def fetch_and_write_db(dates):
    for date in dates:
        params = {"start_date": date, "end_date": date}
        time.sleep(1)
        resp = requests.get("https://api.theracingapi.com/v1/north-america/meets", auth=HTTPBasicAuth(user, password), params=params)

        if resp.status_code == 200:
            meets_data = resp.json()
            for meet in meets_data.get("meets", []):
                time.sleep(1)
                meet_id = meet.get("meet_id")

                # Fetch entries
                resp_entry = requests.get(f"https://api.theracingapi.com/v1/north-america/meets/{meet_id}/entries", auth=HTTPBasicAuth(user, password))
                if resp_entry.status_code == 200:
                    meet_entries = resp_entry.json()
                    if meet_entries:
                        write_entries_to_db(meet_entries)
                    else:
                        st.write("No entries found for meet:", meet_id)
                else:
                    st.write("Error fetching entries for meet:", meet_id)

                # Fetch results
                resp_results = requests.get(f"https://api.theracingapi.com/v1/north-america/meets/{meet_id}/results", auth=HTTPBasicAuth(user, password))
                if resp_results.status_code == 200:
                    meet_results = resp_results.json()
                    if meet_results:
                        write_results_to_db(meet_results)
                    else:
                        st.write("No results found for meet:", meet_id)
                else:
                    st.write("Error fetching results for meet:", meet_id)
        else:
            st.write("Error fetching meets for date:", date)
def fetch_and_write_csv(dates):
    with open('./data/entries_pred.csv', mode='w', newline='') as file1, \
         open('./data/results_pred.csv', mode='w', newline='') as file2:
        entries_writer = csv.writer(file1)
        results_writer = csv.writer(file2)
        
        # Write headers for entries.csv
        entries_writer.writerow(["meet_id", "track_name", "race_key_race_number", "date", "distance_value", "decimal_odds", "morning_line_odds", "live_odds", "runners_horse_name", "jockey_f_name", "jockey_l_name", "jockey_full", "trainer_f_name", "trainer_l_name", "trainer_full", "runners_post", "runners_program_number", "runners_weight","amount","pool_type","course_type","course_type_class","surface_description","track_condition","purse","forecast"])
        
        # Write headers for results.csv
        results_writer.writerow(["meet_id", "owner_f_name","owner_l_name","owner_full","track_name", "race_key_race_number", "date", "runners_horse_name", "finish_position", "place_payoff", "show_payoff", "win_payoff"])
        
        for date in dates:
            params = {"start_date": date, "end_date": date}
            time.sleep(1)
            resp = requests.get("https://api.theracingapi.com/v1/north-america/meets", auth=HTTPBasicAuth(user, password), params=params)
            if resp.status_code == 200:
                meets_data = resp.json()
                for meet in meets_data.get("meets", []):
                    time.sleep(1)
                    meet_id = meet.get("meet_id")
                    # Fetch entries
                    resp_entry = requests.get(f"https://api.theracingapi.com/v1/north-america/meets/{meet_id}/entries", auth=HTTPBasicAuth(user, password))
                    if resp_entry.status_code == 200:
                        meet_entries = resp_entry.json()
                        if meet_entries:
                            write_entries_to_csv(meet_entries, entries_writer)
                        else:
                            print("No entries found for meet:", meet_id)
                    else:
                        print("Error fetching entries for meet:", meet_id)
                    # Fetch results
                    resp_results = requests.get(f"https://api.theracingapi.com/v1/north-america/meets/{meet_id}/results", auth=HTTPBasicAuth(user, password))
                    if resp_results.status_code == 200:
                        meet_results = resp_results.json()
                        if meet_results.get("races"):
                            write_results_to_csv(meet_results, results_writer)
                        else:
                            print("No races found for meet:", meet_id)
                    else:
                        print("Error fetching results for meet:", meet_id)
            else:
                print("Error fetching meets for date:", date)

############################### METHODS FOR APP ############################
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func, and_
from models import Entry, Result

# Define a function to backtest the betting strategy
# Function to capture matplotlib plots as images
def calculate_win_percentage(recent_runs):
    # Split the string into a list of positions
    positions = recent_runs.split('/')
    # Count the number of times the horse finished first
    wins = positions.count('1.0')
    # Calculate the win percentage
    win_percentage = (wins / len(positions)) * 100
    return win_percentage

def get_matplotlib_figure():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer

# Function to display matplotlib plots in Streamlit
def display_plots():
    # Capture the figure
    buffer = get_matplotlib_figure()
    st.image(buffer, use_column_width=True)

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import math
def run_reg_model(model, X_train, y_train, X_test, y_test, X_unseen):
        results = pd.DataFrame(columns=['Model', 'RMSE_train', 'RMSE_test', 
                                'Generalization', 'Top1_Train_Accuracy', 'Top1_Test_Accuracy',
                                'Top3_Train_Accuracy', 'Top3_Test_Accuracy'])
        # Store model name
        model_name = model.__class__.__name__

        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_unseen = scaler.transform(X_unseen)

        # Fit the model         
        model.fit(X_train, y_train)
        
        # Predict on the training set
        y_train_pred = model.predict(X_train)
        y_train_pred = pd.DataFrame(y_train_pred)

        # Predict on the testing set
        y_test_pred = model.predict(X_test)
        y_test_pred = pd.DataFrame(y_test_pred)

        # Calculate the RMSE
        train_rmse = round(math.sqrt(mean_squared_error(y_train, y_train_pred)), 3)
        test_rmse = round(math.sqrt(mean_squared_error(y_test, y_test_pred)), 3)
        
        # Calculate the accuracy
        train_accuracy, train_accuracy_top3 = find_prob(y_train_pred)
        test_accuracy, test_accuracy_top3 = find_prob(y_test_pred)

        # Calculate generalization error percentage
        generalization_error = round((test_rmse - train_rmse)/train_rmse*100, 3)

        # Print the results
        print('Model results for', model_name, ':')
        print('Train RMSE: ', train_rmse)
        print('Test RMSE: ', test_rmse)
        print('Generalization Error: ', generalization_error, '%', '\n')

        print('Train Accuracy for finding Top position: ', train_accuracy)
        print('Test Accuracy for finding Top position: ', test_accuracy, '\n')

        print('Train Accuracy for finding Top 3 positions: ', train_accuracy_top3)
        print('Test Accuracy for finding Top 3 positions: ', test_accuracy_top3)

        # Append the results to the dataframe
        results.loc[len(results)] = [model_name, train_rmse, test_rmse, generalization_error,
                                  train_accuracy, test_accuracy, train_accuracy_top3, test_accuracy_top3]
        
        # predict on unseen data
        y_unseen_pred = model.predict(X_unseen)
        y_unseen_pred = pd.DataFrame(y_unseen_pred)

        return y_unseen_pred
def time_to_seconds(time_str):
    parts = time_str.split(':')
    return int(parts[0]) * 60 + float(parts[1])
def find_prob(y_pred,df_test):
    
    i=0
    count_top_winners = 0
    count_top_correct = 0

    count_top3_winners = 0
    count_top3_correct = 0

    for column in ['HorseWin', 'HorseRankTop3']:
            
        for race in df_test['race_id'].unique():
            
            # Create temp dataframe
            temp = df_test[df_test['race_id']==race]

            # Get the index of the temp dataframe
            temp_index = temp.index

            # Find the index of the winners from the temp dataframe
            if i == 0:
                winners_index = temp[temp['position']==1].index
            else:
                winners_index = temp[temp['position']<=3].index

            # Create a temp dataframe for the predicted probabilities
            temp_pred = y_pred.iloc[temp_index]

            # Sort the temp dataframe by the predicted timings
            temp_pred = temp_pred.sort_values(by=temp_pred.columns[0])

            # Get the index of the winners from the temp pred dataframe
            if i == 0:
                winners_pred_index = temp_pred[:1].index
            else:
                winners_pred_index = temp_pred[:3].index

            # Count the number of winners and correct predictions
            if i == 0:
                count_top_winners += len(winners_index)
                count_top_correct += len(set(winners_index).intersection(set(winners_pred_index)))
            else:
                count_top3_winners += len(winners_index)
                count_top3_correct += len(set(winners_index).intersection(set(winners_pred_index)))
        i+=1
    
    # Calculate the accuracy
    top_accuracy = round(count_top_correct/count_top_winners, 3)
    top3_accuracy = round(count_top3_correct/count_top3_winners, 3)

    return top_accuracy, top3_accuracy

# define regression backtest function
def simple_reg_strategy_UK(model_pred, df_unseen,graph=True):
        
        df_unseen_results = df_unseen[['position', 'race_id','tote_win', 'HorseWin', 'horse_name']]

        # rename columns
        df_unseen_results = df_unseen_results.rename(columns={'race_id': 'RaceID', 
                                                        'horse_name': 'HorseID', 
                                                        'HorseWin':'ActualWin'})
        
        # merge the prediction with the test data
        df_unseen_results['pred_time'] = model_pred[0]

        money = 0
        bets_made = []
        cumulative_money = [0]

        for race_id in df_unseen_results['RaceID'].unique():

                # make a temporary dataframe one for that particular race
                temp_df = df_unseen_results[df_unseen_results['RaceID']==race_id]

                # bet only on the horse with the fastest time
                # return dataframe where the time is the minimum
                bets = temp_df[temp_df['pred_time']==temp_df['pred_time'].min()]

                # deduct money for bets we made
                deduction  = -len(bets)
                bets.loc[:, 'tote_win'] = bets['tote_win'].replace('[£,]', '', regex=True).astype(float)

                # amount won from bets
                # sum of multiplying the odds with the prediction
                amount_won = sum(bets['tote_win']*bets['ActualWin'])
                
                # add the amount won to the money
                money += (amount_won + deduction)

                # append the money to the cumulative money list
                cumulative_money.append(money)

                # append the bets made to the bets made list
                bets_made.append(len(bets))
        
        if graph==True:
                # plot the cumulative money
                plt.figure(figsize=(10, 6))
                plt.plot(cumulative_money)
                plt.axhline(y=0, color='r', linestyle='-')
                plt.title('Cumulative Money for Every $2 Bet')
                plt.xlabel('Bets Made')
                plt.ylabel('Cumulative Money')
                display_plots()

                # plot the bets made
                plt.figure(figsize=(10, 6))
                plt.plot(bets_made)
                plt.title('Bets Made')
                display_plots()

        
        st.write(f'**Final Money:** ${round(money, 3)}')
        st.write(f'**Total Bets Made:** {round(sum(bets_made), 3)}')
        return money, bets_made
                
# Define a function to run the model
def run_model_lr(X_train, y_train, X_test, y_test,kfold):
    df_results = pd.DataFrame(columns=['Model', 'Target', 'CV F1-Score', 'F1 Score', 'PR AUC', 'Recall', 'Precision'])
    df_pred = pd.DataFrame()
    df_pred['RaceID'] = df_unseen['race_id']
    df_pred['HorseID'] = df_unseen['runners_horse_name']
    lr =  LogisticRegression(solver='lbfgs', penalty='l2', C=1, max_iter=500)
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                    'penalty': [ 'l2']}
    grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5)
    for column in ['HorseWin', 'HorseRankTop3']:
        grid.fit(X_train, y_train[column].to_numpy())
        print(grid.best_params_)
        print(grid.best_score_)
        lr = grid.best_estimator_
        print("Model coefficients:",lr.coef_)
      

        model_name = str(lr).split('(')[0]

        # Display results in Streamlit
        # st.write(f"## Results for model {model_name} and target variable {column}:")
        st.markdown(f"<small>Results for model {model_name} and target variable {column}:</small>", unsafe_allow_html=True)

        print("y train is",y_train)
        # Fit the model
        lr.fit(X_train, y_train[column].to_numpy())

        # Calculate the cross-validation score
        cv_score = cross_val_score(lr, X_train, y_train[column].to_numpy(), cv=kfold, scoring='f1_weighted').mean()
        cv_score = round(cv_score, 3)

        # Make predictions
        y_pred = lr.predict(X_test)
        y_unseen_pred = lr.predict(X_unseen)

        # Store the predictions in the dataframe
        df_pred[column] = y_unseen_pred

        # Calculate the f1 score
        f1 = f1_score(y_test[column].to_numpy(), y_pred, average='weighted')
        f1 = round(f1, 3)

        # Calculate PR AUC
        pr_auc = average_precision_score(y_test[column].to_numpy(), y_pred, average='weighted')
        pr_auc = round(pr_auc, 3)

        # Calculate Recall
        tn, fp, fn, tp = confusion_matrix(y_test[column].to_numpy(), y_pred).ravel()
        recall = tp / (tp + fn)
        recall = round(recall, 3)

        # Calculate Precision
        precision = tp / (tp + fp)
        precision = round(precision, 3)

        # Append the results to the dataframe
        df_results.loc[len(df_results)] = [model_name, column, cv_score, f1, pr_auc, recall, precision]

        # Display results in Streamlit
        st.write('**Cross Validation Score (F1-weighted):** ', cv_score)
        st.write('**F1 Score:** ', f1)
        st.write('**PR AUC (Avg Precision):** ', pr_auc)
        st.write('**Recall:** ', recall)
        st.write('**Precision:** ', precision)
        # Debugging output
        print("y_test type:", type(y_test))
        print("y_test columns:", y_test.columns if hasattr(y_test, 'columns') else 'Not a DataFrame')
        print("y_test head:\n", y_test.head())
        print("Current column:", column)

        print("y_test column values:\n", y_test[column].to_numpy())
        print("Shape of y_test[column]:", y_test[column].to_numpy().shape)
        print("Shape of y_pred:", y_pred.shape)
    
        # Display confusion matrix using Streamlit
        st.write(f"### Confusion Matrix for {column}")
        plot_confusion_matrix(y_test, y_pred, column)

        # # Plot precision recall curve
        

        # Capture and display the confusion matrix plot
        st.pyplot(plt.gcf())
        plt.clf()

        # Display precision-recall curve using Streamlit
        st.write(f"### Precision-Recall Curve for {column}")
        plot_pr_auc(X_test, y_test, lr, column)

        # Capture and display the precision-recall curve plot
        st.pyplot(plt.gcf())
        plt.clf()

    # Display the results dataframe in Streamlit
    st.write("### Model Results Summary")
    st.dataframe(df_results)
    return df_pred

def run_model_smote(X_train, y_train, X_test, y_test,kfold):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
    from imblearn.pipeline import Pipeline  # Use imblearn's Pipeline
    from imblearn.over_sampling import SMOTE
    import pandas as pd
     # Create a dataframe to store the predictions for the UNSEEN data
    df_pred = pd.DataFrame()
    df_pred['RaceID'] = df_unseen['race_id']
    df_pred['HorseID'] = df_unseen['runners_horse_name']

    # Smote the training data
    sm = SMOTE(random_state = 42)
    rfc = RandomForestClassifier(max_depth=10, random_state = 42)

    # Steps for the pipeline
    steps = [('smote', sm), ('rfc', rfc)]

    # Create the pipeline
    model = Pipeline(steps = steps)
  
    # Store model name
    model_name = str(model).split('(')[0]

    for column in ['HorseWin', 'HorseRankTop3']:

        # Print the column name
        print(f"Results for model {model_name} and target variable {column}:")
        
        # Fit the model
        model.fit(X_train, y_train[column].to_numpy())
        
        # Calculate the cross validation score
        cv_score = cross_val_score(model, X_train, y_train[column].to_numpy(), cv=kfold, scoring='f1_weighted').mean()
        cv_score = round(cv_score, 3)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_unseen_pred = model.predict(X_unseen)

        # Store the predictions in the dataframe
        df_pred[column] = y_unseen_pred
        
        # Calculate the f1 score
        f1 = f1_score(y_test[column].to_numpy(), y_pred, average='weighted')
        f1 = round(f1, 3)
        
        # Calculate PR AUC
        pr_auc = average_precision_score(y_test[column].to_numpy(), y_pred, average='weighted')
        pr_auc = round(pr_auc, 3)

        # Calculate Recall
        tn, fp, fn, tp = confusion_matrix(y_test[column].to_numpy(), y_pred).ravel()
        recall = tp / (tp + fn)
        recall = round(recall, 3)

        # Calculate Precision
        precision = tp / (tp + fp)
        precision = round(precision, 3)

        # Append the results to the dataframe
        df_results.loc[len(df_results)] = [model_name, column, cv_score, f1, pr_auc, recall, precision]

        # Display results in Streamlit
        st.write('**Cross Validation Score (F1-weighted):** ', cv_score)
        st.write('**F1 Score:** ', f1)
        st.write('**PR AUC (Avg Precision):** ', pr_auc)
        st.write('**Recall:** ', recall)
        st.write('**Precision:** ', precision)
        # Debugging output
        print("y_test type:", type(y_test))
        print("y_test columns:", y_test.columns if hasattr(y_test, 'columns') else 'Not a DataFrame')
        print("y_test head:\n", y_test.head())
        print("Current column:", column)

        print("y_test column values:\n", y_test[column].to_numpy())
        print("Shape of y_test[column]:", y_test[column].to_numpy().shape)
        print("Shape of y_pred:", y_pred.shape)
    
        # Display confusion matrix using Streamlit
        st.write(f"### Confusion Matrix for {column}")
        plot_confusion_matrix(y_test, y_pred, column)

        # # Plot precision recall curve
        

        # Capture and display the confusion matrix plot
        st.pyplot(plt.gcf())
        plt.clf()

        # Display precision-recall curve using Streamlit
        st.write(f"### Precision-Recall Curve for {column}")
        plot_pr_auc(X_test, y_test, model, column)

        # Capture and display the precision-recall curve plot
        st.pyplot(plt.gcf())
        plt.clf()

    # Display the results dataframe in Streamlit
    st.write("### Model Results Summary")
    st.dataframe(df_results)
    return df_pred

def run_model_rfc(X_train, y_train, X_test, y_test,kfold):
    df_results = pd.DataFrame(columns=['Model', 'Target', 'CV F1-Score', 'F1 Score', 'PR AUC', 'Recall', 'Precision'])
    df_pred = pd.DataFrame()
    df_pred['RaceID'] = df_unseen['race_id']
    df_pred['HorseID'] = df_unseen['runners_horse_name']
    rfc = RandomForestClassifier(n_estimators = 100, random_state = 42)
    param_grid = {'max_depth': [20, 40],
                'max_features': [5, 10],
                'min_samples_leaf': [1, 2],
                'min_samples_split': [1, 2]}
    grid = GridSearchCV(rfc, param_grid, cv=kfold, scoring='f1_weighted', verbose=1, n_jobs=-1)
    for column in ['HorseWin', 'HorseRankTop3']:
        grid.fit(X_train, y_train[column].to_numpy())
        print(grid.best_params_)
        print(grid.best_score_)
        rfc = grid.best_estimator_

      
        model_name = str(rfc).split('(')[0]

        # Display results in Streamlit
        # st.write(f"## Results for model {model_name} and target variable {column}:")
        st.markdown(f"<small>Results for model {model_name} and target variable {column}:</small>", unsafe_allow_html=True)

        print("y train is",y_train)
        # Fit the model
        rfc.fit(X_train, y_train[column].to_numpy())

        # Calculate the cross-validation score
        cv_score = cross_val_score(rfc, X_train, y_train[column].to_numpy(), cv=kfold, scoring='f1_weighted').mean()
        cv_score = round(cv_score, 3)

        # Make predictions
        y_pred = rfc.predict(X_test)
        y_unseen_pred = rfc.predict(X_unseen)

        # Store the predictions in the dataframe
        df_pred[column] = y_unseen_pred

        # Calculate the f1 score
        f1 = f1_score(y_test[column].to_numpy(), y_pred, average='weighted')
        f1 = round(f1, 3)

        # Calculate PR AUC
        pr_auc = average_precision_score(y_test[column].to_numpy(), y_pred, average='weighted')
        pr_auc = round(pr_auc, 3)

        # Calculate Recall
        tn, fp, fn, tp = confusion_matrix(y_test[column].to_numpy(), y_pred).ravel()
        recall = tp / (tp + fn)
        recall = round(recall, 3)

        # Calculate Precision
        precision = tp / (tp + fp)
        precision = round(precision, 3)

        # Append the results to the dataframe
        df_results.loc[len(df_results)] = [model_name, column, cv_score, f1, pr_auc, recall, precision]

        # Display results in Streamlit
        st.write('**Cross Validation Score (F1-weighted):** ', cv_score)
        st.write('**F1 Score:** ', f1)
        st.write('**PR AUC (Avg Precision):** ', pr_auc)
        st.write('**Recall:** ', recall)
        st.write('**Precision:** ', precision)
        # Debugging output
        print("y_test type:", type(y_test))
        print("y_test columns:", y_test.columns if hasattr(y_test, 'columns') else 'Not a DataFrame')
        print("y_test head:\n", y_test.head())
        print("Current column:", column)

        print("y_test column values:\n", y_test[column].to_numpy())
        print("Shape of y_test[column]:", y_test[column].to_numpy().shape)
        print("Shape of y_pred:", y_pred.shape)
    
        # Display confusion matrix using Streamlit
        st.write(f"### Confusion Matrix for {column}")
        plot_confusion_matrix(y_test, y_pred, column)

        # # Plot precision recall curve
        

        # Capture and display the confusion matrix plot
        st.pyplot(plt.gcf())
        plt.clf()

        # Display precision-recall curve using Streamlit
        st.write(f"### Precision-Recall Curve for {column}")
        plot_pr_auc(X_test, y_test, lr, column)

        # Capture and display the precision-recall curve plot
        st.pyplot(plt.gcf())
        plt.clf()

    # Display the results dataframe in Streamlit
    st.write("### Model Results Summary")
    st.dataframe(df_results)
    return df_pred
def run_model_reg(model, X_train, y_train, X_test, y_test, X_unseen,df_test):
        results = pd.DataFrame(columns=['Model', 'RMSE_train', 'RMSE_test', 
                                'Generalization', 'Top1_Train_Accuracy', 'Top1_Test_Accuracy',
                                'Top3_Train_Accuracy', 'Top3_Test_Accuracy'])
        # Store model name
        model_name = model.__class__.__name__

        # param_grid_lgbm_regressor = {
        #     'num_leaves': [31, 50, 70],               # Maximum number of leaves in one tree
        #     'learning_rate': [0.01, 0.05, 0.1],       # Learning rate
        #     'n_estimators': [100, 200, 300],           # Number of boosting rounds
        #     'max_depth': [-1, 10, 20],                 # Maximum depth of trees (-1 means no limit)
        #     'min_child_samples': [20, 50, 100],        # Minimum number of samples required to be at a leaf node
        #     'subsample': [0.7, 0.8, 0.9],              # Fraction of samples used for training
        #     'colsample_bytree': [0.7, 0.8, 0.9],       # Fraction of features used per tree
        #     'reg_alpha': [0, 0.1, 0.5],                # L1 regularization term
        #     'reg_lambda': [0, 0.1, 0.5],                # L2 regularization term
        #     'min_split_gain': [0, 0.1, 0.2]            # Minimum gain to make a split
        # }

        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_unseen = scaler.transform(X_unseen)
        param_grid = {
            'n_estimators': [50, 100, 200, 300],  # Number of trees in the forest
            'max_depth': [None, 10, 20, 30, 40],  # Maximum depth of each tree
            'min_samples_split': [2, 5, 10],       # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4],         # Minimum number of samples required to be at a leaf node
            'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
            'bootstrap': [True, False]             # Whether bootstrap samples are used when building trees
        }     
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_root_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        # Predict on the training set
        y_train_pred = best_model.predict(X_train)
        y_train_pred = pd.DataFrame(y_train_pred)

        # Predict on the testing set
        y_test_pred = best_model.predict(X_test)
        y_test_pred = pd.DataFrame(y_test_pred)

        # Calculate the RMSE
        train_rmse = round(math.sqrt(mean_squared_error(y_train, y_train_pred)), 3)
        test_rmse = round(math.sqrt(mean_squared_error(y_test, y_test_pred)), 3)
        
        # Calculate the accuracy
        train_accuracy, train_accuracy_top3 = find_prob(y_train_pred,df_test)
        test_accuracy, test_accuracy_top3 = find_prob(y_test_pred,df_test)

        # Calculate generalization error percentage
        generalization_error = round((test_rmse - train_rmse)/train_rmse*100, 3)

        st.title('Model Evaluation Results')

        # Display Model Name
        st.header(f'Model Results for {model_name}:')

        # Display RMSE
        st.subheader('Root Mean Squared Error (RMSE)')
        st.write(f'Train RMSE: {train_rmse}')
        st.write(f'Test RMSE: {test_rmse}')
        st.write(f'Generalization Error: {generalization_error:.2f}%')

        # Display Accuracy
        st.subheader('Accuracy for Finding Top Positions')
        st.write(f'Train Accuracy for finding Top position: {train_accuracy:.2f}')
        st.write(f'Test Accuracy for finding Top position: {test_accuracy:.2f}')

        st.subheader('Accuracy for Finding Top 3 Positions')
        st.write(f'Train Accuracy for finding Top 3 positions: {train_accuracy_top3:.2f}')
        st.write(f'Test Accuracy for finding Top 3 positions: {test_accuracy_top3:.2f}')
        # Append the results to the dataframe
        results.loc[len(results)] = [model_name, train_rmse, test_rmse, generalization_error,
                                  train_accuracy, test_accuracy, train_accuracy_top3, test_accuracy_top3]
    
        # predict on unseen data
        y_unseen_pred = best_model.predict(X_unseen)
        y_unseen_pred = pd.DataFrame(y_unseen_pred)

        return y_unseen_pred
        
def run_model_UK(model,X_train, y_train, X_test, y_test,kfold):
    df_results = pd.DataFrame(columns=['Model', 'Target', 'CV F1-Score', 'F1 Score', 'PR AUC', 'Recall', 'Precision'])
    df_pred = pd.DataFrame()
    df_pred['RaceID'] = df_unseen['race_id']
    df_pred['HorseID'] = df_unseen['horse_name']
    
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                    'penalty': [ 'l2']}
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    for column in ['HorseWin', 'HorseRankTop3']:
        grid.fit(X_train, y_train[column].to_numpy())
        print(grid.best_params_)
        print(grid.best_score_)
        model = grid.best_estimator_
        print("Model coefficients:",model.coef_)
      

        model_name = str(model).split('(')[0]

        # Display results in Streamlit
        # st.write(f"## Results for model {model_name} and target variable {column}:")
        st.markdown(f"<small>Results for model {model_name} and target variable {column}:</small>", unsafe_allow_html=True)

        print("y train is",y_train)
        # Fit the model
        model.fit(X_train, y_train[column].to_numpy())

        # Calculate the cross-validation score
        cv_score = cross_val_score(model, X_train, y_train[column].to_numpy(), cv=kfold, scoring='f1_weighted').mean()
        cv_score = round(cv_score, 3)

        # Make predictions
        y_pred = model.predict(X_test)
        y_unseen_pred = model.predict(X_unseen)

        # Store the predictions in the dataframe
        df_pred[column] = y_unseen_pred

        # Calculate the f1 score
        f1 = f1_score(y_test[column].to_numpy(), y_pred, average='weighted')
        f1 = round(f1, 3)

        # Calculate PR AUC
        pr_auc = average_precision_score(y_test[column].to_numpy(), y_pred, average='weighted')
        pr_auc = round(pr_auc, 3)

        # Calculate Recall
        tn, fp, fn, tp = confusion_matrix(y_test[column].to_numpy(), y_pred).ravel()
        recall = tp / (tp + fn)
        recall = round(recall, 3)

        # Calculate Precision
        precision = tp / (tp + fp)
        precision = round(precision, 3)

        # Append the results to the dataframe
        df_results.loc[len(df_results)] = [model_name, column, cv_score, f1, pr_auc, recall, precision]

        # Display results in Streamlit
        st.write('**Cross Validation Score (F1-weighted):** ', cv_score)
        st.write('**F1 Score:** ', f1)
        st.write('**PR AUC (Avg Precision):** ', pr_auc)
        st.write('**Recall:** ', recall)
        st.write('**Precision:** ', precision)
        # Debugging output
        print("y_test type:", type(y_test))
        print("y_test columns:", y_test.columns if hasattr(y_test, 'columns') else 'Not a DataFrame')
        print("y_test head:\n", y_test.head())
        print("Current column:", column)

        print("y_test column values:\n", y_test[column].to_numpy())
        print("Shape of y_test[column]:", y_test[column].to_numpy().shape)
        print("Shape of y_pred:", y_pred.shape)
    
        # Display confusion matrix using Streamlit
        st.write(f"### Confusion Matrix for {column}")
        plot_confusion_matrix(y_test, y_pred, column)

        # # Plot precision recall curve
        

        # Capture and display the confusion matrix plot
        st.pyplot(plt.gcf())
        plt.clf()

        # Display precision-recall curve using Streamlit
        st.write(f"### Precision-Recall Curve for {column}")
        plot_pr_auc(X_test, y_test, model, column)

        # Capture and display the precision-recall curve plot
        st.pyplot(plt.gcf())
        plt.clf()

    # Display the results dataframe in Streamlit
    st.write("### Model Results Summary")
    st.dataframe(df_results)
    return df_pred

def group_horse_and_result(element):
    if element[0] == 'finish_position':
        return 200 + element[1] # to make sure results are put near the end
    else:
        return element[1]
    
def group_horse_and_result_UK(element):
    if element[0] == 'position':
        return 200 + element[1] # to make sure results are put near the end
    else:
        return element[1]

def simple_class_strategy(model_pred,df_unseen, graph=True):

    df_unseen_results = df_unseen[['finish_position', 'show_payoff', 'race_id',
                               'HorseWin', 'runners_horse_name','morning_line_odds']]



    # rename columns
    df_unseen_results = df_unseen_results.rename(columns={'race_id': 'RaceID',
                                                    'runners_horse_name': 'HorseID',
                                                    'HorseWin':'ActualWin',
                                                    })


    # merge the prediction with the test data
    df_backtest = pd.merge(model_pred, df_unseen_results, on=('RaceID', 'HorseID'), how='left')
    df_backtest.to_csv('./data/df_backtest.csv')
    df_backtest.dropna(inplace=True)
    money = 0
    bets_made = []
    cumulative_money = [0]

    for race_id in df_backtest['RaceID'].unique():

        # make a temporary dataframe one for that particular race
        temp_df = df_backtest[df_backtest['RaceID']==race_id]
        print("Temp DataFrame head:\n", temp_df.head())
        print("Temp DataFrame columns:\n", temp_df.columns)

        # find out the bets we made
        bets = temp_df[temp_df['HorseWin']==1]
        print("bets are",bets.size)

        # deduct money for bets we made
        deduction  = -2*len(bets)

        # amount won from bets
        # sum of multiplying the odds with the prediction
        amount_won = sum(bets['win_payoff']*bets['ActualWin'])

        # add the amount won to the money
        money += (amount_won + deduction)

        # append the money to the cumulative money list
        cumulative_money.append(money)

        # append the bets made to the bets made list
        bets_made.append(len(bets))
        print("bets made are",bets_made)

    if graph==True:
        # plot the cumulative money
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_money)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Cumulative Money for Every $2 Bet')
        plt.xlabel('Bets Made')
        plt.ylabel('Cumulative Money')
        display_plots()

        # plot the bets made
        plt.figure(figsize=(10, 6))
        plt.plot(bets_made)
        plt.title('Bets Made')
        display_plots()

    # print the final money and bets made
    st.write(f'**Final Money:** ${round(money, 3)}')
    st.write(f'**Total Bets Made:** {round(sum(bets_made), 3)}')

    return money, bets_made



def simple_class_strategy_UK(model_pred,df_unseen, graph=True):
    df_unseen_results = df_unseen[['position', 'tote_win', 'race_id',
                               'HorseWin', 'horse_name']]

    # rename columns
    df_unseen_results = df_unseen_results.rename(columns={'race_id': 'RaceID',
                                                    'horse_name': 'HorseID',
                                                    'HorseWin':'ActualWin'})
    
    

    # merge the prediction with the test data
    df_backtest = pd.merge(model_pred, df_unseen_results, on=('RaceID', 'HorseID'), how='left')
    df_backtest.dropna(inplace=True)
    money = 0
    bets_made = []
    cumulative_money = [0]

    for race_id in df_backtest['RaceID'].unique():

        # make a temporary dataframe one for that particular race
        temp_df = df_backtest[df_backtest['RaceID']==race_id]
        print("Temp DataFrame head:\n", temp_df.head())
        print("Temp DataFrame columns:\n", temp_df.columns)

        # find out the bets we made
        bets = temp_df[temp_df['HorseWin']==1]

        # deduct money for bets we made
        deduction  = -2*len(bets)

        # amount won from bets
        # sum of multiplying the odds with the prediction
        bets.loc[:, 'tote_win'] = bets['tote_win'].replace('[£,]', '', regex=True).astype(float)
        amount_won = sum(bets['tote_win']*bets['ActualWin'])

        # add the amount won to the money
        money += (amount_won + deduction)

        # append the money to the cumulative money list
        cumulative_money.append(money)

        # append the bets made to the bets made list
        bets_made.append(len(bets))

    if graph==True:
        # plot the cumulative money
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_money)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Cumulative Money for Every 2 Pound Bet')
        plt.xlabel('Bets Made')
        plt.ylabel('Cumulative Money')
        display_plots()

        # plot the bets made
        plt.figure(figsize=(10, 6))
        plt.plot(bets_made)
        plt.title('Bets Made')
        display_plots()

    # print the final money and bets made
    st.write(f'**Final Money:** ${round(money, 3)}')
    st.write(f'**Total Bets Made:** {round(sum(bets_made), 3)}')

    return money, bets_made

def simple_class_strategy_UK_2(model_pred,df_unseen, graph=True):
    df_unseen_results = df_unseen[['position', 'tote_win', 'race_id',
                               'HorseWin', 'horse_id']]

    # rename columns
    df_unseen_results = df_unseen_results.rename(columns={'race_id': 'RaceID',
                                                    'horse_id': 'HorseID',
                                                    'HorseWin':'ActualWin'})
    model_pred = model_pred.rename(columns={'race_id': 'RaceID',
                                                    'horse_id': 'HorseID'})
    # merge the prediction with the test data
    df_backtest = pd.merge(df_unseen_results, model_pred,on=('RaceID', 'HorseID'), how='left')
    df_backtest.dropna(inplace=True)
    money = 0
    bets_made = []
    cumulative_money = [0]

    for race_id in df_backtest['RaceID'].unique():

        # make a temporary dataframe one for that particular race
        temp_df = df_backtest[df_backtest['RaceID']==race_id]
        print("Temp DataFrame head:\n", temp_df.head())
        print("Temp DataFrame columns:\n", temp_df.columns)

        # find out the bets we made
        bets = temp_df[temp_df['HorseWin']==1]

        # deduct money for bets we made
        deduction  = -2*len(bets)

        # amount won from bets
        # sum of multiplying the odds with the prediction
        bets.loc[:, 'tote_win'] = bets['tote_win'].replace('[£,]', '', regex=True).astype(float)
        amount_won = sum(bets['tote_win']*bets['ActualWin'])

        # add the amount won to the money
        money += (amount_won + deduction)

        # append the money to the cumulative money list
        cumulative_money.append(money)

        # append the bets made to the bets made list
        bets_made.append(len(bets))

    if graph==True:
        # plot the cumulative money
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_money)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Cumulative Money for Every 2 Pound Bet')
        plt.xlabel('Bets Made')
        plt.ylabel('Cumulative Money')
        display_plots()

        # plot the bets made
        plt.figure(figsize=(10, 6))
        plt.plot(bets_made)
        plt.title('Bets Made')
        display_plots()

    # print the final money and bets made
    st.write(f'**Final Money:** ${round(money, 3)}')
    st.write(f'**Total Bets Made:** {round(sum(bets_made), 3)}')

    return money, bets_made






def top3_strategy(model_pred,df_unseen, graph=True):

    df_unseen_results = df_unseen[['finish_position', 'win_payoff', 'race_id',
                               'HorseWin', 'runners_horse_name']]

    # rename columns
    df_unseen_results = df_unseen_results.rename(columns={'race_id': 'RaceID',
                                                    'runners_horse_name': 'HorseID',
                                                    'HorseWin':'ActualWin'})

    # merge the prediction with the test data
    df_backtest = pd.merge(model_pred, df_unseen_results, on=('RaceID', 'HorseID'), how='left')

    df_backtest.dropna(inplace=True)
    money = 0
    bets_made = []
    cumulative_money = [0]

    for race_id in df_backtest['RaceID'].unique():

        # make a temporary dataframe one for that particular race
        temp_df = df_backtest[df_backtest['RaceID']==race_id]

        # find out the bets we made
        bets = temp_df[temp_df['HorseRankTop3']==1]

        # deduct money for bets we made
        deduction  = -2*len(bets)

        # amount won from bets
        # sum of multiplying the odds with the prediction
        amount_won = sum(bets['win_payoff']*bets['ActualWin'])

        # add the amount won to the money
        money += (amount_won + deduction)

        # append the money to the cumulative money list
        cumulative_money.append(money)

        # append the bets made to the bets made list
        bets_made.append(len(bets))

    if graph==True:
        # plot the cumulative money
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_money)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Cumulative Money for Every $2 Bet')
        plt.xlabel('Bets Made')
        plt.ylabel('Cumulative Money')
        display_plots()

        # plot the bets made
        plt.figure(figsize=(10, 6))
        plt.plot(bets_made)
        plt.title('Bets Made')
        display_plots()

    # print the final money and bets made
    st.write(f'**Final Money:** ${round(money, 3)}')
    st.write(f'**Total Bets Made:** {round(sum(bets_made), 3)}')

    return money, bets_made


def top3_strategy_UK(model_pred,df_unseen, graph=True):

    df_unseen_results = df_unseen[['position', 'tote_win', 'race_id',
                               'HorseWin', 'horse_name']]

    # rename columns
    df_unseen_results = df_unseen_results.rename(columns={'race_id': 'RaceID',
                                                    'horse_name': 'HorseID',
                                                    'HorseWin':'ActualWin'})

    # merge the prediction with the test data
    df_backtest = pd.merge(model_pred, df_unseen_results, on=('RaceID', 'HorseID'), how='left')
    df_backtest.dropna(inplace=True)
    money = 0
    bets_made = []
    cumulative_money = [0]

    for race_id in df_backtest['RaceID'].unique():

        # make a temporary dataframe one for that particular race
        temp_df = df_backtest[df_backtest['RaceID']==race_id]

        # find out the bets we made
        bets = temp_df[temp_df['HorseRankTop3']==1]

        # deduct money for bets we made
        deduction  = -2*len(bets)

        # amount won from bets
        # sum of multiplying the odds with the prediction
        bets.loc[:, 'tote_win'] = bets['tote_win'].replace('[£,]', '', regex=True).astype(float)
        amount_won = sum(bets['tote_win']*bets['ActualWin'])

        # add the amount won to the money
        money += (amount_won + deduction)

        # append the money to the cumulative money list
        cumulative_money.append(money)

        # append the bets made to the bets made list
        bets_made.append(len(bets))

    if graph==True:
        # plot the cumulative money
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_money)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Cumulative Money for Every $2 Bet')
        plt.xlabel('Bets Made')
        plt.ylabel('Cumulative Money')
        display_plots()

        # plot the bets made
        plt.figure(figsize=(10, 6))
        plt.plot(bets_made)
        plt.title('Bets Made')
        display_plots()

    # print the final money and bets made
    st.write(f'**Final Money:** ${round(money, 3)}')
    st.write(f'**Total Bets Made:** {round(sum(bets_made), 3)}')

    return money, bets_made


def remove_future_dates(database_url, model):
    # Create a database engine and session
    engine = sa.create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Get the current date
    current_date = datetime.now().date()

    try:
        # Query for entries with dates past the current date
        future_entries = session.query(model).filter(model.date > current_date).all()
        
        # Remove future entries
        for entry in future_entries:
            session.delete(entry)

        # Commit the changes
        session.commit()

        print(f"Removed {len(future_entries)} entries with dates past {current_date}")

    except Exception as e:
        session.rollback()
        print(f"Error removing future dates: {e}")
    
    finally:
        session.close()


def remove_future_dates_UK(database_url, model):
 
    # Get the current date
    current_date = datetime.now().date()

    try:
        # Query for entries with dates past the current date
        future_entries = session2.query(model).filter(model.date > current_date).all()
        
        # Remove future entries
        for entry in future_entries:
            session2.delete(entry)

        # Commit the changes
        session2.commit()

        print(f"Removed {len(future_entries)} entries with dates past {current_date}")

    except Exception as e:
        session2.rollback()
        print(f"Error removing future dates: {e}")
    
    finally:
        session2.close()

def remove_duplicates(database_url, model, unique_columns):
    """
    Remove duplicates from a table based on unique columns and print the rows that are removed.
    
    :param database_url: Database URL to connect to
    :param model: SQLAlchemy model (ORM class)
    :param unique_columns: List of columns that define uniqueness
    """
    # Create an engine and session
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Dynamically get the columns from the model
    unique_columns_expr = [getattr(model, col) for col in unique_columns]
    
    try:
        # Subquery to find duplicates
        subquery = (
            session.query(
                model.id,
                *unique_columns_expr,
                func.row_number().over(
                    partition_by=unique_columns_expr,
                    order_by=model.id
                ).label('row_number')
            ).subquery()
        )
        
        # Query to select duplicates (row_number > 1)
        duplicates = session.query(model).join(
            subquery, and_(*[getattr(model, col) == getattr(subquery.c, col) for col in unique_columns])
        ).filter(subquery.c.row_number > 1).all()
        
        # Print and delete duplicates
        for duplicate in duplicates:
            print(f"Removing duplicate: {duplicate.__dict__}")
            session.delete(duplicate)
        
        session.commit()
    finally:
        session.close()

def remove_duplicates_UK(database_url, model, unique_columns):
    """
    Remove duplicates from a table based on unique columns and print the rows that are removed.
    
    :param database_url: Database URL to connect to
    :param model: SQLAlchemy model (ORM class)
    :param unique_columns: List of columns that define uniqueness
    """
   
    engine2 = create_engine(database_url)
    Session2 = sessionmaker(bind=engine2)
    session2 = Session2()
    # Dynamically get the columns from the model
    unique_columns_expr = [getattr(model, col) for col in unique_columns]
    
    try:
        # Subquery to find duplicates
        subquery = (
            session2.query(
                model.id,
                *unique_columns_expr,
                func.row_number().over(
                    partition_by=unique_columns_expr,
                    order_by=model.id
                ).label('row_number')
            ).subquery()
        )
        
        # Query to select duplicates (row_number > 1)
        duplicates = session2.query(model).join(
            subquery, and_(*[getattr(model, col) == getattr(subquery.c, col) for col in unique_columns])
        ).filter(subquery.c.row_number > 1).all()
        
        # Print and delete duplicates
        for duplicate in duplicates:
            print(f"Removing duplicate: {duplicate.__dict__}")
            session2.delete(duplicate)
        
        session2.commit()
    finally:
        session2.close()

def compute_average_ranks(row, df_test, df_train):
    # Extract values from the row
    horse_name = row['runners_horse_name']
    jockey = row['jockey_full']
    trainer = row['trainer_full']
    owner=row['owner_full']
    # owner=row['owner_l_name']
    # Compute average rank for horse
    recent_ave_rank = df_train[df_train['runners_horse_name'] == horse_name]['recent_ave_rank'].mean()
    # Compute average rank for jockey
    jockey_ave_rank = df_train[df_train['jockey_full'] == jockey]['jockey_ave_rank'].mean()
    # Compute average rank for trainer
    trainer_ave_rank = df_train[df_train['trainer_full'] == trainer]['trainer_ave_rank'].mean()
    # owner_ave_rank = df_train[df_train['owner_full'] == owner]['owner_ave_rank'].mean()
    # win_percentage = df_train['recent_6_runs'].apply(calculate_win_percentage)
    # Return the computed values
    return pd.Series({
        'recent_ave_rank': recent_ave_rank,
        'jockey_ave_rank': jockey_ave_rank,
        'trainer_ave_rank': trainer_ave_rank,
    })

def compute_average_ranks_UK(row, df_test, df_train):
    # Extract values from the row
    horse_name = row['horse_id']
    jockey = row['jockey_id']
    trainer = row['trainer_id']
    owner= row['owner_id']
    
    # Compute average rank for horse
    recent_ave_rank = df_train[df_train['horse_id'] == horse_name]['recent_ave_rank'].mean()
    # Compute average rank for jockey
    jockey_ave_rank = df_train[df_train['jockey_id'] == jockey]['jockey_ave_rank'].mean()
    # Compute average rank for trainer
    trainer_ave_rank = df_train[df_train['trainer_id'] == trainer]['trainer_ave_rank'].mean()
    
    owner_ave_rank = df_train[df_train['owner_id'] == owner]['owner_ave_rank'].mean()
    
    # Return the computed values
    return pd.Series({
        'recent_ave_rank': recent_ave_rank,
        'jockey_ave_rank': jockey_ave_rank,
        'trainer_ave_rank': trainer_ave_rank,
        'owner_ave_rank':owner_ave_rank
    })

def compute_average_ranks_UK_last_6(row, df_train):
    # Extract values from the row
    horse_name = row['horse_id']
    jockey = row['jockey_id']
    trainer = row['trainer_id']
    owner = row['owner_id']

    # Compute average rank for horse based on the last 6 races
    horse_recent_ranks = df_train[df_train['horse_id'] == horse_name]['recent_ave_rank'].tail(6)
    recent_ave_rank = horse_recent_ranks.mean() if not horse_recent_ranks.empty else None

    # Compute average rank for jockey based on the last 6 races
    jockey_recent_ranks = df_train[df_train['jockey_id'] == jockey]['jockey_ave_rank'].tail(6)
    jockey_ave_rank = jockey_recent_ranks.mean() if not jockey_recent_ranks.empty else None

    # Compute average rank for trainer based on the last 6 races
    trainer_recent_ranks = df_train[df_train['trainer_id'] == trainer]['trainer_ave_rank'].tail(6)
    trainer_ave_rank = trainer_recent_ranks.mean() if not trainer_recent_ranks.empty else None

    # Compute average rank for owner based on the last 6 races
    owner_recent_ranks = df_train[df_train['owner_id'] == owner]['owner_ave_rank'].tail(6)
    owner_ave_rank = owner_recent_ranks.mean() if not owner_recent_ranks.empty else None

    # Return the computed values
    return pd.Series({
        'recent_ave_rank': recent_ave_rank,
        'jockey_ave_rank': jockey_ave_rank,
        'trainer_ave_rank': trainer_ave_rank,
        'owner_ave_rank': owner_ave_rank
    })

import pandas as pd

def compute_average_ranks_all_UK(row, df_test, df_train):
    # Extract values from the row
    horse_name = row['horse_id']
    jockey = row['jockey_id']
    trainer = row['trainer_id']
    owner = row['owner_id']
    
    # Compute average rank for horse
    recent_ave_rank = df_train[df_train['horse_id'] == horse_name]['recent_ave_rank'].mean()
    horse_last_finish_pos = df_train[df_train['horse_id'] == horse_name]['position'].iloc[-1:].mean()
    horse_second_last_finish_pos = df_train[df_train['horse_id'] == horse_name]['position'].iloc[-2:-1].mean()
    horse_third_last_finish_pos = df_train[df_train['horse_id'] == horse_name]['position'].iloc[-3:-2].mean()
    
    # Compute average rank for jockey
    jockey_ave_rank = df_train[df_train['jockey_id'] == jockey]['jockey_ave_rank'].mean()
    last_jockey_finish_pos = df_train[df_train['jockey_id'] == jockey]['position'].iloc[-1:].mean()
    second_last_jockey_finish_pos = df_train[df_train['jockey_id'] == jockey]['position'].iloc[-2:-1].mean()
    third_last_jockey_finish_pos = df_train[df_train['jockey_id'] == jockey]['position'].iloc[-3:-2].mean()
    
    # Compute average rank for trainer
    trainer_ave_rank = df_train[df_train['trainer_id'] == trainer]['trainer_ave_rank'].mean()
    last_trainer_finish_pos = df_train[df_train['trainer_id'] == trainer]['position'].iloc[-1:].mean()
    second_last_trainer_finish_pos = df_train[df_train['trainer_id'] == trainer]['position'].iloc[-2:-1].mean()
    third_last_trainer_finish_pos = df_train[df_train['trainer_id'] == trainer]['position'].iloc[-3:-2].mean()
    
    # Compute average rank for owner
    owner_ave_rank = df_train[df_train['owner_id'] == owner]['owner_ave_rank'].mean()
    last_owner_finish_pos = df_train[df_train['owner_id'] == owner]['position'].iloc[-1:].mean()
    second_last_owner_finish_pos = df_train[df_train['owner_id'] == owner]['position'].iloc[-2:-1].mean()
    third_last_owner_finish_pos = df_train[df_train['owner_id'] == owner]['position'].iloc[-3:-2].mean()
    
    # Return the computed values as a pd.Series
    return pd.Series({
        'recent_ave_rank': recent_ave_rank,
        'horse_last_pos': horse_last_finish_pos,
        'horse_second_last_pos': horse_second_last_finish_pos,
        'horse_third_last_pos': horse_third_last_finish_pos,
        'jockey_ave_rank': jockey_ave_rank,
        'last_jockey_finish_pos': last_jockey_finish_pos,
        'second_last_jockey_finish_pos': second_last_jockey_finish_pos,
        'third_last_jockey_finish_pos': third_last_jockey_finish_pos,
        'trainer_ave_rank': trainer_ave_rank,
        'last_trainer_finish_pos': last_trainer_finish_pos,
        'second_last_trainer_finish_pos': second_last_trainer_finish_pos,
        'third_last_trainer_finish_pos': third_last_trainer_finish_pos,
        'owner_ave_rank': owner_ave_rank,
        'last_owner_finish_pos': last_owner_finish_pos,
        'second_last_owner_finish_pos': second_last_owner_finish_pos,
        'third_last_owner_finish_pos': third_last_owner_finish_pos
    })

def compute_average_ranks_all_UK_with_win_percentage(row, df_test, df_train):
    # Extract values from the row
    horse_name = row['horse_id']
    jockey = row['jockey_id']
    trainer = row['trainer_id']
    owner = row['owner_id']
    
    # Helper function to compute average rank, recent positions, and percentages
    def get_rank_and_positions(df, entity_id, rank_col):
        filtered_df = df[df[entity_id] == row[entity_id]]
        ave_rank = filtered_df[rank_col].mean()
        if 'position' in filtered_df.columns:
            last_pos = filtered_df['position'].tail(1).mean()
            second_last_pos = filtered_df['position'].tail(2).head(1).mean()
            third_last_pos = filtered_df['position'].tail(3).head(1).mean()
            fourth_last_pos = filtered_df['position'].tail(4).head(1).mean()
            fifth_last_pos = filtered_df['position'].tail(5).head(1).mean()
            sixth_last_pos = filtered_df['position'].tail(6).head(1).mean()
        else:
            # If 'position' column is missing, set default values
            last_pos = 7
            second_last_pos = 7
            third_last_pos =7
            fourth_last_pos = 7
            fifth_last_pos = 7
            sixth_last_pos = 7
        
       
        # Calculate win and place percentages
        total_races = len(filtered_df)
        win_percentage = (filtered_df['position'] == 1).sum() / total_races * 100 if total_races > 0 else 0
        place_percentage = (filtered_df['position'] <= 2).sum() / total_races * 100 if total_races > 0 else 0

        return ave_rank, last_pos, second_last_pos, third_last_pos,fourth_last_pos,fifth_last_pos,sixth_last_pos, win_percentage, place_percentage

    # Compute values for horse
    recent_ave_rank, horse_last_finish_pos, horse_second_last_finish_pos, horse_third_last_finish_pos, \
    horse_fourth_last_finish_pos,horse_fifth_last_finish_pos,horse_sixth_last_finish_pos,horse_win_percentage, horse_place_percentage = get_rank_and_positions(df_train, 'horse_id', 'recent_ave_rank')

    # Compute values for jockey
    jockey_ave_rank, last_jockey_finish_pos, second_last_jockey_finish_pos, third_last_jockey_finish_pos, \
    fourth_last_jockey_finish_pos,fifth_last_jockey_finish_pos,sixth_last_jockey_finish_pos,jockey_win_percentage, jockey_place_percentage = get_rank_and_positions(df_train, 'jockey_id', 'jockey_ave_rank')

    # Compute values for trainer
    trainer_ave_rank, last_trainer_finish_pos, second_last_trainer_finish_pos, third_last_trainer_finish_pos, \
    fourth_last_trainer_finish_pos,fifth_last_trainer_finish_pos,sixth_last_trainer_finish_pos,trainer_win_percentage, trainer_place_percentage = get_rank_and_positions(df_train, 'trainer_id', 'trainer_ave_rank')

    # Compute values for owner
    owner_ave_rank, last_owner_finish_pos, second_last_owner_finish_pos, third_last_owner_finish_pos, \
    fourth_last_owner_finish_pos,fifth_last_owner_finish_pos,sixth_last_owner_finish_pos,owner_win_percentage, owner_place_percentage = get_rank_and_positions(df_train, 'owner_id', 'owner_ave_rank')

    # Return the computed values as a pd.Series
    return pd.Series({
        'recent_ave_rank': recent_ave_rank,
        'last_position_1': horse_last_finish_pos,
        'last_position_2': horse_second_last_finish_pos,
        'last_position_3': horse_third_last_finish_pos,
        'last_position_4': horse_fourth_last_finish_pos,
        'last_position_5': horse_fifth_last_finish_pos,
        'last_position_6': horse_sixth_last_finish_pos,
        'horse_win_percentage': horse_win_percentage,
        'horse_place_percentage': horse_place_percentage,
        'jockey_ave_rank': jockey_ave_rank,
        'last_jockey_finish_pos': last_jockey_finish_pos,
        'second_last_jockey_finish_pos': second_last_jockey_finish_pos,
        'third_last_jockey_finish_pos': third_last_jockey_finish_pos,
        'jockey_win_percentage': jockey_win_percentage,
        'jockey_place_percentage': jockey_place_percentage,
        'trainer_ave_rank': trainer_ave_rank,
        'last_trainer_finish_pos': last_trainer_finish_pos,
        'second_last_trainer_finish_pos': second_last_trainer_finish_pos,
        'third_last_trainer_finish_pos': third_last_trainer_finish_pos,
        'trainer_win_percentage': trainer_win_percentage,
        'trainer_place_percentage': trainer_place_percentage,
        'owner_ave_rank': owner_ave_rank,
        'last_owner_finish_pos': last_owner_finish_pos,
        'second_last_owner_finish_pos': second_last_owner_finish_pos,
        'third_last_owner_finish_pos': third_last_owner_finish_pos,
        'owner_win_percentage': owner_win_percentage,
        'owner_place_percentage': owner_place_percentage
    })

def plot_confusion_matrix(y_test, y_pred, column):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test[column].to_numpy(), y_pred)
    cm = pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for ' + column)
    plt.show()

def plot_pr_auc(X_test, y_test, model, column):
    # Get the probabilities of the predictions
    win_prob = model.predict_proba(X_test)[:, 1]

    # Get the precision and recall
    precision, recall, thresholds = precision_recall_curve(y_test[column], win_prob)

    # Calculate proportion of positive class
    proportion_pos_class = y_test[column].mean()

    # Print the PR AUC score
    pr_auc = round(auc(recall, precision),3)
    print(f'PR AUC score for {column}:', pr_auc)

    # Plot the PR AUC curve
    plt.figure(figsize = (8, 6))
    plt.plot([0, 1], [proportion_pos_class, proportion_pos_class], linestyle = '--')
    plt.plot(recall, precision, marker = '.')
    plt.title(f'PR AUC Curve for {column}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()



# Get range of dates for which to pull US data
def get_dates(start, end):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    # Adjust the timedelta and frequency according to your needs
    return pd.date_range(start_dt, end_dt, freq='d').strftime("%Y-%m-%d").tolist()

def cal_ave_rank(df):
    df['jockey_ave_rank'] = '7'
    for jockey in df['jockey_full'].unique():
        temp = df[df.jockey_full == jockey]['finish_position'].values.tolist()
        if temp:
            temp_ave = np.mean(list(map(int, temp)))
            df.loc[df.jockey_full == jockey, 'jockey_ave_rank'] = temp_ave
    df['trainer_ave_rank'] = '7'
    for trainer in df['trainer_full'].unique():
        temp = df[df.trainer_full == trainer]['finish_position'].values.tolist()
        if temp:
            temp_ave = np.mean(list(map(int, temp)))
            df.loc[df.trainer_full == trainer, 'trainer_ave_rank'] = temp_ave
    # df['owner_ave_rank'] = '7'
    # for owner in df['owner_l_name'].unique():
    #     temp = df[df.owner_l_name == owner]['finish_position'].values.tolist()
    #     if temp:
    #         temp_ave = np.mean(list(map(int, temp)))
    #         df.loc[df.owner_l_name == owner, 'owner_ave_rank'] = temp_ave
    return df


import numpy as np
import pandas as pd

def compute_average_ranks_all_UK(row, df_test, df_train):
    # Extract values from the row
    horse_name = row['horse_id']
    jockey = row['jockey_id']
    trainer = row['trainer_id']
    owner = row['owner_id']
    
    # Compute average rank for horse
    recent_ave_rank = df_train[df_train['horse_id'] == horse_name]['recent_ave_rank'].mean()
    horse_last_pos = df_train[df_train['horse_id'] == horse_name]['position'].iloc[-1:].mean()
    horse_second_last_pos = df_train[df_train['horse_id'] == horse_name]['position'].iloc[-2:-1].mean()
    horse_third_last_pos = df_train[df_train['horse_id'] == horse_name]['position'].iloc[-3:-2].mean()
    
    # Compute average rank for jockey
    jockey_ave_rank = df_train[df_train['jockey_id'] == jockey]['jockey_ave_rank'].mean()
    jockey_last_pos = df_train[df_train['jockey_id'] == jockey]['position'].iloc[-1:].mean()
    jockey_second_last_pos = df_train[df_train['jockey_id'] == jockey]['position'].iloc[-2:-1].mean()
    jockey_third_last_pos = df_train[df_train['jockey_id'] == jockey]['position'].iloc[-3:-2].mean()
    
    # Compute average rank for trainer
    trainer_ave_rank = df_train[df_train['trainer_id'] == trainer]['trainer_ave_rank'].mean()
    trainer_last_pos = df_train[df_train['trainer_id'] == trainer]['position'].iloc[-1:].mean()
    trainer_second_last_pos = df_train[df_train['trainer_id'] == trainer]['position'].iloc[-2:-1].mean()
    trainer_third_last_pos = df_train[df_train['trainer_id'] == trainer]['position'].iloc[-3:-2].mean()
    
    # Compute average rank for owner
    owner_ave_rank = df_train[df_train['owner_id'] == owner]['owner_ave_rank'].mean()
    owner_last_pos = df_train[df_train['owner_id'] == owner]['position'].iloc[-1:].mean()
    owner_second_last_pos = df_train[df_train['owner_id'] == owner]['position'].iloc[-2:-1].mean()
    owner_third_last_pos = df_train[df_train['owner_id'] == owner]['position'].iloc[-3:-2].mean()
    
    # Return the computed values as a pd.Series with updated titles
    return pd.Series({
        'recent_ave_rank': recent_ave_rank,
        'horse_last_pos': horse_last_pos,
        'horse_second_last_pos': horse_second_last_pos,
        'horse_third_last_pos': horse_third_last_pos,
        'jockey_ave_rank': jockey_ave_rank,
        'jockey_last_pos': jockey_last_pos,
        'jockey_second_last_pos': jockey_second_last_pos,
        'jockey_third_last_pos': jockey_third_last_pos,
        'trainer_ave_rank': trainer_ave_rank,
        'trainer_last_pos': trainer_last_pos,
        'trainer_second_last_pos': trainer_second_last_pos,
        'trainer_third_last_pos': trainer_third_last_pos,
        'owner_ave_rank': owner_ave_rank,
        'owner_last_pos': owner_last_pos,
        'owner_second_last_pos': owner_second_last_pos,
        'owner_third_last_pos': owner_third_last_pos
    })
import numpy as np

def simp_cal_ave_rank_all_UK(df):
    # Initialize new columns with NaN or a default value
    df['recent_ave_rank'] = 7
    df['jockey_ave_rank'] = 7
    df['trainer_ave_rank'] = 7
    df['owner_ave_rank'] = 7

    # Compute average ranks for horses (recent_ave_rank)
    for horse in df['horse_id'].unique():
        horse_data = df[df['horse_id'] == horse].sort_values(by='date')  # Sort by date
        positions = horse_data['position'].values[-3:]  # Only last 6 positions
        
        if len(positions) > 0:
            df.loc[df['horse_id'] == horse, 'recent_ave_rank'] = np.mean(list(map(int, positions)))

    # Compute average ranks for jockeys (jockey_ave_rank)
    for jockey in df['jockey_id'].unique():
        jockey_data = df[df['jockey_id'] == jockey].sort_values(by='date')
        positions = jockey_data['position'].values[-3:]  # Only last 6 positions
        
        if positions.size > 0:
            jockey_ave = np.mean(list(map(int, positions)))
            df.loc[df['jockey_id'] == jockey, 'jockey_ave_rank'] = jockey_ave

    # Compute average ranks for trainers (trainer_ave_rank)
    for trainer in df['trainer_id'].unique():
        trainer_data = df[df['trainer_id'] == trainer].sort_values(by='date')
        positions = trainer_data['position'].values[-3:]  # Only last 6 positions
        
        if positions.size > 0:
            trainer_ave = np.mean(list(map(int, positions)))
            df.loc[df['trainer_id'] == trainer, 'trainer_ave_rank'] = trainer_ave

    # Compute average ranks for owners (owner_ave_rank)
    for owner in df['owner_id'].unique():
        owner_data = df[df['owner_id'] == owner].sort_values(by='date')
        positions = owner_data['position'].values[-3:]  # Only last 6 positions
        
        if positions.size > 0:
            owner_ave = np.mean(list(map(int, positions)))
            df.loc[df['owner_id'] == owner, 'owner_ave_rank'] = owner_ave

    return df


def cal_ave_rank_all_UK(df):
    # Initialize new columns with NaN
    df['recent_ave_rank'] = 7
    df['horse_last_pos'] = 7
    df['horse_second_last_pos'] = 7
    df['horse_third_last_pos'] = 7
    df['jockey_ave_rank'] = 7
    df['jockey_last_pos'] = 7
    df['jockey_second_last_pos'] =7
    df['jockey_third_last_pos'] = 7
    df['trainer_ave_rank'] = 7
    df['trainer_last_pos'] = 7
    df['trainer_second_last_pos'] = 7
    df['trainer_third_last_pos'] = 7
    df['owner_ave_rank'] = 7
    df['owner_last_pos'] =7
    df['owner_second_last_pos'] =7
    df['owner_third_last_pos'] = 7

    # Compute average ranks and positions for horses
    for horse in df['horse_id'].unique():
        horse_data = df[df['horse_id'] == horse].sort_values(by='date')  # Sort by date
        positions = horse_data['position'].values.tolist()

        # Total number of races for the horse
        total_races = len(horse_data)
        
        # Calculate the number of wins (position == 1) and places (position <= 2)
        wins = sum(horse_data['position'] == 1)
        places = sum(horse_data['position'] <= 2)
        
        # Calculate win and place percentages
        win_percentage = (wins / total_races) * 100 if total_races > 0 else 0
        place_percentage = (places / total_races) * 100 if total_races > 0 else 0
        
        # Assign the calculated percentages to the relevant rows in the DataFrame
        df.loc[df['horse_id'] == horse, 'horse_win_percentage'] = win_percentage
        df.loc[df['horse_id'] == horse, 'horse_place_percentage'] = place_percentage


        if len(positions) > 0:
            df.loc[df['horse_id'] == horse, 'recent_ave_rank'] = np.mean(list(map(int, positions)))
            df.loc[df['horse_id'] == horse, 'horse_last_pos'] = positions[-1]
        if len(positions) > 1:
            df.loc[df['horse_id'] == horse, 'horse_second_last_pos'] = positions[-2]
        if len(positions) > 2:
            df.loc[df['horse_id'] == horse, 'horse_third_last_pos'] = positions[-3]

    # Compute average ranks for jockeys, trainers, and owners
    for jockey in df['jockey_id'].unique():
        jockey_data = df[df['jockey_id'] == jockey].sort_values(by='date')
        temp = df[df['jockey_id'] == jockey]['position'].values.tolist()
        if temp:
                # Total number of races for the horse
            total_races = len(jockey_data)
            
            # Calculate the number of wins (position == 1) and places (position <= 2)
            wins = sum(jockey_data['position'] == 1)
            places = sum(jockey_data['position'] <= 2)
            
            # Calculate win and place percentages
            win_percentage = (wins / total_races) * 100 if total_races > 0 else 0
            place_percentage = (places / total_races) * 100 if total_races > 0 else 0
            
            # Assign the calculated percentages to the relevant rows in the DataFrame
            df.loc[df['jockey_id'] == jockey, 'jockey_win_percentage'] = win_percentage
            df.loc[df['jockey_id'] == jockey, 'jockey_place_percentage'] = place_percentage
            jockey_ave = np.mean(list(map(int, temp)))
            df.loc[df['jockey_id'] == jockey, 'jockey_ave_rank'] = jockey_ave
            df.loc[df['jockey_id'] == jockey, 'jockey_last_pos'] = temp[-1]
            if len(temp) > 1:
                df.loc[df['jockey_id'] == jockey, 'jockey_second_last_pos'] = temp[-2]
            if len(temp) > 2:
                df.loc[df['jockey_id'] == jockey, 'jockey_third_last_pos'] = temp[-3]

    for trainer in df['trainer_id'].unique():
        trainer_data = df[df['trainer_id'] == trainer].sort_values(by='date')
        temp = df[df['trainer_id'] == trainer]['position'].values.tolist()
        if temp:
                 # Total number of races for the horse
            total_races = len(trainer_data)
            
            # Calculate the number of wins (position == 1) and places (position <= 2)
            wins = sum(trainer_data['position'] == 1)
            places = sum(trainer_data['position'] <= 2)
            
            # Calculate win and place percentages
            win_percentage = (wins / total_races) * 100 if total_races > 0 else 0
            place_percentage = (places / total_races) * 100 if total_races > 0 else 0
            
            # Assign the calculated percentages to the relevant rows in the DataFrame
            df.loc[df['trainer_id'] == trainer, 'trainer_win_percentage'] = win_percentage
            df.loc[df['trainer_id'] == trainer, 'trainer_place_percentage'] = place_percentage
            trainer_ave = np.mean(list(map(int, temp)))
            df.loc[df['trainer_id'] == trainer, 'trainer_ave_rank'] = trainer_ave
            df.loc[df['trainer_id'] == trainer, 'trainer_last_pos'] = temp[-1]
            if len(temp) > 1:
                df.loc[df['trainer_id'] == trainer, 'trainer_second_last_pos'] = temp[-2]
            if len(temp) > 2:
                df.loc[df['trainer_id'] == trainer, 'trainer_third_last_pos'] = temp[-3]

    for owner in df['owner_id'].unique():
        temp = df[df['owner_id'] == owner]['position'].values.tolist()
        if temp:
            owner_ave = np.mean(list(map(int, temp)))
            df.loc[df['owner_id'] == owner, 'owner_ave_rank'] = owner_ave
            df.loc[df['owner_id'] == owner, 'owner_last_pos'] = temp[-1]
            if len(temp) > 1:
                df.loc[df['owner_id'] == owner, 'owner_second_last_pos'] = temp[-2]
            if len(temp) > 2:
                df.loc[df['owner_id'] == owner, 'owner_third_last_pos'] = temp[-3]

    return df

def cal_ave_rank_UK(df):
    df['jockey_ave_rank'] = '7'
    for jockey in df['jockey_id'].unique():
        temp = df[df.jockey_id == jockey]['position'].values.tolist()
        if temp:
            temp_ave = np.mean(list(map(int, temp)))
            df.loc[df.jockey_id == jockey, 'jockey_ave_rank'] = temp_ave

    df['trainer_ave_rank'] = '7'
    for trainer in df['trainer_id'].unique():
        temp = df[df.trainer_id == trainer]['position'].values.tolist()
        if temp:
            temp_ave = np.mean(list(map(int, temp)))
            df.loc[df.trainer_id == trainer, 'trainer_ave_rank'] = temp_ave
    df['owner_ave_rank'] = '7'
    for owner in df['owner_id'].unique():
        temp = df[df.owner_id == owner]['position'].values.tolist()
        if temp:
            temp_ave = np.mean(list(map(int, temp)))
            df.loc[df.owner_id == owner, 'owner_ave_rank'] = temp_ave
    return df

# def cal_last_six_positions(df):
#     # Create six columns for the last six finishing positions of each horse
#     for i in range(1, 7):
#         df[f'last_position_{i}'] = np.nan

#     for horse_id in df['horse_id'].unique():
#         # Get the last six positions for each horse
#         horse_positions = df[df.horse_id == horse_id]['position'].astype(int).values[-6:]

#         # Fill the columns with the last six positions, if available
#         for i, position in enumerate(horse_positions[::-1], start=1):
#             df.loc[df.horse_id == horse_id, f'last_position_{i}'] = position

#     return df

def cal_last_six_positions(df):
    # Set default value for missing positions
    default_position = 7

    # Create six columns for the last six finishing positions of each horse, defaulted to `7`
    for i in range(1, 7):
        df[f'last_position_{i}'] = default_position

    # Only proceed if 'position' column exists
    if 'position' in df.columns:
        for horse_id in df['horse_id'].unique():
            # Get the last six positions for each horse, padding with `7` if fewer than six positions
            horse_positions = df[df.horse_id == horse_id]['position'].astype(int).values[-6:]
            horse_positions = np.pad(horse_positions, (6 - len(horse_positions), 0), 
                                     'constant', constant_values=default_position)

            # Fill the columns with the last six positions
            for i, position in enumerate(horse_positions[::-1], start=1):
                df.loc[df.horse_id == horse_id, f'last_position_{i}'] = position

    return df

def cal_ave_rank_UK_last_6(df):
    # Initialize average rank columns with '7'
    df['jockey_ave_rank'] = 7
    df['trainer_ave_rank'] = 7
    df['owner_ave_rank'] = 7

    # Calculate average rank for jockeys
    for jockey in df['jockey_id'].unique():
        # Get positions for the last 6 races of the jockey
        temp = df[df.jockey_id == jockey]['position'].tail(6).values.tolist()
        if temp:
            temp_ave = np.mean(list(map(int, temp)))
            df.loc[df.jockey_id == jockey, 'jockey_ave_rank'] = temp_ave

    # Calculate average rank for trainers
    for trainer in df['trainer_id'].unique():
        # Get positions for the last 6 races of the trainer
        temp = df[df.trainer_id == trainer]['position'].tail(6).values.tolist()
        if temp:
            temp_ave = np.mean(list(map(int, temp)))
            df.loc[df.trainer_id == trainer, 'trainer_ave_rank'] = temp_ave

    # Calculate average rank for owners
    for owner in df['owner_id'].unique():
        # Get positions for the last 6 races of the owner
        temp = df[df.owner_id == owner]['position'].tail(6).values.tolist()
        if temp:
            temp_ave = np.mean(list(map(int, temp)))
            df.loc[df.owner_id == owner, 'owner_ave_rank'] = temp_ave
    return df
# if 'df_train' not in st.session_state:
#     st.session_state.df_train = None


# Function to store race data in the database
from sqlalchemy.exc import IntegrityError







################# METHODS FOR LSTM #################################
import torch.nn as nn

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length, :-1]  # All columns except the last one (features)
        y = data[i + seq_length, -1]     # The last column (target)
        xs.append(x)
        ys.append(y)
         # Print shapes of sequences and targets
    # for i, seq in enumerate(xs):
    #     print(f'Sequence {i} shape: {seq.shape}')
    # for i, target in enumerate(ys):
    #     print(f'Target {i} shape: {target.shape}')
    return np.array(xs), np.array(ys)

def create_test_sequences(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length, :-1]  # All columns except the last one (features)
        xs.append(x)
    return np.array(xs)
def prepare_data(df, seq_length,specific_horse_name=None):
    sequences = []
    targets = []
    for horse_id, group in df.groupby('runners_horse_name'):
        if specific_horse_name is not None and horse_id != specific_horse_name:
            continue
        print(f'Horse ID: {horse_id}')
        data = group[['finish_position','HorseWin']].values
        x, y = create_sequences(data, seq_length)
        if y.size == 0:
            continue  # Skip if targets are empty
        sequences.append(x)
        # print("SEQUENCES ARE",sequences)
        targets.append(y)
    return np.concatenate(sequences, axis=0), np.concatenate(targets, axis=0)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the output from the last time step
        return out
    






################ BEGINNING OF APP  #####################################
from sqlalchemy import create_engine, func
# Streamlit App

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Uncial+Antiqua&display=swap" rel="stylesheet">
    <style>
    body {
        background-color: blue;
        color: white;
        font-family: 'Uncial Antiqua', serif;  /* Apply another Gothic-like font */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# st.markdown("<h1 style='color: red;'>Horse Racing Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='color: red; font-family:\"Uncial Antiqua\", sans-serif;'>Horse Racing Predictor</h1>", unsafe_allow_html=True)

########################### UK  ##############################################
from lightgbm import LGBMRegressor

col1, col2 = st.columns(2)
with col1:
    st.header("GB")
    st.subheader("UPDATE & PROCESS")
    if st.button('UPDATE: GB Database'):
        latest_date = session2.query(func.max(Race.date)).scalar()
        latest_date = latest_date.split(' ')[0]

        # latest_date='2023-10-23'
        start_date=latest_date
        # start_date='2022-11-11'
        end_date='2024-11-30'
        # import sqlalchemy as sa
        # dates = get_dates(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        process_uk_data(start_date=latest_date, end_date=end_date)
        # asyncio.run(main())
        database_url = "sqlite:///uk.db"
        races_unique_columns = ['race_id']
        runners_unique_columns = ['race_id','horse_id']
        # remove_duplicates_UK(database_url, Race, races_unique_columns)
        # remove_duplicates_UK(database_url, Runner, runners_unique_columns)
        # remove_future_dates_UK(database_url, Race)
    if st.button('UPDATE: GB Racecards'):
        # latest_date = session2.query(func.max(Race.date)).scalar()
        # latest_date = latest_date.split(' ')[0]
        # latest_date='2023-10-23'
        start_date='2023-01-01'
        end_date='2024-11-30'
        # import sqlalchemy as sa
        dates = get_dates(start_date, end_date)
        fetch_and_write_racecards_db(dates)
        # asyncio.run(main())
        database_url = "sqlite:///race_cards.db"
        races_unique_columns = ['race_id']
        runners_unique_columns = ['race_id','horse_id']
        # remove_duplicates_UK(database_url, Race, races_unique_columns)
        # remove_duplicates_UK(database_url, Runner, runners_unique_columns)
        # remove_future_dates_UK(database_url, Race)
    if st.button('GET: Upcoming GB Races'):
        now = datetime.now()
        # next_day_1 = datetime.strptime(next_day_1, "%Y-%m-%d")
        # next_day_2 = datetime.strptime(next_day_2, "%Y-%m-%d")
        next_day_1 = now-timedelta(days=4)
        next_day_2 = now -timedelta(days=4)
        now_str = now.strftime("%Y-%m-%d")
        next_day_1_str = next_day_1.strftime("%Y-%m-%d")
        next_day_2_str = next_day_2.strftime("%Y-%m-%d")
        dates2 = get_dates(next_day_1_str, next_day_2_str)
        fetch_and_write_racecards_csv(dates2)
        entries_pred = pd.read_csv('./data/racecards_pred.csv')
        results_pred=pd.read_csv('./data/racecards_runners_pred.csv')
        df_test = pd.merge(entries_pred, results_pred, on=['race_id'], how='outer')
        df_test.to_csv('./data/df_test_UK.csv', index=False)
        
    if st.button('Compare Spreadsheets'):
        backtest = pd.read_csv("./data/df_test_backtest.csv")
        predict = pd.read_csv("./data/df_test_predict.csv")

        # Extract the race ID columns (replace 'race_id' with the actual column name)
        race_ids1 = backtest['race_id']
        race_ids2 = predict['race_id']

        # Convert the columns to sets to compare
        set1 = set(race_ids1)
        set2 = set(race_ids2)

        # Find race IDs in set1 but not in set2
        unique_to_sheet1 = set1 - set2

        # Find race IDs in set2 but not in set1
        unique_to_sheet2 = set2 - set1

        # Optionally, convert results back to lists for easier handling
        unique_to_sheet1 = list(unique_to_sheet1)
        unique_to_sheet2 = list(unique_to_sheet2)

        st.write(unique_to_sheet1)
        st.write(unique_to_sheet2)
       

    if st.button('PROCESS: GB races'):
        export_tables_to_csv()
        races_uk= pd.read_csv('races.csv')
        runners_uk=pd.read_csv('runners.csv')
        races = pd.merge(races_uk, runners_uk, on=['race_id'], how='outer')
        races['position'] = pd.to_numeric(races['position'], errors='coerce')
        if races is not None:
            # st.write(races.head()
            print("processing")
        else:
            st.write("Error occurred while fetching and merging data.")
        #PROCESS TRAIN DATA
        df_horse = races[(races['position'] >= 1) & (races['position'] <= 14)].reset_index(drop=True)
        jockey = df_horse['jockey_id'].unique()
        numJockey = len(jockey)
        jockey_index = range(numJockey)
        trainer = df_horse['trainer_id'].unique()
        numTrainer = len(trainer)
        trainer_index = range(numTrainer)
        # Initialize columns
        df_horse['recent_6_runs'] = '0'
        df_horse['recent_ave_rank'] = '7'
        # Group by 'horse_name' and iterate over each group
        for horse_name, group in df_horse.groupby('horse_id'):
            # Get recent ranks for the horse
            recent_ranks = group['position'].tail(6)  # Get the last 6 ranks
            recent_ranks_list = recent_ranks.astype(str).tolist()  # Convert to list of strings
            # Assign recent ranks to 'recent_6_runs'
            df_horse.loc[group.index, 'recent_6_runs'] = '/'.join(recent_ranks_list)
            # Calculate average rank if recent_ranks is not empty
            if not recent_ranks.empty:
                recent_ave_rank = recent_ranks.mean()
                df_horse.loc[group.index, 'recent_ave_rank'] = recent_ave_rank
        # Convert 'recent_ave_rank' to float
        df_horse['recent_ave_rank'] = df_horse['recent_ave_rank'].astype(float)
        df_horse['position'] = pd.to_numeric(df_horse['position'], errors='coerce')
        # HorseWin
        df_horse['HorseWin'] = (df_horse['position'] == 1).astype(int)
        df_horse['HorseSecond']=(df_horse['position'] == 2).astype(int)
        # HorseRankTop3
        df_horse['HorseRankTop3'] = (df_horse['position'].isin([1, 2, 3])).astype(int)
        # HorseRankTop50Percent
        top_finishes = df_horse.loc[df_horse['position'] == 1].index
        top50_finishes = [idx + round((top_finishes[min(i+1, len(top_finishes)-1)] - idx) * 0.5) for i, idx in enumerate(top_finishes)]
        for i in top50_finishes:
            df_horse.loc[i:min(i+5, len(df_horse)-1), 'HorseRankTop50Percent'] = 1
        # Fill remaining NaN values with 0
        df_horse['HorseRankTop50Percent'].fillna(0, inplace=True)
        df_horse.to_csv('./data/df_horse_UK.csv', index=False)
        df_horse.reset_index(drop=True, inplace=True)
        df_horse['date'] = pd.to_datetime(df_horse['date'])
        # Apply your function cal_ave_rank
        df_horse=cal_ave_rank_all_UK(df_horse)
        df_train = cal_last_six_positions(df_horse)
        df_train.to_csv('./data/df_train_UK_results.csv', index=False)
    
    def winning_time_to_seconds(time_str):
        try:
            # Regular expression to match and extract minutes and seconds
            match = re.match(r"(\d+)m (\d+\.?\d*)s", time_str)
            if match:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                return minutes * 60 + seconds
            return None
        except Exception as e:
            print(f"Error converting time '{time_str}': {e}")
            return None
    
    if st.button('PROCESS: GB racecards'):
        export_racecards_tables_to_csv()
        races_uk= pd.read_csv('racecards.csv')
        runners_uk=pd.read_csv('runners_rc.csv')
        df_train=pd.read_csv('./data/df_train_UK_results.csv')
        races = pd.merge(races_uk, runners_uk, on=['race_id'], how='outer')
        races.to_csv('./data/df_train_UK_racecards_beginning.csv', index=False)
        st.write(races.head(4000))
        races=pd.merge(races,df_train[['race_id', 'horse_id', 'position','time','ovr_btn','tote_win']] ,on=['race_id','horse_id'],how='left')
        st.write("races are",races)
        if races is not None:
            # st.write(races.head()
            print("processing")
        else:
            st.write("Error occurred while fetching and merging data.")

        races.to_csv('./data/df_train_UK_racecards.csv', index=False)
        races['position'] = pd.to_numeric(races['position'], errors='coerce')
        if races is not None:
            # st.write(races.head()
            print("processing")
        else:
            st.write("Error occurred while fetching and merging data.")
        #PROCESS TRAIN DATA
        df_horse = races[(races['position'] >= 1) & (races['position'] <= 14)].reset_index(drop=True)
        jockey = df_horse['jockey_id'].unique()
        numJockey = len(jockey)
        jockey_index = range(numJockey)
        trainer = df_horse['trainer_id'].unique()
        numTrainer = len(trainer)
        trainer_index = range(numTrainer)
        # Initialize columns
        df_horse['recent_6_runs'] = '0'
        df_horse['recent_ave_rank'] = '7'
        # Group by 'horse_name' and iterate over each group
        for horse_name, group in df_horse.groupby('horse_id'):
            # Get recent ranks for the horse
            recent_ranks = group['position'].tail(6)  # Get the last 6 ranks
            recent_ranks_list = recent_ranks.astype(str).tolist()  # Convert to list of strings
            # Assign recent ranks to 'recent_6_runs'
            df_horse.loc[group.index, 'recent_6_runs'] = '/'.join(recent_ranks_list)
            # Calculate average rank if recent_ranks is not empty
            if not recent_ranks.empty:
                recent_ave_rank = recent_ranks.mean()
                df_horse.loc[group.index, 'recent_ave_rank'] = recent_ave_rank
        # Convert 'recent_ave_rank' to float
        df_horse['recent_ave_rank'] = df_horse['recent_ave_rank'].astype(float)
        df_horse['position'] = pd.to_numeric(df_horse['position'], errors='coerce')
        # HorseWin
        df_horse['HorseWin'] = (df_horse['position'] == 1).astype(int)
        df_horse['HorseSecond']=(df_horse['position'] == 2).astype(int)
        # HorseRankTop3
        df_horse['HorseRankTop3'] = (df_horse['position'].isin([1, 2, 3])).astype(int)
        # HorseRankTop50Percent
        top_finishes = df_horse.loc[df_horse['position'] == 1].index
        top50_finishes = [idx + round((top_finishes[min(i+1, len(top_finishes)-1)] - idx) * 0.5) for i, idx in enumerate(top_finishes)]
        for i in top50_finishes:
            df_horse.loc[i:min(i+5, len(df_horse)-1), 'HorseRankTop50Percent'] = 1
        # Fill remaining NaN values with 0
        df_horse['HorseRankTop50Percent'].fillna(0, inplace=True)
        df_horse.to_csv('./data/df_horse_UK_racecards.csv', index=False)
        df_horse.reset_index(drop=True, inplace=True)
        df_horse['date'] = pd.to_datetime(df_horse['date'])
        # Apply your function cal_ave_rank
        df_horse=cal_ave_rank_all_UK(df_horse)
        df_train = cal_last_six_positions(df_horse)
        df_train.to_csv('./data/df_train_UK_racecards.csv', index=False)




    def winning_time_to_seconds(time_str):
        try:
            # Regular expression to match and extract minutes and seconds
            match = re.match(r"(\d+)m (\d+\.?\d*)s", time_str)
            if match:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                return minutes * 60 + seconds
            return None
        except Exception as e:
            print(f"Error converting time '{time_str}': {e}")
            return None
    st.subheader("BACKTEST")
    if st.button('BACKTEST: GB WIN with Logistic Regression' ):
        df_results = pd.DataFrame(columns=['Model', 'Prediction', 'CV-F1', 'F1 Score', 'AUC', 'Recall', 'Precision'])
        # Get unique race IDs
        df_horse=pd.read_csv('./data/df_train_UK.csv')
        unique_race_ids = df_horse['race_id'].unique()
        # Split the race IDs into training, test, and unseen sets
        train_race_ids, test_race_ids = train_test_split(unique_race_ids, test_size=0.2, random_state=42)
        test_race_ids, unseen_race_ids = train_test_split(test_race_ids, test_size=0.5, random_state=42)
        # Split the data based on the race IDs
        df_train_80= df_horse[df_horse['race_id'].isin(train_race_ids)]
        df_test_20 = df_horse[df_horse['race_id'].isin(test_race_ids)]
        df_unseen = df_horse[df_horse['race_id'].isin(unseen_race_ids)]

        df_train_80.to_csv('./data/df_train_80_UK.csv', index=False)
        df_test_20.to_csv('./data/df_test_20_UK.csv', index=False)
        df_unseen.to_csv('./data/df_unseen_UK.csv', index=False)

        df_train_80.reset_index(inplace=True, drop=True)
        df_test_20.reset_index(inplace=True, drop=True)
        df_unseen.reset_index(inplace=True, drop=True)

        df_cleaned = df_train_80.dropna(subset=[  'draw','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        df_train_80 = df_cleaned.reset_index(drop=True)
        df_cleaned = df_test_20.dropna(subset=[  'draw','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        df_test_20 = df_cleaned.reset_index(drop=True)

        df_cleaned = df_unseen.dropna(subset=[  'draw','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        df_unseen= df_cleaned.reset_index(drop=True)
        X_train = df_train_80[[ 'draw',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank']]
        X_test = df_test_20[[ 'draw',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank']]
        X_unseen=df_unseen[[  'draw',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank']]
       
        y_train = df_train_80[['HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']]
        y_test=df_test_20[['HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']]
        y_unseen=df_unseen[['HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']]
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)
        print("y_unseen shape:", y_unseen.shape)
        #MAKE PREDICTIONS WITH LR
        kfold = KFold(n_splits=5)
        model =  LogisticRegression(solver='lbfgs', penalty='l2', C=1, max_iter=500)
        lr_pred = run_model_UK(model,X_train, y_train, X_test, y_test, kfold)
        class_pred_dict = {'Logistic Regression': lr_pred}

        strat1_results = pd.DataFrame(columns=['Model', 'Money', 'Bets Made'])
        strat2_results = pd.DataFrame(columns=['Model', 'Money', 'Bets Made'])
    
    
        for model_name, class_model in class_pred_dict.items():
                print(model_name)
                money, bets = simple_class_strategy_UK(class_model,df_unseen, graph=True)
                money1,bets1=top3_strategy_UK(class_model,df_unseen,graph=True)
                strat1_results.loc[len(strat1_results)] = [model_name, money, sum(bets)]
                strat2_results.loc[len(strat2_results)] = [model_name, money1, sum(bets1)]
    if st.button("BACKTEST: GB WIN with LGBM" ):
        df_results = pd.DataFrame(columns=['Model', 'Prediction', 'CV-F1', 'F1 Score', 'AUC', 'Recall', 'Precision'])
        # Get unique race IDs
        df_horse=pd.read_csv('./data/df_train_UK.csv')
        unique_race_ids = df_horse['race_id'].unique()
        # Split the race IDs into training, test, and unseen sets
        train_race_ids, test_race_ids = train_test_split(unique_race_ids, test_size=0.2, random_state=42)
        test_race_ids, unseen_race_ids = train_test_split(test_race_ids, test_size=0.5, random_state=42)
        # Split the data based on the race IDs
        df_train_80= df_horse[df_horse['race_id'].isin(train_race_ids)]
        df_test_20 = df_horse[df_horse['race_id'].isin(test_race_ids)]
        df_unseen = df_horse[df_horse['race_id'].isin(unseen_race_ids)]

        df_train_80.to_csv('./data/df_train_80_UK.csv', index=False)
        df_test_20.to_csv('./data/df_test_20_UK.csv', index=False)
        df_unseen.to_csv('./data/df_unseen_UK.csv', index=False)

        df_train_80.reset_index(inplace=True, drop=True)
        df_test_20.reset_index(inplace=True, drop=True)
        df_unseen.reset_index(inplace=True, drop=True)

        df_cleaned = df_train_80.dropna(subset=[  'draw','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank','time'])
        df_train_80 = df_cleaned.reset_index(drop=True)
        df_cleaned = df_test_20.dropna(subset=[  'draw','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank','time'])
        df_test_20 = df_cleaned.reset_index(drop=True)

        df_cleaned = df_unseen.dropna(subset=[  'draw','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank','time'])
        df_unseen= df_cleaned.reset_index(drop=True)
        X_train = df_train_80[[ 'draw',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank']]
        X_test = df_test_20[[ 'draw',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank']]
        X_unseen=df_unseen[[  'draw',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank']]
        y_train = df_train_80['time'].apply(time_to_seconds)
        y_test=df_test_20['time'].apply(time_to_seconds)
        y_unseen=df_unseen['time'].apply(time_to_seconds)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)
        print("y_unseen shape:", y_unseen.shape)
        #MAKE PREDICTIONS WITH LR
        
        # lgbm = LGBMRegressor(
        #     num_leaves=31,
        #     learning_rate=0.1,
        #     n_estimators=100,
        #     max_depth=10
        # )

        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        model = RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42, max_features=5,
                            min_samples_split=20, min_samples_leaf=200, n_jobs=-1)
# Create a
        lgbm_pred = run_model_reg(model, X_train, y_train, X_test, y_test, X_unseen,df_test_20)
        # Check the output of lgbm_pred
        print("LGBM Prediction Output:", lgbm_pred.head() if hasattr(lgbm_pred, 'head') else lgbm_pred)

        reg_pred_dict = {'LGBM': lgbm_pred}
        strat3_results = pd.DataFrame(columns=['Model', 'Money', 'Bets Made'])
        for model_name, reg_model in reg_pred_dict.items():
                # Print the model name
                print(model_name)
                # change False to True if you want to view the graph
                money, bets = simple_reg_strategy_UK(reg_model,df_unseen, graph = True)
                # Append the results to the dataframe
                strat3_results.loc[len(strat3_results)] = [model_name, money, sum(bets)]
    
    if st.button("BACKTEST: GB WIN with Deep Learning" ):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder

       
        df_results = pd.DataFrame(columns=['Model', 'Prediction', 'CV-F1', 'F1 Score', 'AUC', 'Recall', 'Precision'])
        # Get unique race IDs
        df_horse=pd.read_csv('./data/df_train_UK_results.csv')

        label_encoder = LabelEncoder()

        # Fit and transform the 'going' column
        df_horse['going'] = label_encoder.fit_transform(df_horse['going'])
        df_horse['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)


        unique_race_ids = df_horse['race_id'].unique()
        # Split the race IDs into training, test, and unseen sets
        train_race_ids, test_race_ids = train_test_split(unique_race_ids, test_size=0.2, random_state=42)
        test_race_ids, unseen_race_ids = train_test_split(test_race_ids, test_size=0.5, random_state=42)
        # Split the data based on the race IDs
        df_train_80= df_horse[df_horse['race_id'].isin(train_race_ids)]
        df_test_20 = df_horse[df_horse['race_id'].isin(test_race_ids)]
        df_unseen = df_horse[df_horse['race_id'].isin(unseen_race_ids)]

        df_train_80.to_csv('./data/df_train_80_UK.csv', index=False)
        df_test_20.to_csv('./data/df_test_20_UK.csv', index=False)
        df_unseen.to_csv('./data/df_unseen_UK.csv', index=False)

        df_train_80.reset_index(inplace=True, drop=True)
        df_test_20.reset_index(inplace=True, drop=True)
        df_unseen.reset_index(inplace=True, drop=True)

        df_cleaned = df_train_80.dropna(subset=[ 'draw','age',
                            'dist_m','weight_lbs','prize'
                             ,'rpr','going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank' ])
        df_train_80 = df_cleaned.reset_index(drop=True)
        df_cleaned = df_test_20.dropna(subset=['draw','age',
                            'dist_m','weight_lbs','prize'
                             ,'rpr','going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank' ])
        df_test_20 = df_cleaned.reset_index(drop=True)

        df_cleaned = df_unseen.dropna(subset=[ 'draw','age',
                            'dist_m','weight_lbs','prize'
                             ,'rpr','going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank' ])
        df_unseen= df_cleaned.reset_index(drop=True)
        X_train = df_train_80[[ 'draw','age',
                            'dist_m','weight_lbs','prize'
                             ,'rpr','going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank']]
        X_test = df_test_20[['draw','age',
                            'dist_m','weight_lbs','prize'
                             ,'rpr','going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank' ]]
        X_unseen=df_unseen[['draw','age',
                            'dist_m','weight_lbs','prize'
                             ,'rpr','going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank' ]]
        y_train = df_train_80['time'].apply(time_to_seconds)
        y_test=df_test_20['time'].apply(time_to_seconds)
        y_unseen=df_unseen['time'].apply(time_to_seconds)
        y_train = y_train.values.reshape(-1, 1) 

      


        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)  # Fit and transform the training data
        X_unseen_scaled=scaler_X.transform(X_unseen)
        y_train_scaled=scaler_y.fit_transform(y_train)
        import tensorflow as tf
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        # Define the model
        model = Sequential([
            Input(shape=(20,)),  # Input layer with 20 features
            Dense(60, activation='tanh'),
            Dense(60, activation='tanh'),  # First hidden layer with 10 neuron
            Dense(1)  # Output layer
        ])
        from keras.callbacks import Callback, EarlyStopping

        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")

        model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mae'])

        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)

        # Fit the model with both callbacks
        history = model.fit(X_train_scaled, y_train_scaled, epochs=5000, batch_size=64, validation_split=0.2,
                            callbacks=[streamlit_callback, early_stopping], verbose=0)
        epochs = range(1, len(history.history['loss']) + 1)


        # for layer in model.layers:
        #     weights = layer.get_weights()
        #     st.write(f"Layer: {layer.name}")
        #     for weight in weights:
        #         st.write(weight)

        # Streamlit app
        st.title('Loss Curve Plot')

        # Plotting function
        def plot_loss(history):
            plt.figure(figsize=(10, 6))
            plt.scatter(epochs,history.history['loss'], label='Training Loss')
            plt.scatter(epochs,history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        # Display the plot
        st.pyplot(plot_loss(history))

        DL_pred = model.predict(X_unseen_scaled) #what should k-fold be?
        # st.write("Model Predictions:")
        # st.write(DL_pred)


        import pandas as pd
        DL_pred_unscaled = scaler_y.inverse_transform(DL_pred)

        df_pred = pd.DataFrame()
        df_pred['RaceID'] = df_unseen['race_id']
        df_pred['HorseID'] = df_unseen['horse_name']
        df_pred['Course Name']=df_unseen['course']
        df_pred['finish_time'] = DL_pred_unscaled
        # df_pred_kempton = df_pred[df_pred['Course Name'] == 'Ayr']
        df_sorted = df_pred.sort_values(by=['RaceID', 'finish_time'])
        df_sorted['HorseWin'] = df_sorted.groupby('RaceID')['finish_time'].transform(lambda x: x == x.min()).astype(int)
        df_sorted['HorseRankTop3'] = df_sorted.groupby('RaceID')['finish_time'].rank(method='min', ascending=True) <= 3
        df_sorted['HorseRankTop3'] = df_sorted['HorseRankTop3'].astype(int)
        st.write(df_sorted)
        class_pred_dict = {'Deep Learning': df_sorted}

    
        strat1_results = pd.DataFrame(columns=['Model', 'Money', 'Bets Made'])
        strat2_results = pd.DataFrame(columns=['Model', 'Money', 'Bets Made'])
    
        for model_name, class_model in class_pred_dict.items():
                print(model_name)
                money, bets = simple_class_strategy_UK(class_model,df_unseen, graph=True)
                money1,bets1=top3_strategy_UK(class_model,df_unseen,graph=True)
                strat1_results.loc[len(strat1_results)] = [model_name, money, sum(bets)]
                strat2_results.loc[len(strat2_results)] = [model_name, money1, sum(bets1)]
    



    if st.button("BACKTEST: GB WIN with DF TEST  DF TEST XGBoost Regression Comprehensive"):

        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        import xgboost as xgb
        from keras.callbacks import Callback, EarlyStopping
        df_train = pd.read_csv('./data/df_train_UK_results.csv')
        df_train.reset_index(inplace=True, drop=True)
        # st.write(df_train[df_train['horse'] == 'Alphonse Le Grande (IRE)'])
        # st.write("df_train with no nans dropped is",df_train)
        label_encoder1 =LabelEncoder()
        label_encoder2 =LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        label_encoder6=LabelEncoder()
        label_encoder7=LabelEncoder()
        label_encoder8=LabelEncoder()
        label_encoder9=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['type'] = label_encoder3.fit_transform(df_train['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_train['horse'] = df_train['horse'].astype(str).str.replace(r"\s*\(.*?\)", "", regex=True)
        df_train['horse'] = label_encoder6.fit_transform(df_train['horse'])
        df_train['jockey'] = label_encoder7.fit_transform(df_train['jockey'])
        df_train['trainer'] = label_encoder8.fit_transform(df_train['trainer'])
        df_train['owner'] = label_encoder9.fit_transform(df_train['owner'])
        df_train['sp_dec'] = pd.to_numeric(df_train['sp_dec'], errors='coerce')

        # long_shot_winners = df_train[(df_train['sp_dec'] >= 12) & (df_train['HorseWin'] == 1)]
        # race_ids = long_shot_winners['race_id']  
        # df_train = df_train[df_train['race_id'].isin(race_ids)]
        #choose course to optimize for
    
        
        df_cleaned = df_train.dropna(subset=['age',
                            'dist_y','weight_lbs',
                             'going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank','horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage',
          'trainer_win_percentage','trainer_place_percentage','draw','course2','rpr','owner','last_position_1','last_position_2','last_position_3','last_position_4','last_position_5','last_position_6'])
        df_train = df_cleaned.reset_index(drop=True)
        df_train['tote_win']= df_train['tote_win'].replace('[£,]', '', regex=True).astype(float)
        # st.write("df_train with dropped nans is:",df_train)


        ############# DF_TEST RACECARD DATA #############
        def safe_transform(df, column, encoder):
            try:
                df[column] = encoder.transform(df[column])
            except ValueError as e:
                # Handle missing labels by setting a default value or handling in another way
                print("Encountered unknown label(s) in column '{}': {}".format(column, e))
                # Optionally, set a default value like -1 or NaN for unknown labels
                df[column] = df[column].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)


        df_test=pd.read_csv('./data/df_test_UK.csv')
        df_cleaned = df_test.dropna(subset=['age',
                            'distance_f','lbs','going'])
        df_test = df_cleaned.reset_index(drop=True)
        df_test = df_test.rename(columns={'horse_name': 'horse'})
        df_test['horse'] = df_test['horse'].str.replace(r"\s*\(.*?\)", "", regex=True)
        try:
            df_test['going'] = label_encoder1.transform(df_test['going'])
        except ValueError as e:
            print(f"Encountered unknown labels: {e}")
            df_test['going'] = df_test['going'].apply(lambda x: label_encoder1.transform([x])[0] if x in label_encoder1.classes_ else -1)  # assign -1 to unseen labels
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test['type'] = label_encoder3.transform(df_test['type'])
        df_test['sex'] = label_encoder4.transform(df_test['sex_code'])
        df_test['course2'] = label_encoder5.transform(df_test['course'])
        # df_test['sp_dec'] = df_test['odds'].apply(lambda x: extract_decimal_odds_for_bookmaker(x, selected_bookmaker))
        safe_transform(df_test, 'horse', label_encoder6)
        safe_transform(df_test, 'jockey', label_encoder7)
        safe_transform(df_test, 'trainer', label_encoder8)
        safe_transform(df_test, 'owner', label_encoder9)
        df_test[['recent_ave_rank', 'last_position_1','last_position_2','last_position_3','last_position_4',
            'last_position_5','last_position_6',
          'horse_win_percentage', 'horse_place_percentage',
          'jockey_ave_rank', 'last_jockey_finish_pos', 'second_last_jockey_finish_pos', 'third_last_jockey_finish_pos',
          'jockey_win_percentage', 'jockey_place_percentage',
          'trainer_ave_rank', 'last_trainer_finish_pos', 'second_last_trainer_finish_pos', 'third_last_trainer_finish_pos',
          'trainer_win_percentage', 'trainer_place_percentage',
          'owner_ave_rank', 'last_owner_finish_pos', 'second_last_owner_finish_pos', 'third_last_owner_finish_pos',
          'owner_win_percentage', 'owner_place_percentage']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK_with_win_percentage(row, df_test, df_train),
            axis=1
        )
        #choose course to optimize for
#         df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
#           'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
#           'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
#           'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
#             lambda row: compute_average_ranks_all_UK(row, df_test, df_train),
#             axis=1
# )       
        df_test.reset_index(inplace=True, drop=True)
        # st.write("df_train with dropped nans is:",df_train)
        # df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
        df_test['distance_f'] = df_test['distance_f'] * 201.168
        df_tote_win = df_train[['race_id','horse_id', 'tote_win','HorseWin','HorseRankTop3']]

        # Merge df_test with df_tote_win on the race_id column
        df_test = pd.merge(df_test, df_tote_win, on=['race_id','horse_id'], how='left')
        df_test.to_csv('./data/df_test_UK2.csv',index=False)


        df_cleaned = df_test.dropna(subset=['age',
                            'distance_f','lbs','going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank','horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage',
          'trainer_win_percentage','trainer_place_percentage','draw','course2','rpr','owner','last_position_1','last_position_2','last_position_3','last_position_4','last_position_5','last_position_6'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)

        st.write(df_test['odds'].iloc[0]) 
        import ast

       
        start_date = '2024-06-01'  # Example start date
        days_to_predict = 30
        all_predictions = []

        

        # for day in range(days_to_predict):
        # 1. Get the training data up to the current day
        current_train = df_train[df_train['date'] <= start_date]
        if len(df_train['race_id'].unique()) >= 3000:
            # Take a random sample of 3,300 unique race IDs
            sampled_race_ids = current_train['race_id'].drop_duplicates().sample(n=3000, random_state=1)  # random_state for reproducibility
            
            # Create a new DataFrame with the sampled race IDs
            current_train = df_train[df_train['race_id'].isin(sampled_race_ids)].reset_index(drop=True)
                
        X_train = current_train[['age', 'type', 'sex',
                        'weight_lbs','dist_m',
                            'going','recent_ave_rank', 'jockey_ave_rank','trainer_ave_rank','owner_ave_rank',
        'horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage','trainer_win_percentage','trainer_place_percentage',
        'draw','course2','rpr','owner','last_position_1','last_position_2','last_position_3','last_position_4','last_position_5','last_position_6','time']]
        # X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
        def time_to_seconds(time_str):
            try:
                if pd.isna(time_str):
                    # If the time is NaN, return None
                    return None

                # Convert the value to string in case it's a float
                time_str = str(time_str)

                # Split the time string by ':'
                parts = time_str.split(':')
                
                # Handle cases where the string might not be in the correct format
                if len(parts) == 2:  # Format like "4:12.11"
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                elif len(parts) == 1:  # Format like "12.11" (no minutes part)
                    minutes = 0
                    seconds = float(parts[0])
                else:
                    raise ValueError(f"Unexpected time format: {time_str}")
                
                # Convert minutes and seconds to total seconds
                return minutes * 60 + seconds
            except (ValueError, IndexError) as e:
                # Handle cases where the time string is invalid
                print(f"Error converting time: {time_str}. Error: {e}")
                return None  # Return None in case of an error
        X_train['time'].replace(['-', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        X_train.dropna(subset=['time'], inplace=True)
        y_train = X_train['time'].apply(time_to_seconds)
        y_train = y_train.values.reshape(-1, 1) 
    
        


        # start_date = '2023-10-16'
        # end_date = '2024-10-16'

        # # Filter X_train and X_test for the date range
        # X_train_filtered = X_train[(X_train['date'] >= start_date) & (X_train['date'] <= end_date)]
        
        # X_train = X_train_filtered.drop(columns=['time','date'])
        X_train = X_train.drop(columns=['time'])
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        # Now scale the cleaned data
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        import tensorflow as tf
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        # 5. Define and compile model
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import ElasticNet
        from sklearn.ensemble import RandomForestRegressor


        # model = ElasticNet()
        # param_grid = {
        #     'alpha': [0.01, 0.1, 1, 10, 100],
        #     'l1_ratio': [0.1, 0.5, 0.7, 0.9]  # Mix of Lasso and Ridge (0 = Ridge, 1 = Lasso)
        # }

        # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)



                # Step 3: Define parameter grid
        # param_grid = {
        #     'n_estimators': [50, 100, 200],  # Number of trees in the forest
        #     'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
        #     'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
        #     'min_samples_leaf': [1, 2, 4],    # Minimum samples required at each leaf node
        #     'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider for splitting
        # }

        # # Step 4: Initialize the RandomForestRegressor
        # rf = RandomForestRegressor(random_state=42)

        # # Step 5: Set up GridSearchCV
        # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

     
        # Convert the data into DMatrix format (XGBoost's optimized format)
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train_scaled)
       

        # Set the parameters for XGBoost
        params = {
            'objective': 'reg:squarederror',  # For regression tasks
            'eval_metric': 'rmse',  # Root mean square error
            'max_depth': 5,  # Depth of the tree
            'learning_rate': 0.01  # Learning rate
        }

        # Train the XGBoost model
        best_model = xgb.train(params, dtrain, num_boost_round=1000)

        # grid_search.fit(X_train_scaled, y_train_scaled)

        # print(f"Best Parameters: {grid_search.best_params_}")
        # print(f"Best Cross-Validated Score (Negative MSE): {grid_search.best_score_}")

        # best_model = grid_search.best_estimator_
        next_day = pd.to_datetime(start_date) + pd.Timedelta(days=1)
        # df_test_day = df_test[df_test['date'] == next_day.strftime('%Y-%m-%d')]
        start_date = pd.to_datetime(start_date)
        end_date = start_date + pd.Timedelta(days=26)  # 14 days after start_date gives a 15-day range

        # Filter df_test to include only rows within the 15-day date range
        df_test_day = df_test[(df_test['date'] >= start_date.strftime('%Y-%m-%d')) & 
                            (df_test['date'] <= end_date.strftime('%Y-%m-%d'))]
        # Debugging: Check if df_test_day has any data
        st.write(f"Test data for {next_day.strftime('%Y-%m-%d')}:")
        st.write(df_test_day)
        # if df_test_day.empty:
        #     st.write(f"No test data available for {next_day}. Skipping...")
        #     start_date = next_day.strftime('%Y-%m-%d')  # Move to the next date
        #     continue

        df_test_day = df_test_day.rename(columns={'lbs': 'weight_lbs', 'distance_f': 'dist_m'})
        # df_test_day=df_test_day[df_test_day['course']=='Goodwood']
                            
        # Now select the columns
        X_test = df_test_day[['age', 'type', 'sex', 'weight_lbs', 'dist_m', 'going', 
                            'recent_ave_rank', 'jockey_ave_rank','trainer_ave_rank','owner_ave_rank',
                            'horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage','trainer_win_percentage','trainer_place_percentage',
        'draw','course2','rpr','owner','last_position_1','last_position_2','last_position_3','last_position_4','last_position_5','last_position_6']]
        

        
        X_test_scaled = scaler_X.transform(X_test)
        dtest = xgb.DMatrix(X_test_scaled)
        # Predict for the available test data
        DL_pred = best_model.predict(dtest)
        DL_pred_unscaled = scaler_y.inverse_transform(DL_pred.reshape(-1, 1))

        # Save predictions for the current day
        df_pred = pd.DataFrame({
            'Date': df_test_day['date'],
            'Course Name': df_test_day['course'],
            'HorseID': df_test_day['horse'],
            'WinAmount':df_test_day['tote_win'],
            'ActualWin':df_test_day['HorseWin'],
            'RaceID': df_test_day['race_id'],
            'finish_time': DL_pred_unscaled.flatten()
        })

        all_predictions.append(df_pred)
        # Move to the next date
        start_date = next_day.strftime('%Y-%m-%d')

        # Concatenate all predictions if available
        if all_predictions:
            df_all_predictions = pd.concat(all_predictions, ignore_index=True)
            st.write(df_all_predictions)
        else:
            st.write("No predictions were made.")
        # Sort and rank predictions
        df_sorted = df_all_predictions.sort_values(by=['RaceID', 'finish_time'])
        df_sorted['HorseWin'] = df_sorted.groupby('RaceID')['finish_time'].transform(lambda x: x == x.min()).astype(int)
        df_sorted['HorseRankTop3'] = df_sorted.groupby('RaceID')['finish_time'].rank(method='min', ascending=True) <= 3
        df_sorted['HorseRankTop3'] = df_sorted['HorseRankTop3'].astype(int)
        df_sorted['MoneyMade'] = df_sorted.apply(lambda row: row['WinAmount'] if row['HorseWin'] == 1 and row['ActualWin'] == 1 else 0, axis=1)
        st.write("df sorted is:",df_sorted)
        # df_filtered = df_sorted[df_sorted['Course Name']]
        # Group by Date to calculate totals
        daily_totals = df_sorted.groupby('Date').agg(
            TotalMoneyMade=('MoneyMade', 'sum'),
            TotalAmountBet=('HorseWin', lambda x:  2*(x == 1).sum())
        ).reset_index()
        # Calculate Profit
        daily_totals['TotalProfit'] = daily_totals['TotalMoneyMade'] - daily_totals['TotalAmountBet']
        # Calculate Cumulative Profit
        daily_totals['CumulativeProfit'] = daily_totals['TotalProfit'].cumsum()
        # Display the daily totals
        st.write("Daily Totals:")
        st.write(daily_totals)

        # Plotting Cumulative Profit
        plt.figure(figsize=(12, 6))
        plt.plot(daily_totals['Date'], daily_totals['CumulativeProfit'], marker='o', label='Cumulative Profit', color='blue')

        plt.title('Cumulative Profit Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Profit')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()

        # Show plot in Streamlit
        st.pyplot(plt)
       
        total_money_made=df_sorted['MoneyMade'].sum()
        st.write("Total Money is:",total_money_made)
        total_bet_amount = 2 * len(df_sorted[df_sorted['HorseWin']==1])
        st.write("Total Amount bet  is:",total_bet_amount)
        st.write("Total Profit is:",total_money_made - total_bet_amount)
        # Split predictions by date into a dictionary
        dates = df_sorted['Date'].unique()
        predictions_by_date = {date: df_sorted[df_sorted['Date'] == date] for date in dates}
        with pd.ExcelWriter('race_predictions.xlsx', engine='openpyxl') as writer:
            for date, df in predictions_by_date.items():
                df.to_excel(writer, sheet_name=f"Predictions_{date}", index=False)
        print("Predictions saved to race_predictions.xlsx with separate tabs for each date.")
    
    







    if st.button("BACKTEST: GB WIN with DF TEST  DF TEST DF TEST DL Comprehensive"):

        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        from keras.callbacks import Callback, EarlyStopping
        df_train = pd.read_csv('./data/df_train_UK_results.csv')
        df_train.reset_index(inplace=True, drop=True)
        # st.write(df_train[df_train['horse'] == 'Alphonse Le Grande (IRE)'])
        # st.write("df_train with no nans dropped is",df_train)
        label_encoder1 =LabelEncoder()
        label_encoder2 =LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        label_encoder6=LabelEncoder()
        label_encoder7=LabelEncoder()
        label_encoder8=LabelEncoder()
        label_encoder9=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['type'] = label_encoder3.fit_transform(df_train['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_train['horse'] = df_train['horse'].astype(str).str.replace(r"\s*\(.*?\)", "", regex=True)
        df_train['horse'] = label_encoder6.fit_transform(df_train['horse'])
        df_train['jockey'] = label_encoder7.fit_transform(df_train['jockey'])
        df_train['trainer'] = label_encoder8.fit_transform(df_train['trainer'])
        df_train['owner'] = label_encoder9.fit_transform(df_train['owner'])
        # df_train['sp_dec'] = pd.to_numeric(df_train['sp_dec'], errors='coerce')
        df_train['ovr_btn'] = pd.to_numeric(df_train['ovr_btn'], errors='coerce')
        

        # long_shot_winners = df_train[(df_train['sp_dec'] >= 12) & (df_train['HorseWin'] == 1)]
        # race_ids = long_shot_winners['race_id']  
        # df_train = df_train[df_train['race_id'].isin(race_ids)]
        #choose course to optimize for
    
        
        df_cleaned = df_train.dropna(subset=['age',
                            'dist_y','weight_lbs',
                             'going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank','horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage',
          'trainer_win_percentage','trainer_place_percentage','draw','course2','rpr','owner','last_position_1','last_position_2','last_position_3','last_position_4','last_position_5','last_position_6'])
        df_train = df_cleaned.reset_index(drop=True)
        df_train['tote_win']= df_train['tote_win'].replace('[£,]', '', regex=True).astype(float)
        # st.write("df_train with dropped nans is:",df_train)
        df_train_sorted = df_train.sort_values(by=['horse', 'date'], ascending=[True, False])

        # 2. Get the most recent `ovr_btn` for each horse by grouping by horse name
        last_ovr_btn = df_train_sorted.groupby('horse').first()['ovr_btn']

        # 3. Merge this information into `X_test`
        # Now we add the last observed `ovr_btn` for each horse in `X_test`
      

        ############# DF_TEST RACECARD DATA #############
        def safe_transform(df, column, encoder):
            try:
                df[column] = encoder.transform(df[column])
            except ValueError as e:
                # Handle missing labels by setting a default value or handling in another way
                print("Encountered unknown label(s) in column '{}': {}".format(column, e))
                # Optionally, set a default value like -1 or NaN for unknown labels
                df[column] = df[column].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)


        df_test=pd.read_csv('./data/df_test_UK.csv')
        df_cleaned = df_test.dropna(subset=['age',
                            'distance_f','lbs','going'])
        df_test = df_cleaned.reset_index(drop=True)
        df_test = df_test.rename(columns={'horse_name': 'horse'})
        df_test['horse'] = df_test['horse'].str.replace(r"\s*\(.*?\)", "", regex=True)
        try:
            df_test['going'] = label_encoder1.transform(df_test['going'])
        except ValueError as e:
            print(f"Encountered unknown labels: {e}")
            df_test['going'] = df_test['going'].apply(lambda x: label_encoder1.transform([x])[0] if x in label_encoder1.classes_ else -1)  # assign -1 to unseen labels
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test['type'] = label_encoder3.transform(df_test['type'])
        df_test['sex'] = label_encoder4.transform(df_test['sex_code'])
        df_test['course2'] = label_encoder5.transform(df_test['course'])
        
        

            # 3. Merge this information into `X_test`
            # Now we add the last observed `ovr_btn` for each horse in `X_test`
      
       
      
            # Now, `X_test_with_last_ovr_btn` has the most recent `ovr_btn` from `X_train` for each horse in `X_test`
        # df_test['sp_dec'] = df_test['odds'].apply(lambda x: extract_decimal_odds_for_bookmaker(x, selected_bookmaker))
        safe_transform(df_test, 'horse', label_encoder6)
        safe_transform(df_test, 'jockey', label_encoder7)
        safe_transform(df_test, 'trainer', label_encoder8)
        safe_transform(df_test, 'owner', label_encoder9)
        df_test[['recent_ave_rank', 'last_position_1','last_position_2','last_position_3','last_position_4',
            'last_position_5','last_position_6',
          'horse_win_percentage', 'horse_place_percentage',
          'jockey_ave_rank', 'last_jockey_finish_pos', 'second_last_jockey_finish_pos', 'third_last_jockey_finish_pos',
          'jockey_win_percentage', 'jockey_place_percentage',
          'trainer_ave_rank', 'last_trainer_finish_pos', 'second_last_trainer_finish_pos', 'third_last_trainer_finish_pos',
          'trainer_win_percentage', 'trainer_place_percentage',
          'owner_ave_rank', 'last_owner_finish_pos', 'second_last_owner_finish_pos', 'third_last_owner_finish_pos',
          'owner_win_percentage', 'owner_place_percentage']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK_with_win_percentage(row, df_test, df_train),
            axis=1
        )
        #choose course to optimize for
#         df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
#           'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
#           'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
#           'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
#             lambda row: compute_average_ranks_all_UK(row, df_test, df_train),
#             axis=1
# )       
        df_test.reset_index(inplace=True, drop=True)
        # st.write("df_train with dropped nans is:",df_train)
        # df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
        df_test['distance_f'] = df_test['distance_f'] * 201.168
        df_tote_win = df_train[['race_id','horse_id', 'tote_win','HorseWin','HorseRankTop3']]

        # Merge df_test with df_tote_win on the race_id column
        df_test = pd.merge(df_test, df_tote_win, on=['race_id','horse_id'], how='left')
        df_test.to_csv('./data/df_test_UK2.csv',index=False)


        df_cleaned = df_test.dropna(subset=['age',
                            'distance_f','lbs','going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank','horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage',
          'trainer_win_percentage','trainer_place_percentage','draw','course2','rpr','owner','last_position_1','last_position_2','last_position_3','last_position_4','last_position_5','last_position_6'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)

        st.write(df_test['odds'].iloc[0]) 
        import ast

       
        start_date = '2024-07-01'  # Example start date
        days_to_predict = 30
        all_predictions = []

        

        # for day in range(days_to_predict):
        # 1. Get the training data up to the current day
        current_train = df_train[df_train['date'] <= start_date]
        if len(df_train['race_id'].unique()) >= 3300:
            # Take a random sample of 3,300 unique race IDs
            sampled_race_ids = current_train['race_id'].drop_duplicates().sample(n=3300, random_state=1)  # random_state for reproducibility
            
            # Create a new DataFrame with the sampled race IDs
            current_train = df_train[df_train['race_id'].isin(sampled_race_ids)].reset_index(drop=True)
                
        X_train = current_train[['age', 'type', 'sex',
                        'weight_lbs','dist_m',
                            'going','recent_ave_rank', 'jockey_ave_rank','trainer_ave_rank','owner_ave_rank',
        'horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage','trainer_win_percentage','trainer_place_percentage',
        'draw','course2','rpr','owner','last_position_1','last_position_2','last_position_3','last_position_4','last_position_5','last_position_6','ovr_btn','time']]
        # X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
        def time_to_seconds(time_str):
            try:
                if pd.isna(time_str):
                    # If the time is NaN, return None
                    return None

                # Convert the value to string in case it's a float
                time_str = str(time_str)

                # Split the time string by ':'
                parts = time_str.split(':')
                
                # Handle cases where the string might not be in the correct format
                if len(parts) == 2:  # Format like "4:12.11"
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                elif len(parts) == 1:  # Format like "12.11" (no minutes part)
                    minutes = 0
                    seconds = float(parts[0])
                else:
                    raise ValueError(f"Unexpected time format: {time_str}")
                
                # Convert minutes and seconds to total seconds
                return minutes * 60 + seconds
            except (ValueError, IndexError) as e:
                # Handle cases where the time string is invalid
                print(f"Error converting time: {time_str}. Error: {e}")
                return None  # Return None in case of an error
        X_train['time'].replace(['-', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        X_train.dropna(subset=['time'], inplace=True)
        y_train = X_train['time'].apply(time_to_seconds)
        y_train = y_train.values.reshape(-1, 1) 
    
        


        # start_date = '2023-10-16'
        # end_date = '2024-10-16'

        # # Filter X_train and X_test for the date range
        # X_train_filtered = X_train[(X_train['date'] >= start_date) & (X_train['date'] <= end_date)]
        
        # X_train = X_train_filtered.drop(columns=['time','date'])
        X_train = X_train.drop(columns=['time'])
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        # Now scale the cleaned data
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        import tensorflow as tf
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        # 5. Define and compile model
        model = Sequential([
            Dense(100, activation='relu', input_shape=(30,)),
            Dense(100, activation='relu'),
            Dense(1)
        ])
        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder
            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")
        model.compile(optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae'])

        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)
        # Fit the model with both callbacks
        history = model.fit(X_train_scaled, y_train_scaled, epochs=5000, batch_size=32, validation_split=0.2,
                            callbacks=[streamlit_callback, early_stopping], verbose=0)
        epochs = range(1, len(history.history['loss']) + 1)
        next_day = pd.to_datetime(start_date) + pd.Timedelta(days=1)
        # df_test_day = df_test[df_test['date'] == next_day.strftime('%Y-%m-%d')]
        start_date = pd.to_datetime(start_date)
        end_date = start_date + pd.Timedelta(days=26)  # 14 days after start_date gives a 15-day range

        # Filter df_test to include only rows within the 15-day date range
        df_test_day = df_test[(df_test['date'] >= start_date.strftime('%Y-%m-%d')) & 
                            (df_test['date'] <= end_date.strftime('%Y-%m-%d'))]
        # Debugging: Check if df_test_day has any data
        st.write(f"Test data for {next_day.strftime('%Y-%m-%d')}:")
        st.write(df_test_day)
        # if df_test_day.empty:
        #     st.write(f"No test data available for {next_day}. Skipping...")
        #     start_date = next_day.strftime('%Y-%m-%d')  # Move to the next date
        #     continue

        df_test_day = df_test_day.rename(columns={'lbs': 'weight_lbs', 'distance_f': 'dist_m'})
        # df_test_day = df_test_day[df_test_day['course'].isin(['Newmarket (July)', 'Newcastle (AW)'])]
        
        df_train_sorted = current_train.sort_values(by=['horse', 'date'], ascending=[True, False])

            # 2. Get the most recent `ovr_btn` for each horse by grouping by horse name
        last_ovr_btn = df_train_sorted.groupby('horse').first()['ovr_btn']
        df_test_day = df_test_day.merge(last_ovr_btn, on='horse', how='left', suffixes=('', '_last'))
        df_test_day.head()           
        # Now select the columns
        X_test = df_test_day[['age', 'type', 'sex', 'weight_lbs', 'dist_m', 'going', 
                            'recent_ave_rank', 'jockey_ave_rank','trainer_ave_rank','owner_ave_rank',
                            'horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage','trainer_win_percentage','trainer_place_percentage',
        'draw','course2','rpr','owner','last_position_1','last_position_2','last_position_3','last_position_4','last_position_5','last_position_6','ovr_btn']]
        
        X_test_scaled = scaler_X.transform(X_test)
        # Predict for the available test data
        DL_pred = model.predict(X_test_scaled)
        DL_pred_unscaled = scaler_y.inverse_transform(DL_pred)
        # Save predictions for the current day
        df_pred = pd.DataFrame({
            'Date': df_test_day['date'],
            'Course Name': df_test_day['course'],
            'HorseID': df_test_day['horse'],
            'WinAmount':df_test_day['tote_win'],
            'ActualWin':df_test_day['HorseWin'],
            'RaceID': df_test_day['race_id'],
            'finish_time': DL_pred_unscaled.flatten()
        })

        all_predictions.append(df_pred)
        # Move to the next date
        start_date = next_day.strftime('%Y-%m-%d')

        # Concatenate all predictions if available
        if all_predictions:
            df_all_predictions = pd.concat(all_predictions, ignore_index=True)
            st.write(df_all_predictions)
        else:
            st.write("No predictions were made.")
        # Sort and rank predictions
        df_sorted = df_all_predictions.sort_values(by=['RaceID', 'finish_time'])
        df_sorted['HorseWin'] = df_sorted.groupby('RaceID')['finish_time'].transform(lambda x: x == x.min()).astype(int)
        df_sorted['HorseRankTop3'] = df_sorted.groupby('RaceID')['finish_time'].rank(method='min', ascending=True) <= 3
        df_sorted['HorseRankTop3'] = df_sorted['HorseRankTop3'].astype(int)
        df_sorted['MoneyMade'] = df_sorted.apply(lambda row: row['WinAmount'] if row['HorseWin'] == 1 and row['ActualWin'] == 1 else 0, axis=1)
        st.write("df sorted is:",df_sorted)
        # df_filtered = df_sorted[df_sorted['Course Name']]
        # Group by Date to calculate totals
        daily_totals = df_sorted.groupby('Date').agg(
            TotalMoneyMade=('MoneyMade', 'sum'),
            TotalAmountBet=('HorseWin', lambda x:  2*(x == 1).sum())
        ).reset_index()
        # Calculate Profit
        daily_totals['TotalProfit'] = daily_totals['TotalMoneyMade'] - daily_totals['TotalAmountBet']
        # Calculate Cumulative Profit
        daily_totals['CumulativeProfit'] = daily_totals['TotalProfit'].cumsum()
        # Display the daily totals
        st.write("Daily Totals:")
        st.write(daily_totals)

        # Plotting Cumulative Profit
        plt.figure(figsize=(12, 6))
        plt.plot(daily_totals['Date'], daily_totals['CumulativeProfit'], marker='o', label='Cumulative Profit', color='blue')

        plt.title('Cumulative Profit Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Profit')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()

        # Show plot in Streamlit
        st.pyplot(plt)
       
        total_money_made=df_sorted['MoneyMade'].sum()
        st.write("Total Money is:",total_money_made)
        total_bet_amount = 2 * len(df_sorted[df_sorted['HorseWin']==1])
        st.write("Total Amount bet  is:",total_bet_amount)
        st.write("Total Profit is:",total_money_made - total_bet_amount)
        # Split predictions by date into a dictionary
        dates = df_sorted['Date'].unique()
        predictions_by_date = {date: df_sorted[df_sorted['Date'] == date] for date in dates}
        with pd.ExcelWriter('race_predictions.xlsx', engine='openpyxl') as writer:
            for date, df in predictions_by_date.items():
                df.to_excel(writer, sheet_name=f"Predictions_{date}", index=False)
        print("Predictions saved to race_predictions.xlsx with separate tabs for each date.")
    
    




    if st.button("BACKTEST: GB WIN with Racecards Comprehensive"):

        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        from keras.callbacks import Callback, EarlyStopping
        df_train = pd.read_csv('./data/df_train_UK_racecards.csv')
        df_train.reset_index(inplace=True, drop=True)
        # st.write(df_train[df_train['horse'] == 'Alphonse Le Grande (IRE)'])
        # st.write("df_train with no nans dropped is",df_train)
        label_encoder1 =LabelEncoder()
        label_encoder2 =LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        label_encoder6=LabelEncoder()
        label_encoder7=LabelEncoder()
        label_encoder8=LabelEncoder()
        label_encoder9=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['type'] = label_encoder3.fit_transform(df_train['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex_code'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_train['horse'] = df_train['horse'].astype(str).str.replace(r"\s*\(.*?\)", "", regex=True)
        df_train['horse'] = label_encoder6.fit_transform(df_train['horse'])
        df_train['jockey'] = label_encoder7.fit_transform(df_train['jockey'])
        df_train['trainer'] = label_encoder8.fit_transform(df_train['trainer'])
        df_train['owner'] = label_encoder9.fit_transform(df_train['owner'])
        # df_train['sp_dec'] = pd.to_numeric(df_train['sp_dec'], errors='coerce')
        df_train['ovr_btn'] = pd.to_numeric(df_train['ovr_btn'], errors='coerce')
        df_train = df_train[pd.to_numeric(df_train['age'], errors='coerce').notna()]

        

        # long_shot_winners = df_train[(df_train['sp_dec'] >= 12) & (df_train['HorseWin'] == 1)]
        # race_ids = long_shot_winners['race_id']  
        # df_train = df_train[df_train['race_id'].isin(race_ids)]
        #choose course to optimize for
    
        
        df_cleaned = df_train.dropna(subset=['age',
                            'distance_f','lbs','ovr_btn',
                             'going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank','horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage',
          'trainer_win_percentage','trainer_place_percentage','draw','course2','rpr','owner','last_position_1','last_position_2','last_position_3','last_position_4','last_position_5','last_position_6'])
        df_train = df_cleaned.reset_index(drop=True)
        df_train['tote_win']= df_train['tote_win'].replace('[£,]', '', regex=True).astype(float)
        # st.write("df_train with dropped nans is:",df_train)
        df_train_sorted = df_train.sort_values(by=['horse', 'date'], ascending=[True, False])

        # 2. Get the most recent `ovr_btn` for each horse by grouping by horse name
        last_ovr_btn = df_train_sorted.groupby('horse').first()['ovr_btn']

        # 3. Merge this information into `X_test`
        # Now we add the last observed `ovr_btn` for each horse in `X_test`
      

        ############# DF_TEST RACECARD DATA #############
        def safe_transform(df, column, encoder):
            try:
                df[column] = encoder.transform(df[column])
            except ValueError as e:
                # Handle missing labels by setting a default value or handling in another way
                print("Encountered unknown label(s) in column '{}': {}".format(column, e))
                # Optionally, set a default value like -1 or NaN for unknown labels
                df[column] = df[column].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)


        df_test=pd.read_csv('./data/df_test_UK.csv')
        df_cleaned = df_test.dropna(subset=['age',
                            'distance_f','lbs','going'])
        df_test = df_cleaned.reset_index(drop=True)
        df_test = df_test.rename(columns={'horse_name': 'horse'})
        df_test['horse'] = df_test['horse'].str.replace(r"\s*\(.*?\)", "", regex=True)
        try:
            df_test['going'] = label_encoder1.transform(df_test['going'])
        except ValueError as e:
            print(f"Encountered unknown labels: {e}")
            df_test['going'] = df_test['going'].apply(lambda x: label_encoder1.transform([x])[0] if x in label_encoder1.classes_ else -1)  # assign -1 to unseen labels
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test['type'] = label_encoder3.transform(df_test['type'])
        df_test['sex'] = label_encoder4.transform(df_test['sex_code'])
        df_test['course2'] = label_encoder5.transform(df_test['course'])
        
        

            # 3. Merge this information into `X_test`
            # Now we add the last observed `ovr_btn` for each horse in `X_test`
      
       
      
            # Now, `X_test_with_last_ovr_btn` has the most recent `ovr_btn` from `X_train` for each horse in `X_test`
        # df_test['sp_dec'] = df_test['odds'].apply(lambda x: extract_decimal_odds_for_bookmaker(x, selected_bookmaker))
        safe_transform(df_test, 'horse', label_encoder6)
        safe_transform(df_test, 'jockey', label_encoder7)
        safe_transform(df_test, 'trainer', label_encoder8)
        safe_transform(df_test, 'owner', label_encoder9)
        df_test[['recent_ave_rank', 'last_position_1','last_position_2','last_position_3','last_position_4',
            'last_position_5','last_position_6',
          'horse_win_percentage', 'horse_place_percentage',
          'jockey_ave_rank', 'last_jockey_finish_pos', 'second_last_jockey_finish_pos', 'third_last_jockey_finish_pos',
          'jockey_win_percentage', 'jockey_place_percentage',
          'trainer_ave_rank', 'last_trainer_finish_pos', 'second_last_trainer_finish_pos', 'third_last_trainer_finish_pos',
          'trainer_win_percentage', 'trainer_place_percentage',
          'owner_ave_rank', 'last_owner_finish_pos', 'second_last_owner_finish_pos', 'third_last_owner_finish_pos',
          'owner_win_percentage', 'owner_place_percentage']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK_with_win_percentage(row, df_test, df_train),
            axis=1
        )
        #choose course to optimize for
#         df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
#           'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
#           'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
#           'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
#             lambda row: compute_average_ranks_all_UK(row, df_test, df_train),
#             axis=1
# )       
        df_test.reset_index(inplace=True, drop=True)
        # st.write("df_train with dropped nans is:",df_train)
        # df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
      
        df_tote_win = df_train[['race_id','horse_id', 'tote_win','HorseWin','HorseRankTop3']]

        # Merge df_test with df_tote_win on the race_id column
        df_test = pd.merge(df_test, df_tote_win, on=['race_id','horse_id'], how='left')
        df_test.to_csv('./data/df_test_UK2.csv',index=False)


        df_cleaned = df_test.dropna(subset=['age',
                            'distance_f','lbs','going','recent_ave_rank','ovr_btn',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank','horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage',
          'trainer_win_percentage','trainer_place_percentage','draw','course2','rpr','owner','last_position_1','last_position_2','last_position_3','last_position_4','last_position_5','last_position_6'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)

        st.write(df_test['odds'].iloc[0]) 
        import ast

       
        start_date = '2024-07-01'  # Example start date
        days_to_predict = 30
        all_predictions = []

        

        # for day in range(days_to_predict):
        # 1. Get the training data up to the current day
        current_train = df_train[df_train['date'] <= start_date]
        if len(df_train['race_id'].unique()) >= 3300:
            # Take a random sample of 3,300 unique race IDs
            sampled_race_ids = current_train['race_id'].drop_duplicates().sample(n=3300, random_state=1)  # random_state for reproducibility
            
            # Create a new DataFrame with the sampled race IDs
            current_train = df_train[df_train['race_id'].isin(sampled_race_ids)].reset_index(drop=True)
                
        X_train = current_train[['age', 'type', 'sex',
                        'lbs','distance_f',
                            'going','recent_ave_rank', 'jockey_ave_rank','trainer_ave_rank','owner_ave_rank',
        'horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage','trainer_win_percentage','trainer_place_percentage',
        'draw','course2','rpr','owner','last_position_1','last_position_2','last_position_3','last_position_4','last_position_5','last_position_6','ovr_btn','time']]
        # X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
        def time_to_seconds(time_str):
            try:
                if pd.isna(time_str):
                    # If the time is NaN, return None
                    return None

                # Convert the value to string in case it's a float
                time_str = str(time_str)

                # Split the time string by ':'
                parts = time_str.split(':')
                
                # Handle cases where the string might not be in the correct format
                if len(parts) == 2:  # Format like "4:12.11"
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                elif len(parts) == 1:  # Format like "12.11" (no minutes part)
                    minutes = 0
                    seconds = float(parts[0])
                else:
                    raise ValueError(f"Unexpected time format: {time_str}")
                
                # Convert minutes and seconds to total seconds
                return minutes * 60 + seconds
            except (ValueError, IndexError) as e:
                # Handle cases where the time string is invalid
                print(f"Error converting time: {time_str}. Error: {e}")
                return None  # Return None in case of an error
        X_train['time'].replace(['-', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        X_train.dropna(subset=['time'], inplace=True)
        y_train = X_train['time'].apply(time_to_seconds)
        y_train = y_train.values.reshape(-1, 1) 
    
        


        # start_date = '2023-10-16'
        # end_date = '2024-10-16'

        # # Filter X_train and X_test for the date range
        # X_train_filtered = X_train[(X_train['date'] >= start_date) & (X_train['date'] <= end_date)]
        
        # X_train = X_train_filtered.drop(columns=['time','date'])
        X_train = X_train.drop(columns=['time'])
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        # Now scale the cleaned data
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        import tensorflow as tf
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        # 5. Define and compile model
        model = Sequential([
            Dense(10, activation='relu', input_shape=(30,)),
            Dense(10, activation='relu'),
            Dense(1)
        ])
        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder
            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")
        model.compile(optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae'])

        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)
        # Fit the model with both callbacks
        history = model.fit(X_train_scaled, y_train_scaled, epochs=5000, batch_size=16, validation_split=0.2,
                            callbacks=[streamlit_callback, early_stopping], verbose=0)
        epochs = range(1, len(history.history['loss']) + 1)
        next_day = pd.to_datetime(start_date) + pd.Timedelta(days=1)
        # df_test_day = df_test[df_test['date'] == next_day.strftime('%Y-%m-%d')]
        start_date = pd.to_datetime(start_date)
        end_date = start_date + pd.Timedelta(days=26)  # 14 days after start_date gives a 15-day range

        # Filter df_test to include only rows within the 15-day date range
        df_test_day = df_test[(df_test['date'] >= start_date.strftime('%Y-%m-%d')) & 
                            (df_test['date'] <= end_date.strftime('%Y-%m-%d'))]
        # Debugging: Check if df_test_day has any data
        st.write(f"Test data for {next_day.strftime('%Y-%m-%d')}:")
        st.write(df_test_day)
        # if df_test_day.empty:
        #     st.write(f"No test data available for {next_day}. Skipping...")
        #     start_date = next_day.strftime('%Y-%m-%d')  # Move to the next date
        #     continue

        # df_test_day = df_test_day[df_test_day['course'].isin(['Newmarket (July)', 'Newcastle (AW)'])]
        
        df_train_sorted = current_train.sort_values(by=['horse', 'date'], ascending=[True, False])

            # 2. Get the most recent `ovr_btn` for each horse by grouping by horse name
        last_ovr_btn = df_train_sorted.groupby('horse').first()['ovr_btn']
        df_test_day = df_test_day.merge(last_ovr_btn, on='horse', how='left', suffixes=('', '_last'))
        df_test_day.head()           
        # Now select the columns
        X_test = df_test_day[['age', 'type', 'sex', 'lbs', 'distance_f', 'going', 
                            'recent_ave_rank', 'jockey_ave_rank','trainer_ave_rank','owner_ave_rank',
                            'horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage','trainer_win_percentage','trainer_place_percentage',
        'draw','course2','rpr','owner','last_position_1','last_position_2','last_position_3','last_position_4','last_position_5','last_position_6','ovr_btn']]
        
        X_test_scaled = scaler_X.transform(X_test)
        # Predict for the available test data
        DL_pred = model.predict(X_test_scaled)
        DL_pred_unscaled = scaler_y.inverse_transform(DL_pred)
        # Save predictions for the current day
        df_pred = pd.DataFrame({
            'Date': df_test_day['date'],
            'Course Name': df_test_day['course'],
            'HorseID': df_test_day['horse'],
            'WinAmount':df_test_day['tote_win'],
            'ActualWin':df_test_day['HorseWin'],
            'RaceID': df_test_day['race_id'],
            'finish_time': DL_pred_unscaled.flatten()
        })

        all_predictions.append(df_pred)
        # Move to the next date
        start_date = next_day.strftime('%Y-%m-%d')

        # Concatenate all predictions if available
        if all_predictions:
            df_all_predictions = pd.concat(all_predictions, ignore_index=True)
            st.write(df_all_predictions)
        else:
            st.write("No predictions were made.")
        # Sort and rank predictions
        df_sorted = df_all_predictions.sort_values(by=['RaceID', 'finish_time'])
        df_sorted['HorseWin'] = df_sorted.groupby('RaceID')['finish_time'].transform(lambda x: x == x.min()).astype(int)
        df_sorted['HorseRankTop3'] = df_sorted.groupby('RaceID')['finish_time'].rank(method='min', ascending=True) <= 3
        df_sorted['HorseRankTop3'] = df_sorted['HorseRankTop3'].astype(int)
        df_sorted['MoneyMade'] = df_sorted.apply(lambda row: row['WinAmount'] if row['HorseWin'] == 1 and row['ActualWin'] == 1 else 0, axis=1)
        st.write("df sorted is:",df_sorted)
        # df_filtered = df_sorted[df_sorted['Course Name']]
        # Group by Date to calculate totals
        daily_totals = df_sorted.groupby('Date').agg(
            TotalMoneyMade=('MoneyMade', 'sum'),
            TotalAmountBet=('HorseWin', lambda x:  2*(x == 1).sum())
        ).reset_index()
        # Calculate Profit
        daily_totals['TotalProfit'] = daily_totals['TotalMoneyMade'] - daily_totals['TotalAmountBet']
        # Calculate Cumulative Profit
        daily_totals['CumulativeProfit'] = daily_totals['TotalProfit'].cumsum()
        # Display the daily totals
        st.write("Daily Totals:")
        st.write(daily_totals)

        # Plotting Cumulative Profit
        plt.figure(figsize=(12, 6))
        plt.plot(daily_totals['Date'], daily_totals['CumulativeProfit'], marker='o', label='Cumulative Profit', color='blue')

        plt.title('Cumulative Profit Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Profit')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()

        # Show plot in Streamlit
        st.pyplot(plt)
       
        total_money_made=df_sorted['MoneyMade'].sum()
        st.write("Total Money is:",total_money_made)
        total_bet_amount = 2 * len(df_sorted[df_sorted['HorseWin']==1])
        st.write("Total Amount bet  is:",total_bet_amount)
        st.write("Total Profit is:",total_money_made - total_bet_amount)
        # Split predictions by date into a dictionary
        dates = df_sorted['Date'].unique()
        predictions_by_date = {date: df_sorted[df_sorted['Date'] == date] for date in dates}
        with pd.ExcelWriter('race_predictions.xlsx', engine='openpyxl') as writer:
            for date, df in predictions_by_date.items():
                df.to_excel(writer, sheet_name=f"Predictions_{date}", index=False)
        print("Predictions saved to race_predictions.xlsx with separate tabs for each date.")
    
    if st.button("NEW NEW BACKTEST: GB WIN with DF TEST NEW NEW"):

        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        from keras.callbacks import Callback, EarlyStopping
        df_train = pd.read_csv('./data/df_train_UK_results.csv')
        df_train.reset_index(inplace=True, drop=True)
        # st.write(df_train[df_train['horse'] == 'Alphonse Le Grande (IRE)'])
        # st.write("df_train with no nans dropped is",df_train)
        label_encoder1 =LabelEncoder()
        label_encoder2 =LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        label_encoder6=LabelEncoder()
        label_encoder7=LabelEncoder()
        label_encoder8=LabelEncoder()
        label_encoder9=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['type'] = label_encoder3.fit_transform(df_train['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_train['horse'] = df_train['horse'].astype(str).str.replace(r"\s*\(.*?\)", "", regex=True)
        df_train['horse'] = label_encoder6.fit_transform(df_train['horse'])
        df_train['jockey'] = label_encoder7.fit_transform(df_train['jockey'])
        df_train['trainer'] = label_encoder8.fit_transform(df_train['trainer'])
        df_train['owner'] = label_encoder9.fit_transform(df_train['owner'])
        df_train['sp_dec'] = pd.to_numeric(df_train['sp_dec'], errors='coerce')

        # long_shot_winners = df_train[(df_train['sp_dec'] >= 12) & (df_train['HorseWin'] == 1)]
        # race_ids = long_shot_winners['race_id']  
        # df_train = df_train[df_train['race_id'].isin(race_ids)]
        #choose course to optimize for
    
        
        df_cleaned = df_train.dropna(subset=['draw','age', 'type', 'sex',
                            'weight_lbs','dist_m',
                             'going','recent_ave_rank',  
            'jockey_ave_rank',  
            'trainer_ave_rank', 
            'owner_ave_rank'])
        df_train = df_cleaned.reset_index(drop=True)
        df_train['tote_win']= df_train['tote_win'].replace('[£,]', '', regex=True).astype(float)
        # st.write("df_train with dropped nans is:",df_train)


        ############# DF_TEST RACECARD DATA #############
        def safe_transform(df, column, encoder):
            try:
                df[column] = encoder.transform(df[column])
            except ValueError as e:
                # Handle missing labels by setting a default value or handling in another way
                print("Encountered unknown label(s) in column '{}': {}".format(column, e))
                # Optionally, set a default value like -1 or NaN for unknown labels
                df[column] = df[column].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)


        df_test=pd.read_csv('./data/df_test_UK.csv')
        df_cleaned = df_test.dropna(subset=['age',
                            'distance_f','lbs','going'])
        df_test = df_cleaned.reset_index(drop=True)
        df_test = df_test.rename(columns={'horse_name': 'horse'})
        df_test['horse'] = df_test['horse'].str.replace(r"\s*\(.*?\)", "", regex=True)
        try:
            df_test['going'] = label_encoder1.transform(df_test['going'])
        except ValueError as e:
            print(f"Encountered unknown labels: {e}")
            df_test['going'] = df_test['going'].apply(lambda x: label_encoder1.transform([x])[0] if x in label_encoder1.classes_ else -1)  # assign -1 to unseen labels
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test['type'] = label_encoder3.transform(df_test['type'])
        df_test['sex'] = label_encoder4.transform(df_test['sex_code'])
        df_test['course2'] = label_encoder5.transform(df_test['course'])
        # df_test['sp_dec'] = df_test['odds'].apply(lambda x: extract_decimal_odds_for_bookmaker(x, selected_bookmaker))
        safe_transform(df_test, 'horse', label_encoder6)
        safe_transform(df_test, 'jockey', label_encoder7)
        safe_transform(df_test, 'trainer', label_encoder8)
        safe_transform(df_test, 'owner', label_encoder9)
        df_test[['recent_ave_rank', 'last_position_1','last_position_2','last_position_3','last_position_4',
            'last_position_5','last_position_6',
          'horse_win_percentage', 'horse_place_percentage',
          'jockey_ave_rank', 'last_jockey_finish_pos', 'second_last_jockey_finish_pos', 'third_last_jockey_finish_pos',
          'jockey_win_percentage', 'jockey_place_percentage',
          'trainer_ave_rank', 'last_trainer_finish_pos', 'second_last_trainer_finish_pos', 'third_last_trainer_finish_pos',
          'trainer_win_percentage', 'trainer_place_percentage',
          'owner_ave_rank', 'last_owner_finish_pos', 'second_last_owner_finish_pos', 'third_last_owner_finish_pos',
          'owner_win_percentage', 'owner_place_percentage']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK_with_win_percentage(row, df_test, df_train),
            axis=1
        )
        #choose course to optimize for
#         df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
#           'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
#           'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
#           'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
#             lambda row: compute_average_ranks_all_UK(row, df_test, df_train),
#             axis=1
# )       
        df_test.reset_index(inplace=True, drop=True)
        # st.write("df_train with dropped nans is:",df_train)
        # df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
        df_test['distance_f'] = df_test['distance_f'] * 201.168
        df_tote_win = df_train[['race_id','horse_id', 'tote_win','HorseWin','HorseRankTop3']]

        # Merge df_test with df_tote_win on the race_id column
        df_test = pd.merge(df_test, df_tote_win, on=['race_id','horse_id'], how='left')
        df_test.to_csv('./data/df_test_UK2.csv',index=False)


        df_cleaned = df_test.dropna(subset=['draw','age', 'type', 'sex',
                            'lbs','distance_f',
                             'going','recent_ave_rank',  
            'jockey_ave_rank',  
            'trainer_ave_rank', 
            'owner_ave_rank'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)

        st.write(df_test['odds'].iloc[0]) 
        import ast

       
        start_date = '2024-06-01'  # Example start date
        days_to_predict = 30
        all_predictions = []

        

        for day in range(days_to_predict):
            # 1. Get the training data up to the current day
            current_train = df_train[df_train['date'] <= start_date]
            if len(df_train['race_id'].unique()) >= 3300:
                # Take a random sample of 3,300 unique race IDs
                sampled_race_ids = current_train['race_id'].drop_duplicates().sample(n=3300, random_state=1)  # random_state for reproducibility
                
                # Create a new DataFrame with the sampled race IDs
                current_train = df_train[df_train['race_id'].isin(sampled_race_ids)].reset_index(drop=True)
                    
            X_train = current_train[['draw','age', 'type', 'sex',
                                'weight_lbs','dist_m',
                                'going','recent_ave_rank',  
                'jockey_ave_rank',  
                'trainer_ave_rank', 
                'owner_ave_rank','time']]
            # X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
        
            def time_to_seconds(time_str):
                try:
                    if pd.isna(time_str):
                        # If the time is NaN, return None
                        return None

                    # Convert the value to string in case it's a float
                    time_str = str(time_str)

                    # Split the time string by ':'
                    parts = time_str.split(':')
                    
                    # Handle cases where the string might not be in the correct format
                    if len(parts) == 2:  # Format like "4:12.11"
                        minutes = int(parts[0])
                        seconds = float(parts[1])
                    elif len(parts) == 1:  # Format like "12.11" (no minutes part)
                        minutes = 0
                        seconds = float(parts[0])
                    else:
                        raise ValueError(f"Unexpected time format: {time_str}")
                    
                    # Convert minutes and seconds to total seconds
                    return minutes * 60 + seconds
                except (ValueError, IndexError) as e:
                    # Handle cases where the time string is invalid
                    print(f"Error converting time: {time_str}. Error: {e}")
                    return None  # Return None in case of an error
            X_train['time'].replace(['-', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
            X_train.dropna(subset=['time'], inplace=True)
            y_train = X_train['time'].apply(time_to_seconds)
            y_train = y_train.values.reshape(-1, 1) 
        

            # start_date = '2023-10-16'
            # end_date = '2024-10-16'

            # # Filter X_train and X_test for the date range
            # X_train_filtered = X_train[(X_train['date'] >= start_date) & (X_train['date'] <= end_date)]
            
            # X_train = X_train_filtered.drop(columns=['time','date'])
            X_train = X_train.drop(columns=['time'])
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            # Now scale the cleaned data
            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train)
            import tensorflow as tf
            import numpy as np
            import random
            np.random.seed(42)
            random.seed(42)
            tf.random.set_seed(42)
            # 5. Define and compile model
            model = Sequential([
                Dense(100, activation='relu', input_shape=(11,)),
                Dense(100, activation='relu'),
                Dense(100, activation='relu'),
                Dense(1)
            ])
            # Define a custom callback to print loss at the end of each epoch
            class StreamlitCallback(Callback):
                def __init__(self, placeholder):
                    super(StreamlitCallback, self).__init__()
                    self.placeholder = placeholder
                def on_epoch_end(self, epoch, logs=None):
                    loss = logs.get('loss')
                    self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")
            model.compile(optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae'])

            # Create a placeholder in Streamlit
            placeholder = st.empty()
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # Create the Streamlit callback
            streamlit_callback = StreamlitCallback(placeholder)
            # Fit the model with both callbacks
            history = model.fit(X_train_scaled, y_train_scaled, epochs=5000, batch_size=16, validation_split=0.2,
                                callbacks=[streamlit_callback, early_stopping], verbose=0)
            epochs = range(1, len(history.history['loss']) + 1)
            next_day = pd.to_datetime(start_date) + pd.Timedelta(days=1)
            df_test_day = df_test[df_test['date'] == next_day.strftime('%Y-%m-%d')]
            # start_date = pd.to_datetime(start_date)
            # end_date = start_date + pd.Timedelta(days=26)  # 14 days after start_date gives a 15-day range

            # Filter df_test to include only rows within the 15-day date range
            # df_test_day = df_test[(df_test['date'] >= start_date.strftime('%Y-%m-%d')) & 
            #                     (df_test['date'] <= end_date.strftime('%Y-%m-%d'))]
            # Debugging: Check if df_test_day has any data
            st.write(f"Test data for {next_day.strftime('%Y-%m-%d')}:")
            st.write(df_test_day)
            # if df_test_day.empty:
            #     st.write(f"No test data available for {next_day}. Skipping...")
            #     start_date = next_day.strftime('%Y-%m-%d')  # Move to the next date
            #     continue

            df_test_day = df_test_day.rename(columns={'lbs': 'weight_lbs', 'distance_f': 'dist_m'})
                                
            # Now select the columns
            X_test = df_test_day[['draw','age', 'type', 'sex',
                                'weight_lbs','dist_m',
                                'going','recent_ave_rank',  
                'jockey_ave_rank',  
                'trainer_ave_rank', 
                'owner_ave_rank']]
            

            X_test_scaled = scaler_X.transform(X_test)
            # Predict for the available test data
            DL_pred = model.predict(X_test_scaled)
            DL_pred_unscaled = scaler_y.inverse_transform(DL_pred)
            # Save predictions for the current day
            df_pred = pd.DataFrame({
                'Date': df_test_day['date'],
                'Course Name': df_test_day['course'],
                'HorseID': df_test_day['horse'],
                'WinAmount':df_test_day['tote_win'],
                'ActualWin':df_test_day['HorseWin'],
                'RaceID': df_test_day['race_id'],
                'finish_time': DL_pred_unscaled.flatten()
            })

            all_predictions.append(df_pred)
            # Move to the next date
            start_date = next_day.strftime('%Y-%m-%d')

        # Concatenate all predictions if available
        if all_predictions:
            df_all_predictions = pd.concat(all_predictions, ignore_index=True)
            st.write(df_all_predictions)
        else:
            st.write("No predictions were made.")
        # Sort and rank predictions
        df_sorted = df_all_predictions.sort_values(by=['RaceID', 'finish_time'])
        df_sorted['HorseWin'] = df_sorted.groupby('RaceID')['finish_time'].transform(lambda x: x == x.min()).astype(int)
        df_sorted['HorseRankTop3'] = df_sorted.groupby('RaceID')['finish_time'].rank(method='min', ascending=True) <= 3
        df_sorted['HorseRankTop3'] = df_sorted['HorseRankTop3'].astype(int)
        df_sorted['MoneyMade'] = df_sorted.apply(lambda row: 5*row['WinAmount'] if row['HorseWin'] == 1 and row['ActualWin'] == 1 else 0, axis=1)
        st.write("df sorted is:",df_sorted)
        # df_filtered = df_sorted[df_sorted['Course Name']]
        # Group by Date to calculate totals
        daily_totals = df_sorted.groupby('Date').agg(
            TotalMoneyMade=('MoneyMade', 'sum'),
            TotalAmountBet=('HorseWin', lambda x:  10*(x == 1).sum())
        ).reset_index()
        # Calculate Profit
        daily_totals['TotalProfit'] = daily_totals['TotalMoneyMade'] - daily_totals['TotalAmountBet']
        # Calculate Cumulative Profit
        daily_totals['CumulativeProfit'] = daily_totals['TotalProfit'].cumsum()
        # Display the daily totals
        st.write("Daily Totals:")
        st.write(daily_totals)

        # Plotting Cumulative Profit
        plt.figure(figsize=(12, 6))
        plt.plot(daily_totals['Date'], daily_totals['CumulativeProfit'], marker='o', label='Cumulative Profit', color='blue')

        plt.title('Cumulative Profit Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Profit')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()

        # Show plot in Streamlit
        st.pyplot(plt)
       
        total_money_made=df_sorted['MoneyMade'].sum()
        st.write("Total Money is:",total_money_made)
        total_bet_amount = 10 * len(df_sorted[df_sorted['HorseWin']==1])
        st.write("Total Amount bet  is:",total_bet_amount)
        st.write("Total Profit is:",total_money_made - total_bet_amount)
        # Split predictions by date into a dictionary
        dates = df_sorted['Date'].unique()
        predictions_by_date = {date: df_sorted[df_sorted['Date'] == date] for date in dates}
        with pd.ExcelWriter('race_predictions.xlsx', engine='openpyxl') as writer:
            for date, df in predictions_by_date.items():
                df.to_excel(writer, sheet_name=f"Predictions_{date}", index=False)
        print("Predictions saved to race_predictions.xlsx with separate tabs for each date.")
    
    
    if st.button("BACKTEST: GB WIN with DL Comprehensive"):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        from keras.callbacks import Callback, EarlyStopping

        df_train = pd.read_csv('./data/df_train_UK_racecards.csv')
        df_train.reset_index(inplace=True, drop=True)
         # X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
        # df_train = df_train[df_train['type'].isin(['Flat', 'NH Flat'])]
        # df_train[df_train['date']=="2024-09-04"].to_csv('./data/df_test_backtest.csv',index=False)
        df_train = df_train[pd.to_numeric(df_train['age'], errors='coerce').notna()]

        # st.write(df_train[df_train['horse'] == 'Alphonse Le Grande (IRE)'])

        # st.write("df_train with no nans dropped is",df_train)
        label_encoder1 = LabelEncoder()
        label_encoder2 = LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        label_encoder6=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['type'] = label_encoder3.fit_transform(df_train['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_train['surface'] = label_encoder6.fit_transform(df_train['surface'])
        # df_train['sp_dec'] = pd.to_numeric(df_train['sp_dec'], errors='coerce')
        df_train['ovr_btn'] = pd.to_numeric(df_train['ovr_btn'], errors='coerce')

        #choose course to optimize for
        df_train = df_train.drop(columns=['recent_ave_rank','jockey_ave_rank','trainer_ave_rank','owner_ave_rank'])

        df_cleaned = df_train.dropna(subset=['draw','age',
                            'lbs',
                             'going','rpr'])
        
        df_train = df_cleaned.reset_index(drop=True)
        df_train['tote_win']= df_train['tote_win'].replace('[£,]', '', regex=True).astype(float)
        # st.write("df_train with dropped nans is:",df_train)
        # df_train[df_train['type']=='Chase']
        
        
       
        start_date = '2024-10-01'  # Example start date
        days_to_predict =3
        all_predictions = []

        

        for day in range(days_to_predict):
            # 1. Get the training data up to the current day
            current_train = df_train[df_train['date'] <= start_date]
            current_train=simp_cal_ave_rank_all_UK(current_train)
            st.write("current train is",current_train)
            # if len(current_train['race_id'].unique()) >= 19700:
            #     # Take a random sample of 3,300 unique race IDs
            #     sampled_race_ids = current_train['race_id'].drop_duplicates().sample(n=19700, random_state=1)  # random_state for reproducibility
                
            #     # Create a new DataFrame with the sampled race IDs
            #     current_train = current_train[current_train['race_id'].isin(sampled_race_ids)].reset_index(drop=True)
            #     st.write("current train 2 is",current_train)

                # long_shot_winners = current_train[(current_train['sp_dec'] >= 12) & (current_train['HorseWin'] == 1)]


                # race_ids = long_shot_winners['race_id']  

                # current_train = current_train[current_train['race_id'].isin(race_ids)]

        
                    



            X_train = current_train[['draw','age', 'type', 'sex',
                            'lbs','distance_f',
                             'going','recent_ave_rank',  
            'jockey_ave_rank',  
            'trainer_ave_rank', 
            'owner_ave_rank','rpr','time']]
        

           

            from sklearn.impute import KNNImputer

          
            def time_to_seconds(time_str):
                try:
                    if pd.isna(time_str):
                        # If the time is NaN, return None
                        return None

                    # Convert the value to string in case it's a float
                    time_str = str(time_str)

                    # Split the time string by ':'
                    parts = time_str.split(':')
                    
                    # Handle cases where the string might not be in the correct format
                    if len(parts) == 2:  # Format like "4:12.11"
                        minutes = int(parts[0])
                        seconds = float(parts[1])
                    elif len(parts) == 1:  # Format like "12.11" (no minutes part)
                        minutes = 0
                        seconds = float(parts[0])
                    else:
                        raise ValueError(f"Unexpected time format: {time_str}")
                    
                    # Convert minutes and seconds to total seconds
                    return minutes * 60 + seconds
                except (ValueError, IndexError) as e:
                    # Handle cases where the time string is invalid
                    print(f"Error converting time: {time_str}. Error: {e}")
                    return None  # Return None in case of an error
            X_train['time'].replace(['-', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
            X_train.dropna(subset=['time'], inplace=True)
          
            y_train = X_train['time'].apply(time_to_seconds)
            y_train = y_train.values.reshape(-1, 1) 
        
            


            # start_date = '2023-10-16'
            # end_date = '2024-10-16'

            # # Filter X_train and X_test for the date range
            # X_train_filtered = X_train[(X_train['date'] >= start_date) & (X_train['date'] <= end_date)]
            
            # X_train = X_train_filtered.drop(columns=['time','date'])
            X_train = X_train.drop(columns=['time'])

            


            from sklearn.impute import KNNImputer


            from sklearn.linear_model import LinearRegression
            import pandas as pd

            # X_train_features = X_train.drop(columns=['rpr'])  # The 11 features
            # y_train_rpr = X_train['rpr']  # Target column

            # # 2. Train a regression model
            # regressor = LinearRegression()
            # regressor.fit(X_train_features, y_train_rpr)

            
          







            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            # Now scale the cleaned data
            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train)
    
            import tensorflow as tf
            from keras.optimizers import SGD
            from keras import regularizers

            import numpy as np
            import random
            np.random.seed(42)
            random.seed(42)
            tf.random.set_seed(42)
            # 5. Define and compile model
            model = Sequential([
                Dense(10, activation='relu', input_shape=(12,)),
                Dense(10, activation='relu'),
                Dense(1)
            ])
            # Define a custom callback to print loss at the end of each epoch
            class StreamlitCallback(Callback):
                def __init__(self, placeholder):
                    super(StreamlitCallback, self).__init__()
                    self.placeholder = placeholder
                def on_epoch_end(self, epoch, logs=None):
                    loss = logs.get('loss')
                    self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")
            model.compile(optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae'])

            # Create a placeholder in Streamlit
            placeholder = st.empty()
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # Create the Streamlit callback
            streamlit_callback = StreamlitCallback(placeholder)
            # Fit the model with both callbacks
            
            next_day = pd.to_datetime(start_date) + pd.Timedelta(days=1)
            df_test_day = df_train[df_train['date'] == next_day.strftime('%Y-%m-%d')]
            st.write("LENGTH OF DF TEST IS",len(df_test_day))
            df_test_day.to_csv("./data/df_test_day.csv", index=False)  # Set index=False to avoid saving the index as a separate column

            # Debugging: Check if df_test_day has any data

            

            if df_test_day.empty:
                st.write(f"No test data available for {next_day}. Skipping...")
                start_date = next_day.strftime('%Y-%m-%d')  # Move to the next date
                continue

            import pandas as pd

            # Assuming df_test_day has these columns and 'runners_horse_name' is the identifier for each horse
            # df_train should have 'runners_horse_name' and 'ovr_btn' columns

            # 1. Sort the training data (`X_train`) by horse name and race date (or race ID)
            # Sort so the most recent race for each horse comes first
            df_train_sorted = current_train.sort_values(by=['horse', 'date'], ascending=[True, False])

            # 2. Get the most recent `ovr_btn` for each horse by grouping by horse name
            last_ovr_btn = df_train_sorted.groupby('horse').first()['ovr_btn']
            df_test_day = df_test_day.merge(last_ovr_btn, on='horse', how='left', suffixes=('', '_last'))
            df_test_day = df_test_day.drop(columns=['ovr_btn'])
            df_test_day = df_test_day.rename(columns={'ovr_btn_last': 'ovr_btn'})
            st.write("DF_TEST is",df_test_day)
            # Now, `X_test_with_last_ovr_btn` has the most recent `ovr_btn` from `X_train` for each horse in `X_test`
            
            # Sort the data by date to ensure the most recent entry comes last
            current_train_sorted = current_train.sort_values('date')

            # Drop duplicates by keeping the most recent entry (last one)
            current_train_most_recent = current_train_sorted.drop_duplicates('horse', keep='last')

            # Merge with df_test_day to get the most recent data for each horse
            df_test_day = df_test_day.merge(current_train_most_recent[['horse', 'recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank', 'owner_ave_rank']],
                                            on='horse', how='left')

                        
            st.write("df test 2 is",df_test_day)
            X_test = df_test_day[['draw','age', 'type', 'sex',
                            'lbs','distance_f',
                             'going','recent_ave_rank',  
            'jockey_ave_rank',  
            'trainer_ave_rank', 
            'owner_ave_rank','rpr']]
            st.write("X test is",X_test)


            # X_test = X_test[X_test['type'].isin(['Flat', 'NH Flat'])]



            # # 3. Predict `rpr` for X_test_scaled
            # X_test_features = X_test.copy()  # Use the same 11 features in X_test_scaled
            # predicted_rpr = regressor.predict(X_test_features)
            #   # 4. Add the `rpr` column to X_test_scaled
            # X_test['rpr'] = predicted_rpr
     


            
    








            # current_train['rpr'] = pd.to_numeric(current_train['rpr'], errors='coerce')
            # df_test_day['rpr'] = pd.to_numeric(df_test_day['rpr'], errors='coerce')

            # def get_last_n_rpr_mean(horse, n=6):
            #     # Filter out the relevant data for the given horse from X_train
            #     horse_races = current_train[current_train['horse'] == horse]
                
            #     # Sort by the race date (or any other column that indicates race order)
            #     horse_races = horse_races.sort_values(by='date', ascending=True)
                
            #     # Get the last 'n' races and calculate the mean of 'rpr'
            #     last_n_races = horse_races.tail(n)
            #     return last_n_races['rpr'].mean()

            # # 2. Apply the function to X_test to assign the rolling mean of 'rpr'
            # X_test['rpr'] = df_test_day['horse'].apply(lambda horse: get_last_n_rpr_mean(horse))
            from sklearn.impute import KNNImputer
            from sklearn.neighbors import KNeighborsRegressor

             

           
        

            # for race_id, race_data in X_test.groupby('race_id'):
            # Select features for the current race
            # X_test_new = race_data[['draw', 'age', 'type', 'sex', 'weight_lbs', 'dist_m', 'rpr', 
            #                     'prize', 'going', 'recent_ave_rank', 'jockey_ave_rank', 
            #                     'trainer_ave_rank', 'owner_ave_rank']]
            
            # Set the batch size to the number of horses in the current race
            # batch_size = len(X_test_new)

            history = model.fit(X_train_scaled, y_train_scaled, epochs=5000, batch_size=16, validation_split=0.2,
                            callbacks=[streamlit_callback, early_stopping], verbose=0)
            epochs = range(1, len(history.history['loss']) + 1)

            
        




    

            if X_test.empty:
                print("X_test is empty, skipping scaling.")
            else:
                X_test_scaled = scaler_X.transform(X_test)

           

           


    

          





                # Predict for the available test data
                DL_pred = model.predict(X_test_scaled)
                DL_pred_unscaled = scaler_y.inverse_transform(DL_pred)
                
                # Save predictions for the current day
                df_pred = pd.DataFrame({
                    'Date': df_test_day['date'],
                    'Course Name': df_test_day['course'],
                    'HorseID': df_test_day['horse'],
                    'WinAmount':df_test_day['tote_win'],
                    'ActualWin':df_test_day['HorseWin'],
                    'RaceID': df_test_day['race_id'],
                    'finish_time': DL_pred_unscaled.flatten()
                })

           

                all_predictions.append(df_pred_filtered)
                
                # Move to the next date
                start_date = next_day.strftime('%Y-%m-%d')

        # Concatenate all predictions if available
        if all_predictions:
            df_all_predictions = pd.concat(all_predictions, ignore_index=True)
            st.write(df_all_predictions)
        else:
            st.write("No predictions were made.")
        # Sort and rank predictions
        df_sorted = df_all_predictions.sort_values(by=['RaceID', 'finish_time'])
        df_sorted['HorseWin'] = df_sorted.groupby('RaceID')['finish_time'].transform(lambda x: x == x.min()).astype(int)
        df_sorted['HorseRankTop3'] = df_sorted.groupby('RaceID')['finish_time'].rank(method='min', ascending=True) <= 3
        df_sorted['HorseRankTop3'] = df_sorted['HorseRankTop3'].astype(int)
        df_sorted['MoneyMade'] = df_sorted.apply(lambda row: row['WinAmount'] if row['HorseWin'] == 1 and row['ActualWin'] == 1 else 0, axis=1)
        # df_filtered = df_sorted[df_sorted['Course Name']]
        st.write("DF_SORTED IS:",df_sorted)

        # Group by Date to calculate totals
        daily_totals = df_sorted.groupby('Date').agg(
            TotalMoneyMade=('MoneyMade', 'sum'),
            TotalBetsAmount=('HorseWin', lambda x:  2*(x == 1).sum())
        ).reset_index()

        # Calculate Profit
        daily_totals['TotalProfit'] = daily_totals['TotalMoneyMade'] - daily_totals['TotalBetsAmount']

        # Calculate Cumulative Profit
        daily_totals['CumulativeProfit'] = daily_totals['TotalProfit'].cumsum()

        # Display the daily totals
        st.write("Daily Totals:")
        st.write(daily_totals)

        # Plotting Cumulative Profit
        plt.figure(figsize=(12, 6))
        plt.plot(daily_totals['Date'], daily_totals['CumulativeProfit'], marker='o', label='Cumulative Profit', color='blue')

        plt.title('Cumulative Profit Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Profit')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()

        # Show plot in Streamlit
        st.pyplot(plt)


        total_money_made=df_sorted['MoneyMade'].sum()
        st.write("Total Money is:",total_money_made)
        total_bet_amount =  2*len(df_sorted[df_sorted['HorseWin']==1])
        st.write("Total Amount bet  is:",total_bet_amount)
        st.write("Total Profit is:",total_money_made - total_bet_amount)
        # Split predictions by date into a dictionary
        dates = df_sorted['Date'].unique()
        predictions_by_date = {date: df_sorted[df_sorted['Date'] == date] for date in dates}
        with pd.ExcelWriter('race_predictions.xlsx', engine='openpyxl') as writer:
            for date, df in predictions_by_date.items():
                df.to_excel(writer, sheet_name=f"Predictions_{date}", index=False)
        print("Predictions saved to race_predictions.xlsx with separate tabs for each date.")
     


    if st.button("BACKTEST: GB WIN with Logistic Regression NEW "):

        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        from keras.callbacks import Callback, EarlyStopping

        df_train = pd.read_csv('./data/df_train_UK_racecards.csv')
        df_train.reset_index(inplace=True, drop=True)
         # X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
        # df_train = df_train[df_train['type'].isin(['Flat', 'NH Flat'])]
        # df_train[df_train['date']=="2024-09-04"].to_csv('./data/df_test_backtest.csv',index=False)
        df_train = df_train[pd.to_numeric(df_train['age'], errors='coerce').notna()]

        # st.write(df_train[df_train['horse'] == 'Alphonse Le Grande (IRE)'])

        # st.write("df_train with no nans dropped is",df_train)
        label_encoder1 = LabelEncoder()
        label_encoder2 = LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        label_encoder6=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['type'] = label_encoder3.fit_transform(df_train['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_train['surface'] = label_encoder6.fit_transform(df_train['surface'])
        # df_train['sp_dec'] = pd.to_numeric(df_train['sp_dec'], errors='coerce')
        df_train['ovr_btn'] = pd.to_numeric(df_train['ovr_btn'], errors='coerce')

        #choose course to optimize for
        df_train = df_train.drop(columns=['recent_ave_rank','jockey_ave_rank','trainer_ave_rank','owner_ave_rank'])

        df_cleaned = df_train.dropna(subset=['draw','age',
                            'lbs',
                             'going','rpr','distance_f','ovr_btn','surface'])
        
        df_train = df_cleaned.reset_index(drop=True)
        df_train['tote_win']= df_train['tote_win'].replace('[£,]', '', regex=True).astype(float)
        # st.write("df_train with dropped nans is:",df_train)
        # df_train[df_train['type']=='Chase']
        
        
       
        start_date = '2024-10-01'  # Example start date
        days_to_predict =10
        all_predictions = []

        

        for day in range(days_to_predict):
            # 1. Get the training data up to the current day
            current_train = df_train[df_train['date'] <= start_date]
            current_train=simp_cal_ave_rank_all_UK(current_train)
            st.write("current train is",current_train)
            # if len(current_train['race_id'].unique()) >= 19700:
            #     # Take a random sample of 3,300 unique race IDs
            #     sampled_race_ids = current_train['race_id'].drop_duplicates().sample(n=19700, random_state=1)  # random_state for reproducibility
                
            #     # Create a new DataFrame with the sampled race IDs
            #     current_train = current_train[current_train['race_id'].isin(sampled_race_ids)].reset_index(drop=True)
            #     st.write("current train 2 is",current_train)

                # long_shot_winners = current_train[(current_train['sp_dec'] >= 12) & (current_train['HorseWin'] == 1)]


                # race_ids = long_shot_winners['race_id']  

                # current_train = current_train[current_train['race_id'].isin(race_ids)]

        
                    



            X_train = current_train[['draw','age', 'type', 'sex',
                            'lbs','distance_f',
                             'going','recent_ave_rank',  
            'jockey_ave_rank',  
            'trainer_ave_rank', 
            'owner_ave_rank','rpr','ovr_btn','surface','time']]
        

            ridge = Ridge()

            # Define the hyperparameter grid to search over
            param_grid = {
                'alpha': [0.01, 0.1, 1, 10, 100]  # Different regularization strengths
            }
                    

            
            grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

        
            def time_to_seconds(time_str):
                try:
                    if pd.isna(time_str):
                        # If the time is NaN, return None
                        return None

                    # Convert the value to string in case it's a float
                    time_str = str(time_str)

                    # Split the time string by ':'
                    parts = time_str.split(':')
                    
                    # Handle cases where the string might not be in the correct format
                    if len(parts) == 2:  # Format like "4:12.11"
                        minutes = int(parts[0])
                        seconds = float(parts[1])
                    elif len(parts) == 1:  # Format like "12.11" (no minutes part)
                        minutes = 0
                        seconds = float(parts[0])
                    else:
                        raise ValueError(f"Unexpected time format: {time_str}")
                    
                    # Convert minutes and seconds to total seconds
                    return minutes * 60 + seconds
                except (ValueError, IndexError) as e:
                    # Handle cases where the time string is invalid
                    print(f"Error converting time: {time_str}. Error: {e}")
                    return None  # Return None in case of an error
            X_train['time'].replace(['-', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
            X_train.dropna(subset=['time'], inplace=True)
          
            y_train = X_train['time'].apply(time_to_seconds)
            y_train = y_train.values.reshape(-1, 1) 
        
            


            # start_date = '2023-10-16'
            # end_date = '2024-10-16'

            # # Filter X_train and X_test for the date range
            # X_train_filtered = X_train[(X_train['date'] >= start_date) & (X_train['date'] <= end_date)]
            
            # X_train = X_train_filtered.drop(columns=['time','date'])
            X_train = X_train.drop(columns=['time'])

            


            from sklearn.impute import KNNImputer


            from sklearn.linear_model import LinearRegression
            import pandas as pd

            # X_train_features = X_train.drop(columns=['rpr'])  # The 11 features
            # y_train_rpr = X_train['rpr']  # Target column

            # # 2. Train a regression model
            # regressor = LinearRegression()
            # regressor.fit(X_train_features, y_train_rpr)

            
          







            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            # Now scale the cleaned data
            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train)
    
            import tensorflow as tf
            from keras.optimizers import SGD
            from keras import regularizers

            import numpy as np
            import random
            np.random.seed(42)
            random.seed(42)
            tf.random.set_seed(42)
            # 5. Define and compile model

                # Fit the model
            grid_search.fit(X_train_scaled, y_train_scaled)

            # Display the best parameters found by GridSearchCV
            print("Best Parameters:", grid_search.best_params_)

            best_ridge_model = grid_search.best_estimator_

            from sklearn.impute import KNNImputer

          
            


            

            next_day = pd.to_datetime(start_date) + pd.Timedelta(days=1)
            df_test_day = df_train[df_train['date'] == next_day.strftime('%Y-%m-%d')]
            st.write("LENGTH OF DF TEST IS",len(df_test_day))
            df_test_day.to_csv("./data/df_test_day.csv", index=False)  # Set index=False to avoid saving the index as a separate column

            # Debugging: Check if df_test_day has any data

            

            if df_test_day.empty:
                st.write(f"No test data available for {next_day}. Skipping...")
                start_date = next_day.strftime('%Y-%m-%d')  # Move to the next date
                continue

            import pandas as pd

            # Assuming df_test_day has these columns and 'runners_horse_name' is the identifier for each horse
            # df_train should have 'runners_horse_name' and 'ovr_btn' columns

            # 1. Sort the training data (`X_train`) by horse name and race date (or race ID)
            # Sort so the most recent race for each horse comes first
            df_train_sorted = current_train.sort_values(by=['horse', 'date'], ascending=[True, False])

            # 2. Get the most recent `ovr_btn` for each horse by grouping by horse name
            last_ovr_btn = df_train_sorted.groupby('horse').first()['ovr_btn']
            df_test_day = df_test_day.merge(last_ovr_btn, on='horse', how='left', suffixes=('', '_last'))
            df_test_day = df_test_day.drop(columns=['ovr_btn'])
            df_test_day = df_test_day.rename(columns={'ovr_btn_last': 'ovr_btn'})
            st.write("DF_TEST is",df_test_day)
            # Now, `X_test_with_last_ovr_btn` has the most recent `ovr_btn` from `X_train` for each horse in `X_test`
            
            # Sort the data by date to ensure the most recent entry comes last
            current_train_sorted = current_train.sort_values('date')

            # Drop duplicates by keeping the most recent entry (last one)
            current_train_most_recent = current_train_sorted.drop_duplicates('horse', keep='last')

            # Merge with df_test_day to get the most recent data for each horse
            df_test_day = df_test_day.merge(current_train_most_recent[['horse', 'recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank', 'owner_ave_rank']],
                                            on='horse', how='left')

            df_cleaned = df_test_day.dropna(subset=['draw','age', 'type', 'sex',
                            'lbs','distance_f',
                             'going','recent_ave_rank',  
            'jockey_ave_rank',  
            'trainer_ave_rank', 
            'owner_ave_rank','rpr','ovr_btn','surface'])
            
            df_test_day = df_cleaned.reset_index(drop=True)
                        
            st.write("df test 2 is",df_test_day)
            X_test = df_test_day[['draw','age', 'type', 'sex',
                            'lbs','distance_f',
                             'going','recent_ave_rank',  
            'jockey_ave_rank',  
            'trainer_ave_rank', 
            'owner_ave_rank','rpr','ovr_btn','surface']]


          
            st.write("X test is",X_test)


     

        




    

            if X_test.empty:
                print("X_test is empty, skipping scaling.")
            else:
                X_test_scaled = scaler_X.transform(X_test)

           

           


    

          





                # Predict for the available test data
                DL_pred = best_ridge_model.predict(X_test_scaled)
                DL_pred_unscaled = scaler_y.inverse_transform(DL_pred)
                
                # Save predictions for the current day
                df_pred = pd.DataFrame({
                    'Date': df_test_day['date'],
                    'Course Name': df_test_day['course'],
                    'HorseID': df_test_day['horse'],
                    'WinAmount':df_test_day['tote_win'],
                    'ActualWin':df_test_day['HorseWin'],
                    'RaceID': df_test_day['race_id'],
                    'finish_time': DL_pred_unscaled.flatten()
                })

                all_predictions.append(df_pred)
                
                
                # Move to the next date
                start_date = next_day.strftime('%Y-%m-%d')

        # Concatenate all predictions if available
        if all_predictions:
            df_all_predictions = pd.concat(all_predictions, ignore_index=True)
            st.write(df_all_predictions)
        else:
            st.write("No predictions were made.")
        # Sort and rank predictions
        df_sorted = df_all_predictions.sort_values(by=['RaceID', 'finish_time'])
        df_sorted['HorseWin'] = df_sorted.groupby('RaceID')['finish_time'].transform(lambda x: x == x.min()).astype(int)
        df_sorted['HorseRankTop3'] = df_sorted.groupby('RaceID')['finish_time'].rank(method='min', ascending=True) <= 3
        df_sorted['HorseRankTop3'] = df_sorted['HorseRankTop3'].astype(int)
        df_sorted['MoneyMade'] = df_sorted.apply(lambda row: row['WinAmount'] if row['HorseWin'] == 1 and row['ActualWin'] == 1 else 0, axis=1)
        # df_filtered = df_sorted[df_sorted['Course Name']]
        st.write("DF_SORTED IS:",df_sorted)

        # Group by Date to calculate totals
        daily_totals = df_sorted.groupby('Date').agg(
            TotalMoneyMade=('MoneyMade', 'sum'),
            TotalBetsAmount=('HorseWin', lambda x:  2*(x == 1).sum())
        ).reset_index()

        # Calculate Profit
        daily_totals['TotalProfit'] = daily_totals['TotalMoneyMade'] - daily_totals['TotalBetsAmount']

        # Calculate Cumulative Profit
        daily_totals['CumulativeProfit'] = daily_totals['TotalProfit'].cumsum()

        # Display the daily totals
        st.write("Daily Totals:")
        st.write(daily_totals)

        # Plotting Cumulative Profit
        plt.figure(figsize=(12, 6))
        plt.plot(daily_totals['Date'], daily_totals['CumulativeProfit'], marker='o', label='Cumulative Profit', color='blue')

        plt.title('Cumulative Profit Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Profit')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()

        # Show plot in Streamlit
        st.pyplot(plt)


        total_money_made=df_sorted['MoneyMade'].sum()
        st.write("Total Money is:",total_money_made)
        total_bet_amount =  2*len(df_sorted[df_sorted['HorseWin']==1])
        st.write("Total Amount bet  is:",total_bet_amount)
        st.write("Total Profit is:",total_money_made - total_bet_amount)
        # Split predictions by date into a dictionary
        dates = df_sorted['Date'].unique()
        predictions_by_date = {date: df_sorted[df_sorted['Date'] == date] for date in dates}
        with pd.ExcelWriter('race_predictions.xlsx', engine='openpyxl') as writer:
            for date, df in predictions_by_date.items():
                df.to_excel(writer, sheet_name=f"Predictions_{date}", index=False)
        print("Predictions saved to race_predictions.xlsx with separate tabs for each date.")



    if st.button("BACKTEST: GB WIN with Logistic Regression - HORSEWIN "):

        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        from keras.callbacks import Callback, EarlyStopping

        df_train = pd.read_csv('./data/df_train_UK_racecards.csv')
        df_train.reset_index(inplace=True, drop=True)
         # X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
        # df_train = df_train[df_train['type'].isin(['Flat', 'NH Flat'])]
        # df_train[df_train['date']=="2024-09-04"].to_csv('./data/df_test_backtest.csv',index=False)
        df_train = df_train[pd.to_numeric(df_train['age'], errors='coerce').notna()]

        # st.write(df_train[df_train['horse'] == 'Alphonse Le Grande (IRE)'])

        # st.write("df_train with no nans dropped is",df_train)
        label_encoder1 = LabelEncoder()
        label_encoder2 = LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        label_encoder6=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['type'] = label_encoder3.fit_transform(df_train['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_train['surface'] = label_encoder6.fit_transform(df_train['surface'])
        # df_train['sp_dec'] = pd.to_numeric(df_train['sp_dec'], errors='coerce')
        df_train['ovr_btn'] = pd.to_numeric(df_train['ovr_btn'], errors='coerce')

        #choose course to optimize for
        df_train = df_train.drop(columns=['recent_ave_rank','jockey_ave_rank','trainer_ave_rank','owner_ave_rank'])

        df_cleaned = df_train.dropna(subset=['draw',
                       'distance_f','lbs'])
        
        df_train = df_cleaned.reset_index(drop=True)
        df_train['tote_win']= df_train['tote_win'].replace('[£,]', '', regex=True).astype(float)
        # st.write("df_train with dropped nans is:",df_train)
        # df_train[df_train['type']=='Chase']
        
        
       
        start_date = '2024-10-01'  # Example start date
        days_to_predict =3
        all_predictions = []

        

        # 1. Get the training data up to the current day
        current_train = df_train[df_train['date'] <= start_date]
        current_train=simp_cal_ave_rank_all_UK(current_train)
        st.write("current train is",current_train)
        # if len(current_train['race_id'].unique()) >=1500:
        #     # Take a random sample of 3,300 unique race IDs
        #     sampled_race_ids = current_train['race_id'].drop_duplicates().sample(n=1500, random_state=1)  # random_state for reproducibility
            
        #     # Create a new DataFrame with the sampled race IDs
        #     current_train = current_train[current_train['race_id'].isin(sampled_race_ids)].reset_index(drop=True)
        #     st.write("current train 2 is",current_train)
        X_train = current_train[['draw','lbs',
                       'distance_f',
                            'recent_ave_rank',  
        'jockey_ave_rank',  
        'trainer_ave_rank']]

        y_train=current_train[['HorseWin']]
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)

        import seaborn as sns


        log_reg = LogisticRegression() # 'liblinear' solver is efficient for small datasets
        from sklearn.svm import SVC 
        # 2. Define the SMOTE resampling method
        # smote = SMOTE(random_state=42)

        # 3. Create a pipeline that first applies SMOTE, then fits the logistic regression model
        # pipeline = Pipeline([
        #     ('smote', smote),
        #     ('log_reg', log_reg)
        # ])

        # # 4. Define the parameter grid for logistic regression hyperparameter tuning
        # param_grid = {
        #     'log_reg__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],  # Logistic Regression's C parameter
        #     'log_reg__penalty': ['l2']  # Regularization type
        # }

        # # 5. Set up GridSearchCV to perform cross-validation and hyperparameter tuning
        # grid_search = GridSearchCV(estimator=pipeline, 
        #                         param_grid=param_grid, 
        #                         cv=5,  # 5-fold cross-validation
        #                         verbose=1,  # Show detailed output
        #                         n_jobs=-1)  # Use all available cores for parallel computation

        # Define the parameter grid
        # param_grid = {
        #     'degree': [2, 3, 4],        # Polynomial degrees to test
        #     'coef0': [0, 1, 10],        # Values of coef0 to try
        #     'C': [0.1, 1, 10],          # Regularization parameter C
        #     'gamma': ['scale', 'auto']  # Kernel coefficient values
        # }

        # svm_poly = SVC(kernel='poly')
        # param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
        #         'penalty': ['l1', 'l2']}

        # # Create a GridSearchCV objec   t
        # grid_search = GridSearchCV(log_reg, param_grid, cv=5,scoring='f1_weighted', n_jobs=-1, verbose=1)

        from imblearn.over_sampling import BorderlineSMOTE 
        from imblearn.under_sampling import TomekLinks 

        # tomek_links = TomekLinks()
        # borderline_smote = BorderlineSMOTE(random_state=42)

        # model = Pipeline([
        #     ('borderline_smote', borderline_smote),  # Apply BorderlineSMOTE for balancing during training
        #     ('classifier', RandomForestClassifier(random_state=42))  # Classifier
        # ])
        param_grid = {
            'n_estimators': [50, 100, 200],           # Number of trees
            'max_depth': [None, 10, 20, 30],         # Maximum depth of each tree
            'min_samples_split': [2, 5, 10],         # Minimum samples to split a node
            'min_samples_leaf': [1, 2, 4],           # Minimum samples per leaf node
            'bootstrap': [True, False]               # Use bootstrap samples
        }
        model = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=5,                # 5-fold cross-validation
            n_jobs=-1,           # Use all CPU cores
            verbose=2            # Print progress
        )

        model.fit(X_train_scaled, y_train)

        
        end_date = '2024-11-20'


        # Filter df_train to get rows where the date is within the specified range
        df_test_day = df_train[(df_train['date'] > start_date) & (df_train['date'] <= end_date)]

        st.write("LENGTH OF DF TEST IS",len(df_test_day))
        df_test_day.to_csv("./data/df_test_day.csv", index=False)  # Set index=False to avoid saving the index as a separate column
       

        import pandas as pd
        df_train_sorted = current_train.sort_values(by=['horse', 'date'], ascending=[True, False])

        # 2. Get the most recent `ovr_btn` for each horse by grouping by horse name
        last_ovr_btn = df_train_sorted.groupby('horse').first()['ovr_btn']
        df_test_day = df_test_day.merge(last_ovr_btn, on='horse', how='left', suffixes=('', '_last'))
        df_test_day = df_test_day.drop(columns=['ovr_btn'])
        df_test_day = df_test_day.rename(columns={'ovr_btn_last': 'ovr_btn'})
        st.write("DF_TEST is",df_test_day)
        
        current_train_sorted = current_train.sort_values('date')

        # Drop duplicates by keeping the most recent entry (last one)
        current_train_most_recent = current_train_sorted.drop_duplicates('horse', keep='last')

        # Merge with df_test_day to get the most recent data for each horse
        df_test_day = df_test_day.merge(current_train_most_recent[['horse', 'recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank', 'owner_ave_rank']],
                                        on='horse', how='left')

        df_cleaned = df_test_day.dropna(subset=['draw','lbs',
                       'distance_f',
                            'recent_ave_rank',  
        'jockey_ave_rank',  
        'trainer_ave_rank'])
        
        df_test_day = df_cleaned.reset_index(drop=True)
                    
        st.write("df test 2 is",df_test_day)
        X_test = df_test_day[['draw','lbs',
                       'distance_f',
                            'recent_ave_rank',  
        'jockey_ave_rank',  
        'trainer_ave_rank']]


        
        st.write("X test is",X_test)
        if X_test.empty:
            print("X_test is empty, skipping scaling.")
        else:
            X_test_scaled = scaler_X.transform(X_test)
            DL_pred = model.predict(X_test_scaled)
            df_pred = pd.DataFrame({
                'Date': df_test_day['date'],
                'Course Name': df_test_day['course'],
                'HorseID': df_test_day['horse'],
                'WinAmount':df_test_day['tote_win'],
                'ActualWin':df_test_day['HorseWin'],
                'RaceID': df_test_day['race_id'],
                'HorseWin': DL_pred.flatten()
            })

            
       
        # Sort and rank predictions
        df_sorted = df_pred.sort_values(by=['RaceID'])
        df_sorted['MoneyMade'] = df_sorted.apply(lambda row: row['WinAmount'] if row['HorseWin'] == 1 and row['ActualWin'] == 1 else 0, axis=1)
        # df_filtered = df_sorted[df_sorted['Course Name']]
        st.write("DF_SORTED IS:",df_sorted)

        # Group by Date to calculate totals
        daily_totals = df_sorted.groupby('Date').agg(
            TotalMoneyMade=('MoneyMade', 'sum'),
            TotalBetsAmount=('HorseWin', lambda x:  2*(x == 1).sum())
        ).reset_index()

        # Calculate Profit
        daily_totals['TotalProfit'] = daily_totals['TotalMoneyMade'] - daily_totals['TotalBetsAmount']

        # Calculate Cumulative Profit
        daily_totals['CumulativeProfit'] = daily_totals['TotalProfit'].cumsum()

        # Display the daily totals
        st.write("Daily Totals:")
        st.write(daily_totals)

        # Plotting Cumulative Profit
        plt.figure(figsize=(12, 6))
        plt.plot(daily_totals['Date'], daily_totals['CumulativeProfit'], marker='o', label='Cumulative Profit', color='blue')

        plt.title('Cumulative Profit Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Profit')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()

        # Show plot in Streamlit
        st.pyplot(plt)


        total_money_made=df_sorted['MoneyMade'].sum()
        st.write("Total Money is:",total_money_made)
        total_bet_amount =  2*len(df_sorted[df_sorted['HorseWin']==1])
        st.write("Total Amount bet  is:",total_bet_amount)
        st.write("Total Profit is:",total_money_made - total_bet_amount)
        # Split predictions by date into a dictionary
        dates = df_sorted['Date'].unique()
        predictions_by_date = {date: df_sorted[df_sorted['Date'] == date] for date in dates}
        with pd.ExcelWriter('race_predictions.xlsx', engine='openpyxl') as writer:
            for date, df in predictions_by_date.items():
                df.to_excel(writer, sheet_name=f"Predictions_{date}", index=False)
        print("Predictions saved to race_predictions.xlsx with separate tabs for each date.")












    if st.button('BACKTEST: GB WIN with Pairs Deep Learning'):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        from sklearn.preprocessing import LabelEncoder
        from keras.layers import Dense, Input
        from keras.optimizers import Adam
        from keras.optimizers import SGD
        from keras.layers import Dropout
        from keras.callbacks import Callback, EarlyStopping
        from keras.regularizers import l2
        from sklearn.preprocessing import StandardScaler
        import random


        df_results = pd.DataFrame(columns=['Model', 'Prediction', 'CV-F1', 'F1 Score', 'AUC', 'Recall', 'Precision'])
        # Get unique race IDs
        df_horse=pd.read_csv('./data/df_train_UK_results.csv')

        # if len(df_horse['race_id'].unique()) >= 3300:
        #         # Take a random sample of 3,300 unique race IDs
        #         sampled_race_ids = df_horse['race_id'].drop_duplicates().sample(n=3300, random_state=1)  # random_state for reproducibility
                
        #         # Create a new DataFrame with the sampled race IDs
        #         df_horse = df_horse[df_horse['race_id'].isin(sampled_race_ids)].reset_index(drop=True)

        label_encoder = LabelEncoder()

        # Fit and transform the 'going' column
        df_horse['going'] = label_encoder.fit_transform(df_horse['going'])
        df_horse['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_horse['tote_win']= df_horse['tote_win'].replace('[£,]', '', regex=True).astype(float)


        unique_race_ids = df_horse['race_id'].unique()
        # Split the race IDs into training, test, and unseen sets
        train_race_ids, test_race_ids = train_test_split(unique_race_ids, test_size=0.2, random_state=42)
        
    
        # Split the data based on the race IDs
        df_train_80= df_horse[df_horse['race_id'].isin(train_race_ids)]
        df_test_20 = df_horse[df_horse['race_id'].isin(test_race_ids)]


        df_train_80.to_csv('./data/df_train_80_UK.csv', index=False)
        df_test_20.to_csv('./data/df_test_20_UK.csv', index=False)
     

        df_train_80.reset_index(inplace=True, drop=True)
        df_test_20.reset_index(inplace=True, drop=True)
   

        df_cleaned = df_train_80.dropna(subset=[  'draw','age','dist_y','weight_lbs',
                             'time','going','recent_ave_rank',  
          'jockey_ave_rank', 
          'trainer_ave_rank',
          'owner_ave_rank','ovr_btn'])
        df_train_80 = df_cleaned.reset_index(drop=True)
        df_cleaned = df_test_20.dropna(subset=[  'draw','age','dist_y','weight_lbs',
                            'going','recent_ave_rank', 
          'jockey_ave_rank',  
          'trainer_ave_rank', 
          'owner_ave_rank'])
        df_test_20 = df_cleaned.reset_index(drop=True)

        df_train_sorted = df_train_80.sort_values(by=['horse', 'date'], ascending=[True, False])

            # 2. Get the most recent `ovr_btn` for each horse by grouping by horse name
        last_ovr_btn = df_train_sorted.groupby('horse').first()['ovr_btn']
        df_test_20 = df_test_20.merge(last_ovr_btn, on='horse', how='left', suffixes=('', '_last'))

    
        X_train = df_train_80[[ 'horse_id','course','race_name','race_id','position','draw','age',
                            'dist_y','weight_lbs',
                            'prize','going','recent_ave_rank',  
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos','ovr_btn']]
        
        X_test = df_test_20[[ 'horse_id','course','race_name','race_id','position','draw','age',
                            'dist_y','weight_lbs',
                            'prize','going','recent_ave_rank', 
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos','ovr_btn']]
        

        list_of_columns = ['race_id','dist_y', 'weight_lbs','recent_ave_rank','horse_id']
        duplicates =X_train.merge(X_test, how='inner', on=list_of_columns)
        st.write(duplicates)
       
        def create_pairwise_data_by_race(df):
            from itertools import combinations
            import random
            print("processing data into pairs")
            pairs = []
            features = ['draw', 'age', 'dist_y', 'weight_lbs',   'going', 'recent_ave_rank',
                        'jockey_ave_rank',
                        'trainer_ave_rank',
                        'owner_ave_rank','ovr_btn']
            df = df.sort_values(by=['race_id', 'position'])
            for race_id, race_df in df.groupby('race_id'):
                if len(race_df) < 2:
                    continue  
                # Shuffle horse indices to randomize the order of comparisons
                horse_indices = list(race_df.index)
                random.shuffle(horse_indices)
                for horse1_idx, horse2_idx in combinations(horse_indices, 2):
                    horse1_features = df.loc[horse1_idx, features].values
                    horse2_features = df.loc[horse2_idx, features].values
                    pair_features = list(horse1_features) + list(horse2_features)
                    # Compare positions
                    pos1 = df.loc[horse1_idx, 'position']
                    pos2 = df.loc[horse2_idx, 'position']
                    # Result is 1 if horse1 finished ahead of horse2, else 0
                    result = 1 if pos1 < pos2 else 0
                    # Get horse IDs
                    horse1_id = df.loc[horse1_idx, 'horse_id']  # Adjust this line based on your DataFrame
                    horse2_id = df.loc[horse2_idx, 'horse_id']  # Adjust this line based on your DataFrame
                    course=df.loc[horse1_idx, 'course']
                    race_name=df.loc[horse1_idx, 'race_name']
                    # Include both horse IDs and race_id with the pair
                    pairs.append((pair_features, result, race_id,course,race_name, horse1_id, horse2_id))
            return pairs
        
        def create_pairwise_data_by_race_test(df):
            from itertools import combinations
            import random
            print("processing data into pairs")
            pairs = []
            features = ['draw', 'age', 'dist_y', 'weight_lbs',   'going', 'recent_ave_rank',
                        'jockey_ave_rank', 'trainer_ave_rank', 'owner_ave_rank','ovr_btn']

            # Sort data by race_id (for consistency)
            df = df.sort_values(by=['race_id'])

            for race_id, race_df in df.groupby('race_id'):
                if len(race_df) < 2:
                    continue  # Skip races with fewer than 2 horses

                # Shuffle horse indices to randomize the order of comparisons
                horse_indices = list(race_df.index)
                random.shuffle(horse_indices)

                for horse1_idx, horse2_idx in combinations(horse_indices, 2):
                    horse1_features = df.loc[horse1_idx, features].values
                    horse2_features = df.loc[horse2_idx, features].values
                    pair_features = list(horse1_features) + list(horse2_features)

                    # Get horse IDs and race details
                    horse1_id = df.loc[horse1_idx, 'horse_id']
                    horse2_id = df.loc[horse2_idx, 'horse_id']
                    course = df.loc[horse1_idx, 'course']
                    race_name = df.loc[horse1_idx, 'race_name']

                    # Append the pair (no result)
                    pairs.append((pair_features, race_id, course, race_name, horse1_id, horse2_id))

            return pairs
        horse_counts = X_train.groupby('race_id').size()
        print(horse_counts.describe())  # This will show min, max, mean, etc., of horses per race
        st.write("size of X_train is",X_train.shape)       
        pairwise_data_train = create_pairwise_data_by_race(X_train)
        pairwise_data_test = create_pairwise_data_by_race_test(X_test)
        print("pairwise data is :", pairwise_data_train)
       

       

       
        X = np.array([pair[0] for pair in pairwise_data_train])  # Input features (horse1 vs horse2)
        y = np.array([pair[1] for pair in pairwise_data_train])  # Target labels (1 for win, 0 for loss)
                # Ensure X_train and X_test are 2D arrays (with shape (n_samples, n_features))
        y=y.reshape(-1,1)

        print(X.shape)  # Should print something like (n_pairs, n_features)
        print(y)
        print(X_train['position'].value_counts())


        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X)  # Fit and transform the training data
        y_train_scaled = scaler_y.fit_transform(y)
        # print(X_train_scaled[:50])  
        # print(y_train_scaled[:50])  
      
        # Define the model
        
        # Build a simple feedforward neural network
        model = Sequential()
        model.add(Dense(100, activation='relu', input_shape=(20,)))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='sigmoid')) 

      

        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")

            # sgd_optimizer = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',  #
              metrics=['accuracy']) 

        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)

        # Fit the model with both callbacks
        history = model.fit(X_train_scaled, y, epochs=5000, batch_size=50, validation_split=0.2,
                            callbacks=[streamlit_callback, early_stopping], verbose=0)
        epochs = range(1, len(history.history['loss']) + 1)

    
    
        
        X2 = np.array([pair[0] for pair in pairwise_data_test])  # Input features (horse1 vs horse2)
        print("Size of X2:", X2.shape)
        X_test_scaled = scaler_x.transform(X2)
       
       # Streamlit app
        st.title('Loss Curve Plot')

        # Plotting function
        def plot_loss(history):
            plt.figure(figsize=(10, 6))
            plt.scatter(epochs,history.history['loss'], label='Training Loss')
            plt.scatter(epochs,history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        # Display the plot
        st.pyplot(plot_loss(history))
        DL_pred = model.predict(X_test_scaled) 

        pairwise_predictions = (DL_pred > 0.5).astype(int)
        # Extract horse_name pairs and race_id from pairwise_data_test
       
          # from collections import defaultdict

        # # Initialize a dictionary to hold the scores for each horse in each race
        horse_scores = defaultdict(lambda: defaultdict(int))

        # Initialize a list to store the data
        horse_data_with_predictions = []

        # Loop through pairwise_predictions and pairwise_data_test to create a combined data structure
        for pred, (_, race_id, _, race_name, horse1_id, horse2_id) in zip(pairwise_predictions, pairwise_data_test):
             # Track predictions for each horse in the race
            if pred == 1:
                horse_scores[race_id][horse1_id] += 1  # Horse1 is predicted to win
            else:
                horse_scores[race_id][horse2_id] += 1  # Horse2 is predicted to win
    
            horse_data_with_predictions.append({
                'race_id': race_id,
                'horse1': horse1_id,
                'horse2': horse2_id,
                'race_name': race_name,
                'prediction': int(pred)  # 1 if horse1 wins, 0 if horse2 wins
            })
        
        # Convert the list to a pandas DataFrame
        horse_data_df = pd.DataFrame(horse_data_with_predictions)

        # Determine the winner for each race (the horse with the most predicted wins)
        race_winners = []
        for race_id, horses in horse_scores.items():
            # Get the horse with the maximum win count
            winner = max(horses, key=horses.get)
            race_winners.append({
                'race_id': race_id,
                'winner': winner
            })

        # Convert race winners to a DataFrame
        winners_df = pd.DataFrame(race_winners)

        # Merge the horse data with predictions DataFrame and the winners DataFrame
        final_df = pd.merge(horse_data_df, winners_df, on='race_id', how='left')

        # Display the final DataFrame in Streamlit
        st.write("Horse Race Predictions and Winners:")
        st.dataframe(final_df)

        






      
        # # Loop through the pairwise data and predictions to tally wins for each horse
        # for pred, (pair_features, race_id, course, race_name, horse1_id, horse2_id) in zip(pairwise_predictions, pairwise_data_test):
        #     if pred == 1:
        #         # If horse1 is predicted to beat horse2, increment horse1's score
        #         horse_scores[race_id][horse1_id] += 1
        #     else:
        #         # Otherwise, increment horse2's score
        #         horse_scores[race_id][horse2_id] += 1

        
        horse_data_df = pd.DataFrame(horse_data_with_predictions)

        # Display the DataFrame in Streamlit
        st.write("Horse Pairs with Predictions:")
        st.dataframe(horse_data_df)


    if st.button('BACKTEST: GB WIN with Pairs Deep Learning 2222222222'):
        
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        from keras.callbacks import Callback, EarlyStopping

        df_train = pd.read_csv('./data/df_train_UK_results.csv')
        df_train.reset_index(inplace=True, drop=True)
        # st.write(df_train[df_train['horse'] == 'Alphonse Le Grande (IRE)'])
        # st.write("df_train with no nans dropped is",df_train)
        label_encoder1 = LabelEncoder()
        label_encoder2 = LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        label_encoder6=LabelEncoder()
        label_encoder7=LabelEncoder()
        label_encoder8=LabelEncoder()
        label_encoder9=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        
        df_train['type'] = label_encoder3.fit_transform(df_train['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_train['horse'] = df_train['horse'].astype(str).str.replace(r"\s*\(.*?\)", "", regex=True)
        df_train['horse'] = label_encoder6.fit_transform(df_train['horse'])
        df_train['jockey'] = label_encoder7.fit_transform(df_train['jockey'])
        df_train['trainer'] = label_encoder8.fit_transform(df_train['trainer'])
        df_train['owner'] = label_encoder9.fit_transform(df_train['owner'])
        df_train['sp_dec'] = pd.to_numeric(df_train['sp_dec'], errors='coerce')

        # long_shot_winners = df_train[(df_train['sp_dec'] >= 12) & (df_train['HorseWin'] == 1)]


        # race_ids = long_shot_winners['race_id']  

        # df_train = df_train[df_train['race_id'].isin(race_ids)]

  
        #choose course to optimize for
       
        
        df_cleaned = df_train.dropna(subset=['age',
                            'dist_y','weight_lbs',
                             'going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank','horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage',
          'trainer_win_percentage','trainer_place_percentage','draw','course2','rpr','owner'])
        df_train = df_cleaned.reset_index(drop=True)
        df_train['tote_win']= df_train['tote_win'].replace('[£,]', '', regex=True).astype(float)
        # st.write("df_train with dropped nans is:",df_train)


        ############# DF_TEST RACECARD DATA #############

        def safe_transform(df, column, encoder):
            try:
                df[column] = encoder.transform(df[column])
            except ValueError as e:
                # Handle missing labels by setting a default value or handling in another way
                print("Encountered unknown label(s) in column '{}': {}".format(column, e))
                # Optionally, set a default value like -1 or NaN for unknown labels
                df[column] = df[column].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)


        df_test=pd.read_csv('./data/df_test_UK.csv')
        df_cleaned = df_test.dropna(subset=['age',
                            'distance_f','lbs','going'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)
        df_test = df_test.rename(columns={'horse_name': 'horse'})
        df_test['horse'] = df_test['horse'].str.replace(r"\s*\(.*?\)", "", regex=True)

        try:
            df_test['going'] = label_encoder1.transform(df_test['going'])
        except ValueError as e:
            print(f"Encountered unknown labels: {e}")
            df_test['going'] = df_test['going'].apply(lambda x: label_encoder1.transform([x])[0] if x in label_encoder1.classes_ else -1)  # assign -1 to unseen labels
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test['type'] = label_encoder3.transform(df_test['type'])
        df_test['sex'] = label_encoder4.transform(df_test['sex_code'])
        df_test['course2'] = label_encoder5.transform(df_test['course'])
        # df_test['sp_dec'] = df_test['odds'].apply(lambda x: extract_decimal_odds_for_bookmaker(x, selected_bookmaker))
        safe_transform(df_test, 'horse', label_encoder6)
        safe_transform(df_test, 'jockey', label_encoder7)
        safe_transform(df_test, 'trainer', label_encoder8)
        safe_transform(df_test, 'owner', label_encoder9)



        df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
          'horse_win_percentage', 'horse_place_percentage',
          'jockey_ave_rank', 'last_jockey_finish_pos', 'second_last_jockey_finish_pos', 'third_last_jockey_finish_pos',
          'jockey_win_percentage', 'jockey_place_percentage',
          'trainer_ave_rank', 'last_trainer_finish_pos', 'second_last_trainer_finish_pos', 'third_last_trainer_finish_pos',
          'trainer_win_percentage', 'trainer_place_percentage',
          'owner_ave_rank', 'last_owner_finish_pos', 'second_last_owner_finish_pos', 'third_last_owner_finish_pos',
          'owner_win_percentage', 'owner_place_percentage']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK_with_win_percentage(row, df_test, df_train),
            axis=1
        )

        #choose course to optimize for
#         df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
#           'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
#           'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
#           'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
#             lambda row: compute_average_ranks_all_UK(row, df_test, df_train),
#             axis=1
# )       
        
      
        df_test.reset_index(inplace=True, drop=True)
        # st.write("df_train with dropped nans is:",df_train)
        # df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
        df_test['distance_f'] = df_test['distance_f'] * 201.168
        df_tote_win = df_train[['race_id','horse_id', 'tote_win','HorseWin','HorseRankTop3']]

        # Merge df_test with df_tote_win on the race_id column
        df_test = pd.merge(df_test, df_tote_win, on=['race_id','horse_id'], how='left')
        df_test.to_csv('./data/df_test_UK2.csv',index=False)


        df_cleaned = df_test.dropna(subset=['age',
                            'distance_f','lbs','going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank','horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage',
          'trainer_win_percentage','trainer_place_percentage','draw','course2','rpr','owner'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)

        st.write(df_test['odds'].iloc[0]) 
        import ast
    





        
       
        start_date = '2024-10-01'  # Example start date
        days_to_predict = 15
        all_predictions = []

        

    
        # 1. Get the training data up to the current day
        current_train = df_train[df_train['date'] <= start_date]
        # if len(df_train['race_id'].unique()) >= 3300:
        #     # Take a random sample of 3,300 unique race IDs
        #     sampled_race_ids = current_train['race_id'].drop_duplicates().sample(n=3300, random_state=1)  # random_state for reproducibility
            
        #     # Create a new DataFrame with the sampled race IDs
        #     current_train = df_train[df_train['race_id'].isin(sampled_race_ids)].reset_index(drop=True)
                
                



        X_train = current_train[['horse_id','HorseWin','course','race_name','race_id','position','age', 'type', 'sex',
                        'weight_lbs','dist_m',
                            'going','recent_ave_rank', 
        'jockey_ave_rank',  
        'trainer_ave_rank', 
        'owner_ave_rank','horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage',
        'trainer_win_percentage','trainer_place_percentage','draw','course2','rpr','owner','time']]
        # X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
    

        
        def time_to_seconds(time_str):
            try:
                if pd.isna(time_str):
                    # If the time is NaN, return None
                    return None

                # Convert the value to string in case it's a float
                time_str = str(time_str)

                # Split the time string by ':'
                parts = time_str.split(':')
                
                # Handle cases where the string might not be in the correct format
                if len(parts) == 2:  # Format like "4:12.11"
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                elif len(parts) == 1:  # Format like "12.11" (no minutes part)
                    minutes = 0
                    seconds = float(parts[0])
                else:
                    raise ValueError(f"Unexpected time format: {time_str}")
                
                # Convert minutes and seconds to total seconds
                return minutes * 60 + seconds
            except (ValueError, IndexError) as e:
                # Handle cases where the time string is invalid
                print(f"Error converting time: {time_str}. Error: {e}")
                return None  # Return None in case of an error
        X_train['time'].replace(['-', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        X_train.dropna(subset=['time'], inplace=True)
        y_train = X_train['time'].apply(time_to_seconds)
        y_train = y_train.values.reshape(-1, 1) 
    
            


            # start_date = '2023-10-16'
            # end_date = '2024-10-16'

            # # Filter X_train and X_test for the date range
            # X_train_filtered = X_train[(X_train['date'] >= start_date) & (X_train['date'] <= end_date)]
            
            # X_train = X_train_filtered.drop(columns=['time','date'])
        X_train = X_train.drop(columns=['time'])


        import tensorflow as tf
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        # 5. Define and compile model
        model = Sequential([
            Dense(128, activation='tanh', input_shape=(44,)),
            Dense(128, activation='tanh'),
            Dense(128, activation='tanh'),
            Dense(128, activation='tanh'),
            Dense(128, activation='tanh'),
            Dense(1)
        ])
        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder
            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")
        model.compile(optimizer=Adam(learning_rate=0.0001),
            loss='mean_squared_error',
            metrics=['mae'])

        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)
        # Fit the model with both callbacks
        history = model.fit(X_train_scaled, y_train_scaled, epochs=5000, batch_size=16, validation_split=0.2,
                            callbacks=[streamlit_callback, early_stopping], verbose=0)
        epochs = range(1, len(history.history['loss']) + 1)
        # next_day = pd.to_datetime(start_date) + pd.Timedelta(days=1)
        # df_test_day = df_test[df_test['date'] == next_day.strftime('%Y-%m-%d')]

        # # Debugging: Check if df_test_day has any data
        # st.write(f"Test data for {next_day.strftime('%Y-%m-%d')}:")
        # st.write(df_test_day)
        
        start_date = next_day.strftime('%Y-%m-%d')  # Move to the next date
            


        # Now select the columns
        X_test = df_test[['horse_id','course','race_name','race_id','position','age', 'type', 'sex', 'weight_lbs', 'dist_m', 'going', 
                            'recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank', 
                            'owner_ave_rank','horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage',
        'trainer_win_percentage','trainer_place_percentage','draw','course2','rpr','owner']]
        
        def create_pairwise_data_by_race(df):
            from itertools import combinations
            import random
            print("processing data into pairs")
            pairs = []
            features = ['age',
                            'distance_f','lbs','going','recent_ave_rank',
                            'jockey_ave_rank', 
                            'trainer_ave_rank', 
                            'owner_ave_rank','horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage',
                            'trainer_win_percentage','trainer_place_percentage','draw','course2','rpr','owner']
            df = df.sort_values(by=['race_id', 'position'])
            for race_id, race_df in df.groupby('race_id'):
                if len(race_df) < 2:
                    continue  
                # Shuffle horse indices to randomize the order of comparisons
                horse_indices = list(race_df.index)
                random.shuffle(horse_indices)
                for horse1_idx, horse2_idx in combinations(horse_indices, 2):
                    horse1_features = df.loc[horse1_idx, features].values
                    horse2_features = df.loc[horse2_idx, features].values
                    pair_features = list(horse1_features) + list(horse2_features)
                    # Compare positions
                    pos1 = df.loc[horse1_idx, 'position']
                    pos2 = df.loc[horse2_idx, 'position']
                    # Result is 1 if horse1 finished ahead of horse2, else 0
                    result = 1 if pos1 < pos2 else 0
                    # Get horse IDs
                    horse1_id = df.loc[horse1_idx, 'horse_id']  # Adjust this line based on your DataFrame
                    horse2_id = df.loc[horse2_idx, 'horse_id']  # Adjust this line based on your DataFrame
                    course=df.loc[horse1_idx, 'course']
                    race_name=df.loc[horse1_idx, 'race_name']
                    # Include both horse IDs and race_id with the pair
                    pairs.append((pair_features, result, race_id,course,race_name, horse1_id, horse2_id))
            return pairs
        
        def create_pairwise_data_by_race_test(df):
            from itertools import combinations
            import random
            print("processing data into pairs")
            pairs = []
            features = ['age',
                            'distance_f','lbs','going','recent_ave_rank',
                        'jockey_ave_rank', 
                        'trainer_ave_rank', 
                        'owner_ave_rank','horse_win_percentage','horse_place_percentage','horse','jockey','trainer','jockey_win_percentage','jockey_place_percentage',
                        'trainer_win_percentage','trainer_place_percentage','draw','course2','rpr','owner']

            # Sort data by race_id (for consistency)
            df = df.sort_values(by=['race_id'])

            for race_id, race_df in df.groupby('race_id'):
                if len(race_df) < 2:
                    continue  # Skip races with fewer than 2 horses

                # Shuffle horse indices to randomize the order of comparisons
                horse_indices = list(race_df.index)
                random.shuffle(horse_indices)

                for horse1_idx, horse2_idx in combinations(horse_indices, 2):
                    horse1_features = df.loc[horse1_idx, features].values
                    horse2_features = df.loc[horse2_idx, features].values
                    pair_features = list(horse1_features) + list(horse2_features)

                    # Get horse IDs and race details
                    horse1_id = df.loc[horse1_idx, 'horse_id']
                    horse2_id = df.loc[horse2_idx, 'horse_id']
                    course = df.loc[horse1_idx, 'course']
                    race_name = df.loc[horse1_idx, 'race_name']

                    # Append the pair (no result)
                    pairs.append((pair_features, race_id, course, race_name, horse1_id, horse2_id))

            return pairs
                
        pairwise_data_train = create_pairwise_data_by_race(X_train)
        pairwise_data_test = create_pairwise_data_by_race_test(X_test)
        print("pairwise data is :", pairwise_data_train)

       

       
        X = np.array([pair[0] for pair in pairwise_data_train])  # Input features (horse1 vs horse2)
        y = np.array([pair[1] for pair in pairwise_data_train])  # Target labels (1 for win, 0 for loss)
                # Ensure X_train and X_test are 2D arrays (with shape (n_samples, n_features))
        y=y.reshape(-1,1)

        print(X)  # Should print something like (n_pairs, n_features)
        print(y)
        print(X_train['position'].value_counts())


        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X)  # Fit and transform the training data
        y_train_scaled = scaler_y.fit_transform(y)
        # print(X_train_scaled[:50])  
        # print(y_train_scaled[:50])  
      
        # Define the model
        
        # Build a simple feedforward neural network
 
      

        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")

            # sgd_optimizer = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',  #
              metrics=['accuracy']) 

        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)

        # Fit the model with both callbacks
        history = model.fit(X_train_scaled, y, epochs=20, batch_size=100, validation_split=0.2,
                            callbacks=[streamlit_callback, early_stopping], verbose=0)
        epochs = range(1, len(history.history['loss']) + 1)

        X2 = np.array([pair[0] for pair in pairwise_data_test])  # Input features (horse1 vs horse2)
        print("Size of X2:", X2.shape)
        X_test_scaled = scaler_x.fit_transform(X2)
       
       # Streamlit app
        st.title('Loss Curve Plot')

        # Plotting function
        def plot_loss(history):
            plt.figure(figsize=(10, 6))
            plt.scatter(epochs,history.history['loss'], label='Training Loss')
            plt.scatter(epochs,history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        # Display the plot
        st.pyplot(plot_loss(history))
        DL_pred = model.predict(X_test_scaled) 

        pairwise_predictions = (DL_pred > 0.5).astype(int)


        from collections import defaultdict

        # Initialize a dictionary to hold the scores for each horse in each race
        horse_scores = defaultdict(lambda: defaultdict(int))

        # Loop through the pairwise data and predictions to tally wins for each horse
        for pred, (pair_features, race_id, course, race_name, horse1_id, horse2_id) in zip(pairwise_predictions, pairwise_data_test):
            if pred == 1:
                # If horse1 is predicted to beat horse2, increment horse1's score
                horse_scores[race_id][horse1_id] += 1
            else:
                # Otherwise, increment horse2's score
                horse_scores[race_id][horse2_id] += 1

        # Now horse_scores contains the number of wins for each horse in each race


           #Create a DataFrame to hold the predicted winners and their horse IDs
        predicted_winners_dict = {
            'race_id': [],
            'predicted_winner': [],
            'horse_id': []
        }

        for race_id, scores in horse_scores.items():
            predicted_winner = max(scores, key=scores.get)
      
            # Get the horse_id corresponding to the predicted winner
            horse_id = predicted_winner  # In this case, horse_id is the same as the predicted_winner
            predicted_winners_dict['race_id'].append(race_id)
            predicted_winners_dict['predicted_winner'].append(predicted_winner)
            predicted_winners_dict['horse_id'].append(horse_id)

        predicted_winners_df = pd.DataFrame(predicted_winners_dict)

        # Assuming X_unseen contains the actual winners of each race
        # Adjust this according to your actual structure
        # actual_winners = X_test.groupby('race_id')['horse_id'].first().reset_index()
        # st.write("actual winners is",actual_winners)
        # actual_winners.columns = ['race_id', 'actual_winner']

        # Merge actual winners with predicted winners
        comparison = pd.merge(df_test, predicted_winners_df, on='race_id',how='left')
        comparison.to_csv('./data/comparison_results.csv', index=False)
        
        st.write(comparison)
        # # Calculate accuracy
        # accuracy = (comparison['actual_winner'] == comparison['predicted_winner']).mean()
        # print(f'Prediction accuracy: {accuracy * 100:.2f}%')




    st.subheader("PREDICT")
    

    if st.button('PREDICT: GB WIN with Logistic Regression'):
        df_train = pd.read_csv('./data/df_train_UK_results.csv')
        df_train.reset_index(inplace=True, drop=True)
        df_test=pd.read_csv('./data/df_test_UK.csv')
        df_test[['recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank','owner_ave_rank']] = df_test.apply(
        lambda row: compute_average_ranks_UK(row, df_test, df_train),
        axis=1)
        df_test.to_csv('./data/df_test_UK2.csv',index=False)
        df_test.reset_index(inplace=True, drop=True)
        df_cleaned = df_train.dropna(subset=['draw',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        # Reset index
        df_train = df_cleaned.reset_index(drop=True)

        df_cleaned = df_test.dropna(subset=['draw',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)
        X_train = df_train[['draw',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank']]
        X_test = df_test[['draw',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank']]
        y_train = df_train[['HorseWin']]
        
        time_data=df_train[['race_id','draw',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank','time','winning_time_detail']]
        time_data['winning_time_detail']=time_data['winning_time_detail'].apply(winning_time_to_seconds)
        time_data['time']=time_data['time'].apply(time_to_seconds)
        time_data.to_csv('./data/train_time_data.csv')

#         from sklearn.model_selection import train_test_split
#         import tensorflow as tf
#         from tensorflow import keras
#         import keras
#         from keras.models import Sequential
#         from keras.layers import Dense,Input,Dropout
#         from keras.optimizers import Adam
#         from sklearn.utils import class_weight
#         from keras.regularizers import l2
#         from sklearn.preprocessing import LabelEncoder
#         df_train = pd.read_csv('./data/df_train_UK.csv')
#         df_train.reset_index(inplace=True, drop=True)
#         df_test=pd.read_csv('./data/df_test_UK.csv')
#         st.write("df_train with no nans dropped is",df_train)
#         label_encoder1 = LabelEncoder()
#         label_encoder2 = LabelEncoder()
#         label_encoder3=LabelEncoder()
#         label_encoder4=LabelEncoder()
#         label_encoder5=LabelEncoder()
#         # Fit and transform the 'going' column
#         df_train['going'] = label_encoder1.fit_transform(df_train['going'])
#         df_test['going'] = label_encoder1.fit_transform(df_test['going'])
#         df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
#         df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
#         df_train['race_type'] = label_encoder3.fit_transform(df_train['race_type'])
#         df_test['type'] = label_encoder3.fit_transform(df_test['type'])
#         df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
#         df_test['sex'] = label_encoder4.fit_transform(df_test['sex'])
#         df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
#         df_test['course2'] = label_encoder5.fit_transform(df_test['course'])
#         #choose course to optimize for
#         df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
#           'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
#           'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
#           'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
#             lambda row: compute_average_ranks_all_UK(row, df_test, df_train),
#             axis=1
# )
#         df_test.to_csv('./data/df_test_UK2.csv',index=False)
#         df_test.reset_index(inplace=True, drop=True)
#         df_cleaned = df_train.dropna(subset=['draw','age',
#                             'dist_y','weight_lbs','prize',
#                              'rpr','going','recent_ave_rank',  
#           'jockey_ave_rank', 
#           'trainer_ave_rank', 
#           'owner_ave_rank'])
#         df_train = df_cleaned.reset_index(drop=True)
#         st.write("df_train with dropped nans is:",df_train)
#         df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
#         df_test['distance_f'] = df_test['distance_f'] * 201.168
#         df_cleaned = df_test.dropna(subset=['draw','age',
#                             'distance_f','lbs','prize'
#                              ,'rpr','going','recent_ave_rank',
#           'jockey_ave_rank', 
#           'trainer_ave_rank', 
#           'owner_ave_rank'])
#         # Reset index
#         df_test = df_cleaned.reset_index(drop=True)
#         X_train = df_train[['draw','age', 'race_type', 'sex','course2',
#                             'weight_lbs','dist_m',
#                              'rpr','going','recent_ave_rank',  
#           'jockey_ave_rank',  
#           'trainer_ave_rank', 
#           'owner_ave_rank']]
#         X_test = df_test[['draw','age','type','sex','course2',
#                             'lbs','distance_f',
#                              'rpr','going','recent_ave_rank', 
#           'jockey_ave_rank',  
#           'trainer_ave_rank',
#           'owner_ave_rank']]
        
#         scaler_X = StandardScaler()
       
#         X_train_scaled = scaler_X.fit_transform(X_train)  # Fit and transform the training data
#         X_test_scaled=scaler_X.fit_transform(X_test)
#         y_train = df_train[['HorseWin']]
#         import tensorflow as tf
#         import numpy as np
#         import random
#         np.random.seed(42)
#         random.seed(42)
#         tf.random.set_seed(42)


        lr =  LogisticRegression(solver='lbfgs', penalty='l2', C=1, max_iter=350)
        # Do hyperparameter tuning
        param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                        'penalty': [ 'l2']}
        kfold = KFold(n_splits=5)
        grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5)
        # find the best parameters
        grid.fit(X_train, y_train['HorseWin'].to_numpy())
        # Print the best parameters
        print(grid.best_params_)
        print(grid.best_score_)
        # Initialize the model using best parameters
        lr = grid.best_estimator_
        print("Model coefficients:",lr.coef_)
        # Create a dataframe to store the predictions
        df_pred = pd.DataFrame()
        df_pred['Date']=df_test['date']
        df_pred['RaceID'] = df_test['race_id']
        df_pred['Track']=df_test['course']
        df_pred['Race Name']=df_test['race_name']
        df_pred['HorseID'] = df_test['horse_name']
        # Make predictions
        y_test_pred = lr.predict(X_test)
        # Store the predictions in the dataframe
        df_pred['HorseWin'] = y_test_pred
        # Save predictions to csv
        pd.DataFrame(df_pred).to_csv('./predictions/deploy_pred_UK.csv')
        # Filter the DataFrame to include only rows where HorseWin = 1
        winning_horses = df_pred[df_pred['HorseWin'] == 1]
        # # Step 2: Count occurrences of each race_id and find those with more than one occurrence
        # race_id_counts = winning_horses['RaceID'].value_counts()
        # duplicate_race_ids = race_id_counts[race_id_counts > 1].index

        # # Step 3: Exclude races with duplicate race_ids
        # filtered_winning_horses = winning_horses[~winning_horses['RaceID'].isin(duplicate_race_ids)]

        # # Reset the index of the filtered DataFrame
        # filtered_winning_horses = filtered_winning_horses.reset_index(drop=True)

        # Display the filtered DataFrame in Streamlit
        st.title("Bet on the races below")
        st.dataframe(winning_horses)
    
    if st.button('PREDICT: GB WIN with Persistent Homology'):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        from keras.callbacks import Callback, EarlyStopping

        df_train = pd.read_csv('./data/df_train_UK_racecards.csv')
        df_test=pd.read_csv('./data/df_test.csv')
        df_train.reset_index(inplace=True, drop=True)
        label_encoder1 = LabelEncoder()
        label_encoder2 = LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        label_encoder6=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['type'] = label_encoder3.fit_transform(df_train['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_train['surface'] = label_encoder6.fit_transform(df_train['surface'])
        # df_train['sp_dec'] = pd.to_numeric(df_train['sp_dec'], errors='coerce')
        df_train['ovr_btn'] = pd.to_numeric(df_train['ovr_btn'], errors='coerce')
        df_train['horse_id'] = pd.factorize(df_train['horse_id'])[0]
        #choose course to optimize for
        # df_train = df_train.drop(columns=['recent_ave_rank','jockey_ave_rank','trainer_ave_rank','owner_ave_rank'])
        df_cleaned = df_train.dropna(subset=['draw','age',
                            'lbs',
                             'going','rpr','type','distance_f','sex'])
        df_train = df_cleaned.reset_index(drop=True)
        df_train['tote_win']= df_train['tote_win'].replace('[£,]', '', regex=True).astype(float)
        numeric_columns = ['age', 'draw','type','sex','lbs','distance_f','going', 'rpr']
        # Coerce each column and drop rows with invalid data
        for col in numeric_columns:
            df_train = df_train[pd.to_numeric(df_train[col], errors='coerce').notna()]
        import pandas as pd
        df_test['going'] = label_encoder1.transform(df_test['going'])
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test['type'] = label_encoder3.transform(df_test['type'])
        df_test['sex'] = s.transform(df_test['sex'])
        df_test['course2'] = label_encoder5.transform(df_test['course'])
        df_test['surface'] = label_encoder6.transform(df_test['surface'])
        df_test = df_test.apply(lambda row: compute_average_ranks_all_UK_with_win_percentage(row, df_test, df_train), axis=1)



        X_train = df_train[['draw','age', 'type', 'sex',
                            'lbs','distance_f',
                             'going','recent_ave_rank',  
            'jockey_ave_rank',  
            'trainer_ave_rank', 
            'owner_ave_rank','rpr','time']]
        
        df_cleaned = df_test.dropna(subset=['draw','age', 'type', 'sex',
                            'lbs','distance_f',
                             'going','recent_ave_rank',  
            'jockey_ave_rank',  
            'trainer_ave_rank', 
            'owner_ave_rank','rpr'])
        df_test = df_cleaned.reset_index(drop=True)

        X_test = df_test[['draw','age', 'type', 'sex',
                            'lbs','distance_f',
                             'going','recent_ave_rank',  
            'jockey_ave_rank',  
            'trainer_ave_rank', 
            'owner_ave_rank','rpr']]
        
        


          
        def time_to_seconds(time_str):
            try:
                if pd.isna(time_str):
                    # If the time is NaN, return None
                    return None

                # Convert the value to string in case it's a float
                time_str = str(time_str)

                # Split the time string by ':'
                parts = time_str.split(':')
                
                # Handle cases where the string might not be in the correct format
                if len(parts) == 2:  # Format like "4:12.11"
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                elif len(parts) == 1:  # Format like "12.11" (no minutes part)
                    minutes = 0
                    seconds = float(parts[0])
                else:
                    raise ValueError(f"Unexpected time format: {time_str}")
                
                # Convert minutes and seconds to total seconds
                return minutes * 60 + seconds
            except (ValueError, IndexError) as e:
                # Handle cases where the time string is invalid
                print(f"Error converting time: {time_str}. Error: {e}")
                return None  # Return None in case of an error
        X_train['time'].replace(['-', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        X_train.dropna(subset=['time'], inplace=True)
        
        y_train = X_train['time'].apply(time_to_seconds)
        y_train = y_train.values.reshape(-1, 1) 

        X_train = X_train.drop(columns=['time'])


       
        def create_features(df):
            df = df.sort_values(by=['horse_id', 'date'], ascending=[True, False])
            grouped = df.groupby('horse_id')
            # Create features
            features = []
            for horse_id, group in grouped:
                group = group.head(6)  # Get the last 6 races
                feature_row = {'horse_id': horse_id}
                for i, (_, race) in enumerate(group.iterrows(), start=1):
                    feature_row.update({
                        f"race_{i}_draw": race['draw'],
                        f"race_{i}_age": race['age'],
                        f"race_{i}_type": race['type'],
                        f"race_{i}_sex": race['sex'],
                        f"race_{i}_lbs": race['lbs'],
                        f"race_{i}_recent_ave_rank": race['recent_ave_rank'],
                        f"race_{i}_jockey_ave_rank": race['jockey_ave_rank'],
                        f"race_{i}_trainer": race['trainer_ave_rank'],
                        f"race_{i}_owner": race['owner_ave_rank'],
                        f"race_{i}_lbs": race['lbs'],
                        f"race_{i}_distance_f": race['distance_f'],
                        f"race_{i}_going": race['going'],
                        f"race_{i}_rpr": race['rpr'],
                    })
                features.append(feature_row)

            # Create a new DataFrame from features
            features_df = pd.DataFrame(features)
            return features_df
        features_df = create_features(df_train)
        features_df = features_df.fillna(0)
        features_df = features_df.apply(pd.to_numeric, errors='coerce')
        st.write(features_df)
        # Assuming your features are stored in features_df, excluding the 'horse_id' column
        X = features_df.drop(columns=['horse_id']).values
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        from mpl_toolkits.mplot3d import Axes3D
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import gudhi as gd

        # Sidebar for configuration
        st.sidebar.title("PCA and Persistent Homology")
        # components = st.sidebar.slider("Number of PCA Components", 2, 3, value=2)
        components=2
        # Standardize the data
        X = features_df.drop(columns=['horse_id']).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=components)
        X_pca = pca.fit_transform(X_scaled)
        st.write("X PCA is",X_pca)

        from sklearn.cluster import KMeans
        # Fit KMeans to the PCA data (assuming you have already done PCA and have X_pca)
        n_clusters = 5  # You can adjust this based on your data or use heuristics to find optimal clusters
        kmeans = KMeans(n_clusters=n_clusters)
        horse_id_labels = kmeans.fit_predict(X_pca)  # Assign cluster labels based on the clustering

        features_df['cluster'] = horse_id_labels
        
        # # Display PCA Visualization
        # st.title("Horse Racing Data: PCA Visualization")
        # if components == 2:
        #     fig, ax = plt.subplots()
        #     ax.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', s=10)
        #     ax.set_title("2D PCA Projection")
        #     ax.set_xlabel("Principal Component 1")
        #     ax.set_ylabel("Principal Component 2")
        #     st.pyplot(fig)


        model = Sequential([
            Input(shape=(7,)),  
            Dense(60, activation='tanh'),  
            Dense(60, activation='tanh'),
            Dense(1)])  

        from keras.callbacks import Callback, EarlyStopping

        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder
            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")
        model.compile(optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae'])
        # from keras.callbacks import ReduceLROnPlateau

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)

        # Fit the model with both callbacks
        
        history = model.fit(X_train_scaled,y_train_scaled,
                    epochs=5000,
                    validation_split=0.2,  # Pass the validation generator here
                    batch_size=7,
                    callbacks=[streamlit_callback, early_stopping],
                    verbose=0)
        epochs = range(1, len(history.history['loss']) + 1)

        tf.keras.utils.plot_model(model, to_file='./data/model.png', show_shapes=True, show_layer_names=True)

        st.title('Loss Curve Plot')
        # Plotting function
        def plot_loss(history):
            plt.figure(figsize=(10, 6))
            plt.scatter(epochs,history.history['loss'], label='Training Loss')
            plt.scatter(epochs,history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        # Display the plot
        st.pyplot(plot_loss(history))

        DL_pred = model.predict(X_test_scaled) #what should k-fold be?
        st.write("Model Predictions:")
        DL_pred_unscaled = scaler_y.inverse_transform(DL_pred)
        df_pred = pd.DataFrame()
        df_pred['Date'] = df_test['date']
        df_pred['Course Name']=df_test['course']
        df_pred['HorseID'] = df_test['horse']
        df_pred['RaceID'] = df_test['race_id']
        df_pred['Race Name'] = df_test['race_name']
        df_pred['Horse Number']=df_test['number']
        df_pred['finish_time'] = DL_pred_unscaled
        # df_pred_kempton = df_pred[df_pred['Course Name'] == 'Ayr']
        df_sorted = df_pred.sort_values(by=['RaceID', 'finish_time'])
        df_sorted['HorseWin'] = df_sorted.groupby('RaceID')['finish_time'].transform(lambda x: x == x.min()).astype(int)
        df_sorted['HorseRankTop3'] = df_sorted.groupby('RaceID')['finish_time'].rank(method='min', ascending=True) <= 3
        df_sorted['HorseRankTop3'] = df_sorted['HorseRankTop3'].astype(int)
        winning_horses = df_sorted[df_sorted['HorseWin'] == 1]
        filtered_winning_horses = winning_horses.reset_index(drop=True)
        # Display the filtered DataFrame in Streamlit
        st.title("Bet on the races below")
        st.dataframe(filtered_winning_horses)

        

    

        

# Now 'features_df' contains a 'cluster' column, where each horse_id has a cluster label
        # st.write(features_df[['horse_id', 'cluster']])

        # # Plot with clusters
        # cmap = plt.cm.get_cmap('tab10', n_clusters)  # Ensure enough colors for clusters
        # scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=horse_id_labels, cmap=cmap, s=10)
        # ax.set_title("2D PCA Projection with Clusters")
        # ax.set_xlabel("Principal Component 1")
        # ax.set_ylabel("Principal Component 2")

        # # Add a color legend
        # legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
        # ax.add_artist(legend1)
        # st.pyplot(fig)
        # from ripser import ripser

        # result = ripser(X_pca, maxdim=1)

        # # Extract persistence diagram
        # diagram_0d = result['dgms'][1]  # 0D features (connected com

        #         # Streamlit app title
        # st.title("Persistence Diagrams Visualization")

        # # Create a subplot with 3 columns for different dimensions
        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # # Plot 0D features (connected components)
        # axs[0].scatter(diagram_0d[:, 0], diagram_0d[:, 1], label="0D (connected components)")
        # axs[0].set_title("0D Persistence Diagram")
        # axs[0].set_xlabel("Birth")
        # axs[0].set_ylabel("Death")
        
        # st.pyplot(fig)




    if st.button('PREDICT: GB WIN with SMOTE+LR'):

        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        from keras.callbacks import Callback, EarlyStopping

        df_train = pd.read_csv('./data/df_train_UK_racecards.csv')
        df_train.reset_index(inplace=True, drop=True)
         # X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
        # df_train = df_train[df_train['type'].isin(['Flat', 'NH Flat'])]
        # df_train[df_train['date']=="2024-09-04"].to_csv('./data/df_test_backtest.csv',index=False)
        df_train = df_train[pd.to_numeric(df_train['age'], errors='coerce').notna()]

        # st.write(df_train[df_train['horse'] == 'Alphonse Le Grande (IRE)'])

        # st.write("df_train with no nans dropped is",df_train)
        label_encoder1 = LabelEncoder()
        label_encoder2 = LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        label_encoder6=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['type'] = label_encoder3.fit_transform(df_train['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_train['surface'] = label_encoder6.fit_transform(df_train['surface'])
        # df_train['sp_dec'] = pd.to_numeric(df_train['sp_dec'], errors='coerce')
        df_train['ovr_btn'] = pd.to_numeric(df_train['ovr_btn'], errors='coerce')

        #choose course to optimize for
        
        df_cleaned = df_train.dropna(subset=['draw','age',
                            'lbs',
                             'going','recent_ave_rank', 
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank','rpr','ovr_btn','surface'])
        
        df_train = df_cleaned.reset_index(drop=True)
        df_train['tote_win']= df_train['tote_win'].replace('[£,]', '', regex=True).astype(float)
        # st.write("df_train with dropped nans is:",df_train)
       
        
       
        start_date = '2024-02-01'  # Example start date
        days_to_predict =10
        all_predictions = []

        

        for day in range(days_to_predict):
            # 1. Get the training data up to the current day
            current_train = df_train[df_train['date'] <= start_date]
            if len(current_train['race_id'].unique()) >= 3300:
                # Take a random sample of 3,300 unique race IDs
                sampled_race_ids = current_train['race_id'].drop_duplicates().sample(n=3300, random_state=1)  # random_state for reproducibility
                
                # Create a new DataFrame with the sampled race IDs
                current_train = current_train[current_train['race_id'].isin(sampled_race_ids)].reset_index(drop=True)
                

                # long_shot_winners = current_train[(current_train['sp_dec'] >= 12) & (current_train['HorseWin'] == 1)]


                # race_ids = long_shot_winners['race_id']  

                # current_train = current_train[current_train['race_id'].isin(race_ids)]

        
                    



            X_train = current_train[['draw','age', 'type', 'sex',
                            'lbs','distance_f',
                             'going','recent_ave_rank',  
            'jockey_ave_rank',  
            'trainer_ave_rank', 
            'owner_ave_rank','rpr','ovr_btn','surface','time']]
        

           

            from sklearn.impute import KNNImputer

          
            def time_to_seconds(time_str):
                try:
                    if pd.isna(time_str):
                        # If the time is NaN, return None
                        return None

                    # Convert the value to string in case it's a float
                    time_str = str(time_str)

                    # Split the time string by ':'
                    parts = time_str.split(':')
                    
                    # Handle cases where the string might not be in the correct format
                    if len(parts) == 2:  # Format like "4:12.11"
                        minutes = int(parts[0])
                        seconds = float(parts[1])
                    elif len(parts) == 1:  # Format like "12.11" (no minutes part)
                        minutes = 0
                        seconds = float(parts[0])
                    else:
                        raise ValueError(f"Unexpected time format: {time_str}")
                    
                    # Convert minutes and seconds to total seconds
                    return minutes * 60 + seconds
                except (ValueError, IndexError) as e:
                    # Handle cases where the time string is invalid
                    print(f"Error converting time: {time_str}. Error: {e}")
                    return None  # Return None in case of an error
            X_train['time'].replace(['-', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
            X_train.dropna(subset=['time'], inplace=True)
          
            y_train = X_train['time'].apply(time_to_seconds)
            y_train = y_train.values.reshape(-1, 1) 
        
            


            # start_date = '2023-10-16'
            # end_date = '2024-10-16'

            # # Filter X_train and X_test for the date range
            # X_train_filtered = X_train[(X_train['date'] >= start_date) & (X_train['date'] <= end_date)]
            
            # X_train = X_train_filtered.drop(columns=['time','date'])
            X_train = X_train.drop(columns=['time'])

            


            from sklearn.impute import KNNImputer


            from sklearn.linear_model import LinearRegression
            import pandas as pd

            # X_train_features = X_train.drop(columns=['rpr'])  # The 11 features
            # y_train_rpr = X_train['rpr']  # Target column

            # # 2. Train a regression model
            # regressor = LinearRegression()
            # regressor.fit(X_train_features, y_train_rpr)

            
          







            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            # Now scale the cleaned data
            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train)

        # df_train = pd.read_csv('./data/df_train_UK_results.csv')
        # df_train.reset_index(inplace=True, drop=True)
        # df_test=pd.read_csv('./data/df_test_UK.csv')
        # df_test[['recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank','owner_ave_rank']] = df_test.apply(
        # lambda row: compute_average_ranks_UK(row, df_test, df_train),
        # axis=1)
        # df_test.to_csv('./data/df_test_UK2.csv',index=False)
        # df_test.reset_index(inplace=True, drop=True)
        # df_cleaned = df_train.dropna(subset=['draw',
        #                     'jockey_ave_rank',
        #                     'trainer_ave_rank', 'recent_ave_rank'])
        # df_train = df_cleaned.reset_index(drop=True)
        # df_cleaned = df_test.dropna(subset=['draw',
        #                     'jockey_ave_rank',
        #                     'trainer_ave_rank', 'recent_ave_rank'])
        # df_test = df_cleaned.reset_index(drop=True)
        # X_train = df_train[['draw',
        #                     'jockey_ave_rank',
        #                     'trainer_ave_rank', 'recent_ave_rank']]
        # X_test = df_test[['draw',
        #                     'jockey_ave_rank',
        #                     'trainer_ave_rank', 'recent_ave_rank']]
        # y_train = df_train[['HorseWin']]
        
        # time_data=df_train[['race_id','draw',
        #                     'jockey_ave_rank',
        #                     'trainer_ave_rank', 'recent_ave_rank','time','winning_time_detail']]
        # time_data['winning_time_detail']=time_data['winning_time_detail'].apply(winning_time_to_seconds)
        # time_data['time']=time_data['time'].apply(time_to_seconds)
        # time_data.to_csv('./data/train_time_data.csv')
        
        lr = LogisticRegression(solver='lbfgs', penalty='l2', C=1, max_iter=500)

        # Set up the parameter grid for hyperparameter tuning
        param_grid = {'logisticregression__C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                    'logisticregression__penalty': ['l2']}

        # Create a pipeline that first applies SMOTE, then fits the logistic regression model
        smote = SMOTE(random_state=42)
        pipeline = Pipeline([('smote', smote), ('logisticregression', lr)])

        # Set up cross-validation with KFold
        kfold = KFold(n_splits=5)

        # Perform grid search with cross-validation
        grid = GridSearchCV(estimator=smote, param_grid=param_grid, cv=kfold)

        # Fit the model
        grid.fit(X_train_scaled, y_train_scaled['HorseWin'].to_numpy())
        # Print the best parameters
        print(grid.best_params_)
        print(grid.best_score_)
        # Initialize the model using best parameters
        lr = grid.best_estimator_
        # Create a dataframe to store the predictions
        df_pred = pd.DataFrame()
        df_pred['Date']=df_test['date']
        df_pred['RaceID'] = df_test['race_id']
        df_pred['Track']=df_test['course']
        df_pred['Race Name']=df_test['race_name']
        df_pred['HorseID'] = df_test['horse_name']
        # Make predictions
        y_test_pred = lr.predict(X_test)
        # Store the predictions in the dataframe
        df_pred['HorseWin'] = y_test_pred
        # Save predictions to csv
        pd.DataFrame(df_pred).to_csv('./predictions/deploy_pred_UK.csv')
        # Filter the DataFrame to include only rows where HorseWin = 1
        winning_horses = df_pred[df_pred['HorseWin'] == 1]
        # # Step 2: Count occurrences of each race_id and find those with more than one occurrence
        # race_id_counts = winning_horses['RaceID'].value_counts()
        # duplicate_race_ids = race_id_counts[race_id_counts > 1].index

        # # Step 3: Exclude races with duplicate race_ids
        # filtered_winning_horses = winning_horses[~winning_horses['RaceID'].isin(duplicate_race_ids)]

        # # Reset the index of the filtered DataFrame
        # filtered_winning_horses = filtered_winning_horses.reset_index(drop=True)

        # Display the filtered DataFrame in Streamlit
        st.title("Bet on the races below")
        st.dataframe(winning_horses)
    #
    if st.button("PREDICT: GB WIN with LGBM"):
        # df_train = pd.read_csv('./data/df_train_UK.csv')
        # df_train.reset_index(inplace=True, drop=True)
        # df_test=pd.read_csv('./data/df_test_UK.csv')
        # df_test[['recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank','owner_ave_rank']] = df_test.apply(
        # lambda row: compute_average_ranks_UK(row, df_test, df_train),
        # axis=1)
        # df_test.to_csv('./data/df_test_UK2.csv',index=False)
        # df_test.reset_index(inplace=True, drop=True)
        # df_cleaned = df_train.dropna(subset=['draw',
        #                     'jockey_ave_rank',
        #                     'trainer_ave_rank', 'recent_ave_rank'])
        # # Reset index
        # df_train = df_cleaned.reset_index(drop=True)
        
        # df_cleaned = df_test.dropna(subset=['draw',
        #                     'jockey_ave_rank',
        #                     'trainer_ave_rank', 'recent_ave_rank'])
        # # Reset index
        # df_test = df_cleaned.reset_index(drop=True)
        # X_train = df_train[['draw',
        #                     'jockey_ave_rank',
        #                     'trainer_ave_rank', 'recent_ave_rank']]
        # X_test = df_test[['draw',
        #                     'jockey_ave_rank',
        #                     'trainer_ave_rank', 'recent_ave_rank']]
        # y_train = df_train['time'].apply(time_to_seconds)
        # # ridge = Ridge(alpha=2600)
        # lgbm = LGBMRegressor(n_estimators=20, max_depth=5, random_state=42, num_leaves=100,
        #              min_child_samples=10, min_child_weight=10, n_jobs=-1)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)


        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        df_train = pd.read_csv('./data/df_train_UK.csv')
        df_train.reset_index(inplace=True, drop=True)
        df_test=pd.read_csv('./data/df_test_UK.csv')
        label_encoder1 = LabelEncoder()
        label_encoder2 = LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_test['going'] = label_encoder1.fit_transform(df_test['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['race_type'] = label_encoder3.fit_transform(df_train['race_type'])
        df_test['type'] = label_encoder3.fit_transform(df_test['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        df_test['sex'] = label_encoder4.fit_transform(df_test['sex'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_test['course2'] = label_encoder5.fit_transform(df_test['course'])



        df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK(row, df_test, df_train),
            axis=1
)
        df_test.to_csv('./data/df_test_UK2.csv',index=False)
        df_test.reset_index(inplace=True, drop=True)
        df_cleaned = df_train.dropna(subset=['draw','age',
                            'dist_y','weight_lbs',
                             'prize','rpr','going','recent_ave_rank', 
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank'])
        df_train = df_cleaned.reset_index(drop=True)

        df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
        df_cleaned = df_test.dropna(subset=['draw','age',
                            'distance_f','lbs'
                             ,'prize','rpr','going','recent_ave_rank',  
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)
        X_train = df_train[['draw','age',
                            'weight_lbs',
                             'prize','rpr','going','race_type','sex','course2','recent_ave_rank',  
          'jockey_ave_rank',  
          'trainer_ave_rank', 
          'owner_ave_rank']]
        X_train.rename(columns={'dist_y': 'distance_f', 'weight_lbs': 'lbs','race_type':'type'}, inplace=True)

        X_test = df_test[['draw','age',
                           'lbs',
                             'prize','rpr','going','type','sex','course2','recent_ave_rank', 
          'jockey_ave_rank',  
          'trainer_ave_rank',
          'owner_ave_rank']]
        y_train = df_train['time'].apply(time_to_seconds)
        y_train = y_train.values.reshape(-1, 1) 

        # lgbm = LGBMRegressor(n_estimators=20, max_depth=5, random_state=42, num_leaves=100,
        #              min_child_samples=10, min_child_weight=10, n_jobs=-1)
        lgbm = LGBMRegressor()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        lgbm.fit(X_train, y_train)
        y_pred = lgbm.predict(X_test)  # Assuming X_test is defined
        df_test['predicted_time'] = y_pred
        df_test_sorted = df_test.sort_values(by=['race_id', 'predicted_time'])
        # Optionally, get the fastest horse per race
        df_fastest_per_race = df_test_sorted.groupby('race_id').head(1)
        df_fastest_per_race = df_fastest_per_race[['date','race_id','course','horse_name',  'race_name']]
        df_fastest_per_race = df_fastest_per_race.reset_index(drop=True)  # Reassign the result
        # Display the filtered DataFrame in Streamlit
        st.title("Fastest Horses in Each Race")
        st.dataframe(df_fastest_per_race)

    
    if st.button('PREDICT: GB WIN with Deep-Q'):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        df_train = pd.read_csv('./data/df_train_UK.csv')
        df_train.reset_index(inplace=True, drop=True)
        df_test=pd.read_csv('./data/df_test_UK.csv')

        if len(df_train['race_id'].unique()) >= 3300:
                # Take a random sample of 3,300 unique race IDs
                sampled_race_ids = df_train['race_id'].drop_duplicates().sample(n=3300, random_state=1)  # random_state for reproducibility
                
                # Create a new DataFrame with the sampled race IDs
                df_train = df_train[df_train['race_id'].isin(sampled_race_ids)].reset_index(drop=True)
                





        st.write("df_train with no nans dropped is",df_train)
        label_encoder1 = LabelEncoder()
        label_encoder2 = LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_test['going'] = label_encoder1.fit_transform(df_test['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['type'] = label_encoder3.fit_transform(df_train['type'])
        df_test['type'] = label_encoder3.fit_transform(df_test['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        df_test['sex'] = label_encoder4.fit_transform(df_test['sex'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_test['course2'] = label_encoder5.fit_transform(df_test['course'])
        #choose course to optimize for
        df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK(row, df_test, df_train),
            axis=1
)
        df_test.to_csv('./data/df_test_UK2.csv',index=False)
        df_test.reset_index(inplace=True, drop=True)
        df_cleaned = df_train.dropna(subset=['draw','age',
                            'dist_y','weight_lbs','prize',
                             'rpr','going','recent_ave_rank',  
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank'])
        df_train = df_cleaned.reset_index(drop=True)
        st.write("df_train with dropped nans is:",df_train)
        df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
        df_test['distance_f'] = df_test['distance_f'] * 201.168
        df_cleaned = df_test.dropna(subset=['draw','age',
                            'distance_f','lbs','prize'
                             ,'rpr','going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)
        X_train = df_train[['draw','age', 'type', 'sex','course2',
                            'weight_lbs','dist_m','prize',
                             'rpr','going','recent_ave_rank',  
          'jockey_ave_rank',  
          'trainer_ave_rank', 
          'owner_ave_rank']]
        X_test = df_test[['draw','age','type','sex','course2',
                            'lbs','distance_f','prize',
                             'rpr','going','recent_ave_rank', 
          'jockey_ave_rank',  
          'trainer_ave_rank',
          'owner_ave_rank']]
        X_test.rename(columns={
            'lbs': 'weight_lbs',       # Rename lbs to weight_lbs
            'distance_f': 'dist_m'     # Rename distance_f to dist_m
        }, inplace=True)
        y_train = df_train['time'].apply(time_to_seconds)
        y_train = y_train.values.reshape(-1, 1) 
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)  # Fit and transform the training data
        X_test_scaled=scaler_X.fit_transform(X_test)
        y_train_scaled=scaler_y.fit_transform(y_train)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train_scaled, test_size=0.2, random_state=42)

   
        import tensorflow as tf
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        import random
        from collections import deque
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.optimizers import Adam

        class DQNAgent:
            def __init__(self, state_size, action_size):
                self.state_size = state_size
                self.action_size = action_size
                self.memory = deque(maxlen=2000)
                self.gamma = 0.95  
                self.epsilon = 0.1
                self.epsilon_min = 0.01
                self.epsilon_decay = 0.995
                self.model = self._build_model()

            def _build_model(self):
                # Neural network for Q-value approximation
                model = Sequential()
                model.add(Dense(60, input_dim=self.state_size, activation='tanh'))
                model.add(Dense(60, activation='tanh'))
                model.add(Dense(self.action_size))  # Linear output for Q-values
                model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
                return model

            def remember(self, state, action, reward, next_state, done):
                self.memory.append((state, action, reward, next_state, done))

            def act(self, state):
                if np.random.rand() <= self.epsilon:
                    return random.randrange(self.action_size)  # Explore
                act_values = self.model.predict(state)  # Predict action values
                return np.argmax(act_values[0])  # Exploit

            def replay(self, batch_size):
                if len(self.memory) < batch_size:
                    return
                minibatch = random.sample(self.memory, batch_size)
                for state, action, reward, next_state, done in minibatch:
                    target = reward
                    if not done:
                        target = reward + self.gamma * np.max(self.model.predict(next_state)[0])
                    target_f = self.model.predict(state)
                    target_f[0][action] = target
                    self.model.fit(state, target_f, epochs=1, verbose=0)
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
    # Parameters
        state_size = X_train.shape[1]  # Number of features
        action_size = 1  # Predict finish time, so one action
        agent = DQNAgent(state_size, action_size)
        episodes = 1000  

        # Training the agent
        for e in range(episodes):
            state = X_train[np.random.randint(len(X_train))].reshape(1, -1)  # Initial state
            for time in range(200):  # You can adjust the number of steps
                action = agent.act(state)  # Get action
                # Here we assume you have a way to define your reward based on action
                reward = -abs(y_train[np.argmax(state)] - action)  # Example reward
                next_state = state  # For simplicity, next_state is the same in this context
                done = True if time == 199 else False  # End the episode if max steps reached

                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    print(f"Episode: {e+1}/{episodes}, score: {time}, epsilon: {agent.epsilon:.2}")
                    break

            agent.replay(32)  # Train the model based on the experience replay

       
        predictions = agent.model.predict(X_val)

        # Create a DataFrame to compare actual vs predicted values
        comparison = pd.DataFrame({'Actual': y_val, 'Predicted': predictions.flatten()})
        st.write(comparison)
        
        from sklearn.metrics import mean_absolute_error

        mae = mean_absolute_error(y_val, predictions)
        print(f'Mean Absolute Error: {mae}')


    if st.button('PREDICT: GB WIN with NEW HYPERPARAMETER'):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        df_train = pd.read_csv('./data/df_train_UK.csv')
        df_train.reset_index(inplace=True, drop=True)
        df_test=pd.read_csv('./data/df_test_UK.csv')
        st.write("df_train with no nans dropped is",df_train)
        label_encoder1 = LabelEncoder()
        label_encoder2 = LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_test['going'] = label_encoder1.fit_transform(df_test['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['race_type'] = label_encoder3.fit_transform(df_train['race_type'])
        df_test['type'] = label_encoder3.fit_transform(df_test['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        df_test['sex'] = label_encoder4.fit_transform(df_test['sex'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_test['course2'] = label_encoder5.fit_transform(df_test['course'])
        #choose course to optimize for
        df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK(row, df_test, df_train),
            axis=1
)
        df_test.to_csv('./data/df_test_UK2.csv',index=False)
        df_test.reset_index(inplace=True, drop=True)
        df_cleaned = df_train.dropna(subset=['draw','age',
                            'dist_y','weight_lbs','prize',
                             'rpr','going','recent_ave_rank',  
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank'])
        df_train = df_cleaned.reset_index(drop=True)
        st.write("df_train with dropped nans is:",df_train)
        df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
        df_test['distance_f'] = df_test['distance_f'] * 201.168
        df_cleaned = df_test.dropna(subset=['draw','age',
                            'distance_f','lbs','prize'
                             ,'rpr','going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)
        X_train = df_train[['draw','age', 'race_type', 'sex','course2',
                            'weight_lbs','dist_m',
                             'rpr','going','recent_ave_rank',  
          'jockey_ave_rank',  
          'trainer_ave_rank', 
          'owner_ave_rank']]
        X_test = df_test[['draw','age','type','sex','course2',
                            'lbs','distance_f',
                             'rpr','going','recent_ave_rank', 
          'jockey_ave_rank',  
          'trainer_ave_rank',
          'owner_ave_rank']]
        y_train = df_train['time'].apply(time_to_seconds)
        y_train = y_train.values.reshape(-1, 1) 
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)  # Fit and transform the training data
        X_test_scaled=scaler_X.fit_transform(X_test)
        y_train_scaled=scaler_y.fit_transform(y_train)


        
        import tensorflow as tf
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
       
        
        
        from keras.callbacks import Callback, EarlyStopping
        # Set parameters
        batch_size = 96
        dropout_rate = 0.2
        epochs = 1800
        your_input_shape=13
        your_output_neurons=1
        input_shape = (your_input_shape,)  # Replace with the actual input shape

        # Create a Sequential model
        model = Sequential()

        # Add first hidden layer with 96 neurons, tanh activation, and dropout
        model.add(Dense(96, activation='tanh', input_shape=input_shape))
        model.add(Dropout(dropout_rate))

        # Add second hidden layer with 96 neurons, tanh activation, and dropout
        model.add(Dense(96, activation='tanh'))
        model.add(Dropout(dropout_rate))

        # Add output layer (adjust the number of neurons and activation function based on your task)
        model.add(Dense(your_output_neurons))  # For regression ta

        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")
        model.compile(optimizer=Adam(learning_rate=0.0036),
              loss='mean_squared_error',
              metrics=['mae'])

        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)

        # Fit the model with both callbacks
        history = model.fit(X_train_scaled, y_train_scaled, epochs=1800, batch_size=96, validation_split=0.2,
                            callbacks=[streamlit_callback], verbose=0)
        epochs = range(1, len(history.history['loss']) + 1)

        st.title('Loss Curve Plot')
        # Plotting function
        def plot_loss(history):
            plt.figure(figsize=(10, 6))
            plt.scatter(epochs,history.history['loss'], label='Training Loss')
            plt.scatter(epochs,history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        # Display the plot
        st.pyplot(plot_loss(history))

        DL_pred = model.predict(X_test_scaled) #what should k-fold be?
        st.write("Model Predictions:")
        DL_pred_unscaled = scaler_y.inverse_transform(DL_pred)
        df_pred = pd.DataFrame()
        df_pred['Date'] = df_test['date']
        df_pred['Course Name']=df_test['course']
        df_pred['HorseID'] = df_test['horse_name']
        df_pred['RaceID'] = df_test['race_id']
        df_pred['Race Name'] = df_test['race_name']
        df_pred['Horse Number']=df_test['number']
        df_pred['finish_time'] = DL_pred_unscaled
        # df_pred_kempton = df_pred[df_pred['Course Name'] == 'Ayr']
        df_sorted = df_pred.sort_values(by=['RaceID', 'finish_time'])
        df_sorted['HorseWin'] = df_sorted.groupby('RaceID')['finish_time'].transform(lambda x: x == x.min()).astype(int)
        df_sorted['HorseRankTop3'] = df_sorted.groupby('RaceID')['finish_time'].rank(method='min', ascending=True) <= 3
        df_sorted['HorseRankTop3'] = df_sorted['HorseRankTop3'].astype(int)
        winning_horses = df_sorted[df_sorted['HorseWin'] == 1]
        filtered_winning_horses = winning_horses.reset_index(drop=True)
        # Display the filtered DataFrame in Streamlit
        st.title("Bet on the races below")
        st.dataframe(filtered_winning_horses)



    if st.button('PREDICT: GB WIN with Deep Learning'):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder


        from keras.utils import Sequence

        class DynamicBatchGenerator(Sequence):
            def __init__(self, X, y, race_ids):
                self.X = X
                self.y = y
                self.race_ids = race_ids
                self.indices = np.arange(len(X))
                self.race_index_map = self._create_race_index_map()

            def _create_race_index_map(self):
                """Create a mapping of race IDs to the indices of their runners."""
                race_index_map = {}
                for index, race_id in enumerate(self.race_ids):
                    if race_id not in race_index_map:
                        race_index_map[race_id] = []
                    race_index_map[race_id].append(index)
                return race_index_map

            def __len__(self):
                # Total number of batches based on the number of races
                return len(self.race_index_map)

            def __getitem__(self, index):
                # Get the race ID for the current batch
                race_id = list(self.race_index_map.keys())[index]
                runner_indices = self.race_index_map[race_id]
                
                # Get the corresponding data for this race
                X_batch = self.X[runner_indices]
                y_batch = self.y[runner_indices]
                
                return X_batch, y_batch
        df_train = pd.read_csv('./data/df_train_UK_results.csv')
        edge_weight_diff_df=pd.read_csv('./data/graph/edge_weight_differences_UK.csv')
        df_train.reset_index(inplace=True, drop=True)
        edge_weight_diff_df.reset_index(inplace=True, drop=True)
       
        edge_weight_diff_df['horse'] = edge_weight_diff_df['horse'].str.replace(r' \([A-Z]{3}\)', '', regex=True).str.strip()
        df_train['horse'] = df_train['horse'].str.replace(r' \([A-Z]{3}\)', '', regex=True).str.strip()
        df_train['sp_dec'] = pd.to_numeric(df_train['sp_dec'], errors='coerce')

        # Filter for rows where 'sp_dec' is greater than or equal to 7
        # df_train = df_train[df_train['sp_dec'] >= 10]


        # start_date = '2023-10-23'
        # end_date = '2024-10-23'

        # # Filter X_train and X_test for the date range
        # df_train = df_train[(df_train['date'] >= start_date) & (df_train['date'] <= end_date)]
        if len(df_train['race_id'].unique()) >= 3300:
            # Take a random sample of 3,300 unique race IDs
            sampled_race_ids = df_train['race_id'].drop_duplicates().sample(n=3300, random_state=1)  # random_state for reproducibility
            
            # Create a new DataFrame with the sampled race IDs
            df_train = df_train[df_train['race_id'].isin(sampled_race_ids)].reset_index(drop=True)
                
        
        df_train=pd.merge(df_train,edge_weight_diff_df,on='horse',how='left')
        df_test=pd.read_csv('./data/df_test_UK.csv')
        df_test = df_test.rename(columns={'horse_name': 'horse'})
        df_test['horse'] = df_test['horse'].str.strip()
        df_test=pd.merge(df_test,edge_weight_diff_df,on='horse',how='left')
        st.write(df_test)
        # st.write("df_train with no nans dropped is",df_train)
        label_encoder1 = LabelEncoder()
        label_encoder2 = LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_test['going'] = label_encoder1.transform(df_test['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['type'] = label_encoder3.fit_transform(df_train['type'])
        df_test['type'] = label_encoder3.transform(df_test['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        df_test['sex_code'] = label_encoder4.transform(df_test['sex_code'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_test['course2'] = label_encoder5.transform(df_test['course'])
        #choose course to optimize for
        # Assuming df_test and df_train are defined and contain the relevant columns:
        df_test = df_test.apply(lambda row: compute_average_ranks_all_UK_with_win_percentage(row, df_test, df_train), axis=1)
        df_test.to_csv('./data/df_test_UK2.csv',index=False)
        df_test.reset_index(inplace=True, drop=True)
        df_test.to_csv("./data/df_test_predict.csv", index=False)

        import ast
        selected_bookmaker = 'Bet365'

        # Function to extract decimal odds for the specified bookmaker
        def extract_decimal_odds_for_bookmaker(odds_string, bookmaker):
            try:
                # Convert the string representation of the list to an actual list
                odds_list = ast.literal_eval(odds_string)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing odds string: {e}")
                return np.nan
            
            # Extract the decimal odds for the specified bookmaker
            if isinstance(odds_list, list):
                matched_odds = []
                for odd in odds_list:
                    if odd.get('bookmaker') == bookmaker:
                        decimal_odds = pd.to_numeric(odd.get('decimal', np.nan), errors='coerce')
                        matched_odds.append(decimal_odds)
                
                # Return the first matched decimal odds or np.nan if none found
                return matched_odds[0] if matched_odds else np.nan
            
            return np.nan  # Return NaN if odds_list is not a list

        # Apply the function to the 'odds' column to create a new 'decimal_odds' column for the selected bookmaker
        # df_test['sp_dec'] = df_test['odds'].apply(lambda x: extract_decimal_odds_for_bookmaker(x, selected_bookmaker))


        df_cleaned = df_train.dropna(subset=['age','type','sex',
                            'dist_y','weight_lbs',
                             'going','recent_ave_rank',
          ])
        df_train = df_cleaned.reset_index(drop=True)

        
        # st.write("df_train with dropped nans is:",df_train)
        df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
       
        df_test['distance_f'] = df_test['distance_f'] * 201.168
        df_cleaned = df_test.dropna(subset=['age','type','sex',
                            'distance_f','lbs',
                             'going','recent_ave_rank',
         ])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)
        st.write("DF TEST IS",df_test)
        # date_to_filter = '2024-09-20'
        # df_filtered = df_test[df_test['date'] == date_to_filter]
        # df_filtered.to_csv("./data/df_predict.csv", index=False)  # Set index=False to avoid saving the index as a separate column

        # st.write("LENGTH OF DF TEST IS",len(df_filtered))
        # st.write(df_test['odds'].iloc[0]) 
        
        # df_test = df_test[df_test['decimal_odds'] >= 10]

        # Display the resulting DataFrame
        # st.write(df_test[['odds', 'decimal_odds']])
        X_train = df_train[['age', 'type', 'sex',
                            'weight_lbs','dist_m',
                             'going','recent_ave_rank',  
         'time']]
        # X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
        df_test = df_test.drop(columns=['sex'], errors='ignore')

        df_test = df_test.rename(columns={'lbs': 'weight_lbs', 'distance_f': 'dist_m','sex_code':'sex'})

# Now select the renamed columns
        X_test = df_test[[ 'age', 'type', 'sex', 'weight_lbs', 'dist_m', 
                  'going', 'recent_ave_rank']]
        
       
        st.write(X_test)
        def time_to_seconds(time_str):
            try:
                if pd.isna(time_str):
                    # If the time is NaN, return None
                    return None

                # Convert the value to string in case it's a float
                time_str = str(time_str)

                # Split the time string by ':'
                parts = time_str.split(':')
                
                # Handle cases where the string might not be in the correct format
                if len(parts) == 2:  # Format like "4:12.11"
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                elif len(parts) == 1:  # Format like "12.11" (no minutes part)
                    minutes = 0
                    seconds = float(parts[0])
                else:
                    raise ValueError(f"Unexpected time format: {time_str}")
                
                # Convert minutes and seconds to total seconds
                return minutes * 60 + seconds
            except (ValueError, IndexError) as e:
                # Handle cases where the time string is invalid
                print(f"Error converting time: {time_str}. Error: {e}")
                return None  # Return None in case of an error
        X_train['time'].replace(['-', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        X_train.dropna(subset=['time'], inplace=True)
        y_train = X_train['time'].apply(time_to_seconds)
        y_train = y_train.values.reshape(-1, 1) 

       

    
        # start_date = '2023-10-16'
        # end_date = '2024-10-16'

        # # Filter X_train and X_test for the date range
        # X_train_filtered = X_train[(X_train['date'] >= start_date) & (X_train['date'] <= end_date)]
        
        # X_train = X_train_filtered.drop(columns=['time','date'])
        X_train = X_train.drop(columns=['time'])

        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        # Now scale the cleaned data
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)

        indices = np.arange(X_train_scaled.shape[0])
        np.random.shuffle(indices)

        # Shuffle both X_train and y_data using the same indices
        X_train_scaled = X_train_scaled[indices]
        y_train_scaled = y_train_scaled[indices]

        # race_ids = df_train['race_id'].values  # Replace with your actual race ID column

        # X_train_split, X_val_split, y_train_split, y_val_split, race_ids_train, race_ids_val = train_test_split(
        # X_train_scaled, y_train_scaled, race_ids, test_size=0.2, random_state=42)

        # # Create generators for training and validation
        # train_generator = DynamicBatchGenerator(X_train_split, y_train_split, race_ids_train)
        # val_generator = DynamicBatchGenerator(X_val_split, y_val_split, race_ids_val)
      
        X_test_scaled=scaler_X.transform(X_test)
        import tensorflow as tf
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        # from hyperopt import hp
        # from hyperopt import Trials, STATUS_OK
        # space = {
        #     'num_layers': hp.quniform('num_layers', 2, 6, 1),  # Sample integers between 2 and 6
        #     'units': hp.quniform('units', 10, 200, 16),  # Sample units between 10 and 512 in steps of 16
        #     # 'dropout': hp.uniform('dropout', 0.2, 0.4),  # Dropout between 0.2 and 0.4
        #     'learning_rate': hp.loguniform('learning_rate', -6, -2),  # Log-uniform sample for learning rates
        #     'batch_size': hp.quniform('batch_size', 16, 100, 16),  # Sample batch sizes in multiples of 16
        #     'epochs': hp.choice('epochs', [500, 1000]) # Sample epochs in steps of 100
        # }
        # def objective(params):
        #     model = Sequential()
        #     # Adding input layer
        #     layers=int(params['num_layers'])
        #     units = int(params['units'])  # Convert to integer
        #     batch_size=int(params['batch_size'])
        #     epochs=int(params['epochs'])
        #     model.add(Dense(units, activation='tanh', input_shape=(13,)))
        #     # Adding hidden layers
        #     for _ in range(layers):
        #         model.add(Dense(units, activation='tanh'))
        #         # model.add(Dropout(params['dropout']))
        #     # Adding output layer
        #     model.add(Dense(1))  
        #     # Compile the model
        #     optimizer = Adam(learning_rate=params['learning_rate'])
        #     model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        #     # Train the model
        #     history = model.fit(X_train_scaled, y_train_scaled, 
        #                         epochs=epochs,
        #                         batch_size=batch_size,
        #                         validation_split=0.2,
        #                         verbose=0)
        #     val_mae = min(history.history['val_mae'])  # MAE for regression
        #     return {'loss': val_mae, 'status': STATUS_OK} 
        # from hyperopt import fmin, tpe
        # trials = Trials()
        # best = fmin(fn=objective,
        #             space=space,
        #             algo=tpe.suggest,
        #             max_evals=50,  # Number of trials
        #             trials=trials, verbose=True)
        # st.write("Best Hyperparameters:", best)
        
        log_dir = "./data/logs/fit/"  # Specify your log directory
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)    
        model = Sequential([
            Input(shape=(7,)),  
            Dense(60, activation='tanh'),  
            Dense(60, activation='tanh'),
            Dense(1)])  
    
        from keras.callbacks import Callback, EarlyStopping

        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder
            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")
        model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='mean_squared_error',
              metrics=['mae'])
        # from keras.callbacks import ReduceLROnPlateau

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)

        # Fit the model with both callbacks
        
        history = model.fit(X_train_scaled,y_train_scaled,
                    epochs=5000,
                    validation_split=0.2,  # Pass the validation generator here
                    batch_size=7,
                    callbacks=[streamlit_callback, tensorboard_callback, early_stopping],
                    verbose=0)
        epochs = range(1, len(history.history['loss']) + 1)

        tf.keras.utils.plot_model(model, to_file='./data/model.png', show_shapes=True, show_layer_names=True)

        st.title('Loss Curve Plot')
        # Plotting function
        def plot_loss(history):
            plt.figure(figsize=(10, 6))
            plt.scatter(epochs,history.history['loss'], label='Training Loss')
            plt.scatter(epochs,history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        # Display the plot
        st.pyplot(plot_loss(history))

        DL_pred = model.predict(X_test_scaled) #what should k-fold be?
        st.write("Model Predictions:")
        DL_pred_unscaled = scaler_y.inverse_transform(DL_pred)
        df_pred = pd.DataFrame()
        df_pred['Date'] = df_test['date']
        df_pred['Course Name']=df_test['course']
        df_pred['HorseID'] = df_test['horse']
        df_pred['RaceID'] = df_test['race_id']
        df_pred['Race Name'] = df_test['race_name']
        df_pred['Horse Number']=df_test['number']
        df_pred['finish_time'] = DL_pred_unscaled
        # df_pred_kempton = df_pred[df_pred['Course Name'] == 'Ayr']
        df_sorted = df_pred.sort_values(by=['RaceID', 'finish_time'])
        df_sorted['HorseWin'] = df_sorted.groupby('RaceID')['finish_time'].transform(lambda x: x == x.min()).astype(int)
        df_sorted['HorseRankTop3'] = df_sorted.groupby('RaceID')['finish_time'].rank(method='min', ascending=True) <= 3
        df_sorted['HorseRankTop3'] = df_sorted['HorseRankTop3'].astype(int)
        winning_horses = df_sorted[df_sorted['HorseWin'] == 1]
        filtered_winning_horses = winning_horses.reset_index(drop=True)
        # Display the filtered DataFrame in Streamlit
        st.title("Bet on the races below")
        st.dataframe(filtered_winning_horses)


   
       

    if st.button('PREDICT: GB WIN with PyTorch'):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        df_train = pd.read_csv('./data/df_train_UK.csv')
        df_train.reset_index(inplace=True, drop=True)
        df_test=pd.read_csv('./data/df_test_UK.csv')
        st.write("df_train with no nans dropped is",df_train)
        label_encoder1 = LabelEncoder()
        label_encoder2 = LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_test['going'] = label_encoder1.fit_transform(df_test['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['race_type'] = label_encoder3.fit_transform(df_train['race_type'])
        df_test['type'] = label_encoder3.fit_transform(df_test['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        df_test['sex'] = label_encoder4.fit_transform(df_test['sex'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_test['course2'] = label_encoder5.fit_transform(df_test['course'])
        #choose course to optimize for
        df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK(row, df_test, df_train),
            axis=1
)
        df_test.to_csv('./data/df_test_UK2.csv',index=False)
        df_test.reset_index(inplace=True, drop=True)
        df_cleaned = df_train.dropna(subset=['draw','age',
                            'dist_y','weight_lbs','prize',
                             'rpr','going','recent_ave_rank',  
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank'])
        df_train = df_cleaned.reset_index(drop=True)
        st.write("df_train with dropped nans is:",df_train)
        df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
        df_test['distance_f'] = df_test['distance_f'] * 201.168
        df_cleaned = df_test.dropna(subset=['draw','age',
                            'distance_f','lbs','prize'
                             ,'rpr','going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)
        X_train = df_train[['draw','age', 'race_type', 'sex','course2',
                            'weight_lbs','dist_m',
                             'rpr','going','recent_ave_rank',  
          'jockey_ave_rank',  
          'trainer_ave_rank', 
          'owner_ave_rank']]
        X_test = df_test[['draw','age','type','sex','course2',
                            'lbs','distance_f',
                             'rpr','going','recent_ave_rank', 
          'jockey_ave_rank',  
          'trainer_ave_rank',
          'owner_ave_rank']]
        y_train = df_train['time'].apply(time_to_seconds)
        y_train = y_train.values.reshape(-1, 1) 
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)  # Fit and transform the training data
        X_test_scaled=scaler_X.fit_transform(X_test)
        y_train_scaled=scaler_y.fit_transform(y_train)
        import tensorflow as tf
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        import torch
        import torch.nn as nn
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(13, 60)  # Input layer (13 features), 60 units
                self.fc2 = nn.Linear(60, 60)  # Hidden layer (60 units)
                self.fc3 = nn.Linear(60, 1)   # Output layer (1 unit)
                self.tanh = nn.Tanh()         # Tanh activation function
            def forward(self, x):
                x = self.tanh(self.fc1(x))
                x = self.tanh(self.fc2(x))
                x = self.fc3(x)
                return x

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()  # Mean Squared Error loss

        import streamlit as st

        # Custom early stopping
        class EarlyStopping:
            def __init__(self, patience=10, min_delta=0):
                self.patience = patience
                self.min_delta = min_delta
                self.best_loss = None
                self.counter = 0
            def should_stop(self, val_loss):
                if self.best_loss is None:
                    self.best_loss = val_loss
                    return False
                if val_loss < self.best_loss - self.min_delta:
                    self.best_loss = val_loss
                    self.counter = 0
                else:
                    self.counter += 1
                return self.counter >= self.patience

        # Placeholder in Streamlit
        placeholder = st.empty()

        # Instantiate early stopping
        early_stopping = EarlyStopping(patience=10)

        import torch
        from torch.utils.data import random_split, TensorDataset

        # Convert the data into PyTorch tensors
        X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

        # Combine the features and labels into a dataset
        dataset = TensorDataset(X_tensor, y_tensor)

        # Split the dataset into 80% training and 20% validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # DataLoader to automatically handle batching
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

        import torch
        import matplotlib.pyplot as plt
        import streamlit as st

        # Updated training function with loss tracking and Streamlit plotting
        def train(model, train_loader, val_loader, epochs=5000):
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                
                for X_batch, y_batch in train_loader:
                    # Forward pass
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                # Average training loss
                avg_train_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # Validation loss
                model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        val_outputs = model(X_val)
                        val_loss += criterion(val_outputs, y_val).item()

                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                # Log loss at the end of each epoch
                placeholder.write(f"Epoch {epoch + 1}: Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

                # Early stopping check
                if early_stopping.should_stop(avg_val_loss):
                    placeholder.write("Early stopping triggered!")
                    break

            # Plotting the losses
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss', color='blue')
            plt.plot(val_losses, label='Validation Loss', color='orange')
            plt.title('Loss over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()

            # Display the plot in Streamlit
            st.pyplot(plt)  # Show the plot in the Streamlit app

        # Call the training function
        train(model, train_loader, val_loader, epochs=5000)

        model.eval()
    
        # Convert the input data to a PyTorch tensor
        input_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

        # Perform the prediction
        with torch.no_grad():  # No need to track gradients during inference
            DL_pred = model(input_tensor)
            
            
        st.write("Model Predictions:")
        DL_pred_unscaled = scaler_y.inverse_transform(DL_pred)
        df_pred = pd.DataFrame()
        df_pred['Date'] = df_test['date']
        df_pred['Course Name']=df_test['course']
        df_pred['HorseID'] = df_test['horse_name']
        df_pred['RaceID'] = df_test['race_id']
        df_pred['Race Name'] = df_test['race_name']
        df_pred['Horse Number']=df_test['number']
        df_pred['finish_time'] = DL_pred_unscaled
        # df_pred_kempton = df_pred[df_pred['Course Name'] == 'Ayr']
        df_sorted = df_pred.sort_values(by=['RaceID', 'finish_time'])
        df_sorted['HorseWin'] = df_sorted.groupby('RaceID')['finish_time'].transform(lambda x: x == x.min()).astype(int)
        df_sorted['HorseRankTop3'] = df_sorted.groupby('RaceID')['finish_time'].rank(method='min', ascending=True) <= 3
        df_sorted['HorseRankTop3'] = df_sorted['HorseRankTop3'].astype(int)
        winning_horses = df_sorted[df_sorted['HorseWin'] == 1]
        filtered_winning_horses = winning_horses.reset_index(drop=True)
        # Display the filtered DataFrame in Streamlit
        st.title("Bet on the races below")
        st.dataframe(filtered_winning_horses)




    if st.button('PREDICT: GB WIN with DL from Scratch'):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        df_train = pd.read_csv('./data/df_train_UK.csv')
        df_train.reset_index(inplace=True, drop=True)
        df_test=pd.read_csv('./data/df_test_UK.csv')
        st.write("df_train with no nans dropped is",df_train)
        label_encoder1 = LabelEncoder()
        label_encoder2 = LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()
        label_encoder5=LabelEncoder()
        # Fit and transform the 'going' column
        df_train['going'] = label_encoder1.fit_transform(df_train['going'])
        df_test['going'] = label_encoder1.fit_transform(df_test['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['race_type'] = label_encoder3.fit_transform(df_train['race_type'])
        df_test['type'] = label_encoder3.fit_transform(df_test['type'])
        df_train['sex'] = label_encoder4.fit_transform(df_train['sex'])
        df_test['sex'] = label_encoder4.fit_transform(df_test['sex'])
        df_train['course2'] = label_encoder5.fit_transform(df_train['course'])
        df_test['course2'] = label_encoder5.fit_transform(df_test['course'])
        #choose course to optimize for
        df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK(row, df_test, df_train),
            axis=1
)
        df_test.to_csv('./data/df_test_UK2.csv',index=False)
        df_test.reset_index(inplace=True, drop=True)
        df_cleaned = df_train.dropna(subset=['draw','age',
                            'dist_y','weight_lbs','prize',
                             'rpr','going','recent_ave_rank',  
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank'])
        df_train = df_cleaned.reset_index(drop=True)
        st.write("df_train with dropped nans is:",df_train)
        df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
        df_test['distance_f'] = df_test['distance_f'] * 201.168
        df_cleaned = df_test.dropna(subset=['draw','age',
                            'distance_f','lbs','prize'
                             ,'rpr','going','recent_ave_rank',
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)
        X_train = df_train[['draw','age', 'race_type', 'sex','course2',
                            'weight_lbs','dist_m',
                             'rpr','going','recent_ave_rank',  
          'jockey_ave_rank',  
          'trainer_ave_rank', 
          'owner_ave_rank']]
        X_test = df_test[['draw','age','type','sex','course2',
                            'lbs','distance_f',
                             'rpr','going','recent_ave_rank', 
          'jockey_ave_rank',  
          'trainer_ave_rank',
          'owner_ave_rank']]
        y_train = df_train['time'].apply(time_to_seconds)
        y_train = y_train.values.reshape(-1, 1) 
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)  # Fit and transform the training data
        st.write("X train is:",X_train_scaled)
        X_test_scaled=scaler_X.fit_transform(X_test)
        y_train_scaled=scaler_y.fit_transform(y_train)
        import tensorflow as tf
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        import numpy as np
        from scipy.stats import truncnorm
     
        def truncated_normal(mean=0, sd=1, low=0, upp=10):
            return truncnorm(
                (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
        class NeuralNetwork:
            def __init__(self, 
                        no_of_in_nodes, 
                        no_of_out_nodes, 
                        no_of_hidden_nodes,
                        learning_rate,
                        bias=None):  
                self.no_of_in_nodes = no_of_in_nodes
                self.no_of_hidden_nodes = no_of_hidden_nodes
                self.no_of_out_nodes = no_of_out_nodes
                self.learning_rate = learning_rate 
                self.bias = bias
                self.create_weight_matrices()
            def create_weight_matrices(self):
                """ A method to initialize the weight matrices of the neural 
                network with optional bias nodes"""   
                bias_node = 1 if self.bias else 0 
                rad = 1 / np.sqrt(self.no_of_in_nodes + bias_node)
                X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
                self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, 
                                                self.no_of_in_nodes + bias_node))
                rad = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
                X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
                self.weights_hidden_out = X.rvs((self.no_of_out_nodes, 
                                                self.no_of_hidden_nodes + bias_node))
            def train(self, input_vector, target_vector):
                """ input_vector and target_vector can be tuple, list or ndarray """
                # make sure that the vectors have the right shap
                input_vector = np.array(input_vector)
                input_vector = input_vector.reshape(-1, 1)        
                if self.bias:
                    # adding bias node to the end of the input_vector
                    input_vector = np.concatenate( (input_vector, [[self.bias]]) )
                target_vector = np.array(target_vector).reshape(-1, 1)

                output_vector_hidden = np.tanh(self.weights_in_hidden @ input_vector)
                if self.bias:
                    output_vector_hidden = np.concatenate( (output_vector_hidden, [[self.bias]]) ) 
                output_vector_network = (self.weights_hidden_out @ output_vector_hidden)
                output_error = target_vector - output_vector_network  
                # calculate hidden errors:
                hidden_errors = self.weights_hidden_out.T @ output_error
                # update the weights:
                tmp = output_error * output_vector_network * (1.0 - output_vector_network)     
                self.weights_hidden_out += self.learning_rate  * (tmp @ output_vector_hidden.T)
                # update the weights:
                tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
                if self.bias:
                    x = (tmp @input_vector.T)[:-1,:]     # last row cut off,
                else:
                    x = tmp @ input_vector.T
                self.weights_in_hidden += self.learning_rate *  x 
            def run(self, input_vector):
                """
                running the network with an input vector 'input_vector'. 
                'input_vector' can be tuple, list or ndarray
                """
                # make sure that input_vector is a column vector:
                input_vector = np.array(input_vector)
                input_vector = input_vector.reshape(-1, 1)
                if self.bias:
                    # adding bias node to the end of the inpuy_vector
                    input_vector = np.concatenate( (input_vector, [[1]]) )
                input4hidden = np.tanh(self.weights_in_hidden @ input_vector)
                if self.bias:
                    input4hidden = np.concatenate( (input4hidden, [[1]]) )
                output_vector_network = np.tanh(self.weights_hidden_out @ input4hidden)
                return output_vector_network
            def evaluate(self, data, labels):
                total_error = 0
                for i in range(len(data)):
                    predicted = self.run(data[i,:])
                    true_value = labels[i,:]
                    
                    # Calculate the squared error
                    error = (predicted - true_value) ** 2
                    total_error += error[0, 0]  # Assuming predicted and true_value are 2D arrays
                    
                # Calculate Mean Squared Error
                mse = total_error / len(data)
                return mse
        

        simple_network = NeuralNetwork(no_of_in_nodes=13, 
                                    no_of_out_nodes=1, 
                                    no_of_hidden_nodes=2,
                                    learning_rate=0.0001,
                                    bias=1)
            
        st.write("length of Xtrain",len(X_train_scaled))

        st.write("shape of Xtrain",X_train_scaled.shape)
        for i in range(len(X_train_scaled)):
            simple_network.train(X_train_scaled[i,:], y_train_scaled[i,:])

            
        evaluation_result=simple_network.evaluate(X_train_scaled,y_train_scaled)

        st.write("Model Evaluation Result:", evaluation_result)



################################### TRASH FOR DEEP LEARNING ###########################
          # Define the model
        # from sklearn.model_selection import GridSearchCV
        # from scikeras.wrappers import KerasClassifier,KerasRegressor

        # def create_model(optimizer='adam'):
        #     model = Sequential()
        #     model.add(Input(shape=(14,))),
        #     model.add(Dense(60, activation='tanh')),
        #     model.add(Dense(60, activation='tanh')),
        #     model.add(Dense(1))  
        #     model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        #     return model

        # # Wrap the Keras model using KerasClassifier
        # model = KerasRegressor(build_fn=create_model, verbose=0)

        # from keras.callbacks import Callback, EarlyStopping

        # # Define the grid of hyperparameters to test
        # param_grid = {
        #     'batch_size': [10, 20, 40],
        #     'epochs': [10, 2000],
        #     'optimizer': ['adam', 'rmsprop']  # Specify the kernel initializer here
        # }

        # # Perform grid search
        # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,cv=3)
        # grid_result = grid.fit(X_train, y_train)
    ############################## END TRASH DEEP LEARNING ####################

     
     
    if st.button('PREDICT: GB WIN with CNN - RACECARDS'):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        from sklearn.preprocessing import LabelEncoder
        from keras.layers import Dense, Input
        from keras.optimizers import Adam
        from keras.optimizers import SGD
        from keras.layers import Dropout
        from keras.callbacks import Callback, EarlyStopping
        from keras.regularizers import l2
        from sklearn.preprocessing import StandardScaler
        import random
        import pandas as pd
        import networkx as nx
        import matplotlib.pyplot as plt
        from itertools import combinations
        from torch_geometric.data import Data
        df_results = pd.DataFrame(columns=['Model', 'Prediction', 'CV-F1', 'F1 Score', 'AUC', 'Recall', 'Precision'])
        # Get unique race IDs
        df_train=pd.read_csv('./data/df_train_UK_racecards.csv')
        df_train = df_train[df_train['date'] < '2024-07-01']

    
         # ################ GRAPH-BASED FEATURES #######################
       
        # edge_weight_diff_df=pd.read_csv('./data/graph/edge_weight_differences_UK.csv')
        df_test=pd.read_csv('./data/df_test_UK.csv')
        df_test = df_test.rename(columns={'horse_name': 'horse'})
        # edge_weight_diff_df['horse'] = edge_weight_diff_df['horse'].str.replace(r' \([A-Z]{3}\)', '', regex=True).str.strip()
        df_train['horse'] = df_train['horse'].str.replace(r' \([A-Z]{3}\)', '', regex=True).str.strip()
    
    ####################################################################################
        label_encoder = LabelEncoder()
        label_encoder2=LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()

        unique_classes = np.unique(df_train['going'].tolist() + df_test['going'].tolist())
        label_encoder = LabelEncoder()
        label_encoder.fit(unique_classes)  # Ensure all possible values are included

        # Fit and transform the 'going' column
        df_train = df_train[pd.to_numeric(df_train['age'], errors='coerce').notna()]

        df_train['going'] = label_encoder.transform(df_train['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['dam']=label_encoder2.fit_transform(df_train['dam'])
        df_train['sire']=label_encoder3.fit_transform(df_train['sire'])
        df_train['course2']=label_encoder4.fit_transform(df_train['course'])
        df_test['going'] = label_encoder.transform(df_test['going'])
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['tote_win']= df_train['tote_win'].replace('[£,]', '', regex=True).astype(float)
        # df_test['dam']=label_encoder2.transform(df_test['dam'])
        # df_test['sire']=label_encoder3.transform(df_test['sire'])
        df_test['course2']=label_encoder4.transform(df_test['course'])
        df_train.reset_index(inplace=True, drop=True)
        df_test.reset_index(inplace=True, drop=True)
        df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
        df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK(row, df_test, df_train),
            axis=1)
        df_test.to_csv('./data/df_test_UK2.csv',index=False)
        df_cleaned = df_train.dropna(subset=[  'age','distance_f','lbs',
                             'time','going','recent_ave_rank',  
          'jockey_ave_rank', 
          'trainer_ave_rank',
          'owner_ave_rank','draw','rpr'])
        df_train = df_cleaned.reset_index(drop=True)
        df_cleaned = df_test.dropna(subset=[  'age','distance_f','lbs',
                           'going','recent_ave_rank', 
          'jockey_ave_rank',  
          'trainer_ave_rank', 
          'owner_ave_rank','draw','rpr']) 
        df_test = df_cleaned.reset_index(drop=True)
       

        df_train_sorted = df_train.sort_values(by=['horse', 'date'], ascending=[True, False])

            # 2. Get the most recent `ovr_btn` for each horse by grouping by horse name
        last_ovr_btn = df_train_sorted.groupby('horse').first()['ovr_btn']
        df_test = df_test.merge(last_ovr_btn, on='horse', how='left', suffixes=('', '_last'))


        X_train = df_train[['horse','dam','sire','course','course2','race_name','race_id','position','age',
                            'distance_f','lbs',
                           'rpr','going','recent_ave_rank',  
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos','horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos','draw','ovr_btn']]
       
       
        X_test = df_test[[ 'horse','dam','sire','course','course2','race_name','race_id','age',
                            'distance_f','lbs',
                            'rpr','going','recent_ave_rank', 
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos','horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos','draw','ovr_btn']]
        
        
       
                

           

        from sklearn.impute import KNNImputer

          
        def time_to_seconds(time_str):
                try:
                    if pd.isna(time_str):
                        # If the time is NaN, return None
                        return None

                    # Convert the value to string in case it's a float
                    time_str = str(time_str)

                    # Split the time string by ':'
                    parts = time_str.split(':')
                    
                    # Handle cases where the string might not be in the correct format
                    if len(parts) == 2:  # Format like "4:12.11"
                        minutes = int(parts[0])
                        seconds = float(parts[1])
                    elif len(parts) == 1:  # Format like "12.11" (no minutes part)
                        minutes = 0
                        seconds = float(parts[0])
                    else:
                        raise ValueError(f"Unexpected time format: {time_str}")
                    
                    # Convert minutes and seconds to total seconds
                    return minutes * 60 + seconds
                except (ValueError, IndexError) as e:
                    # Handle cases where the time string is invalid
                    print(f"Error converting time: {time_str}. Error: {e}")
                    return None  # Return None in case of an error
        X_train['time'].replace(['-', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        X_train.dropna(subset=['time'], inplace=True)
          
        y_train = X_train['time'].apply(time_to_seconds)
        y_train = y_train.values.reshape(-1, 1) 
        
            


            # start_date = '2023-10-16'
            # end_date = '2024-10-16'

            # # Filter X_train and X_test for the date range
            # X_train_filtered = X_train[(X_train['date'] >= start_date) & (X_train['date'] <= end_date)]
            
            # X_train = X_train_filtered.drop(columns=['time','date'])
        X_train = X_train.drop(columns=['time'])

            


        from sklearn.impute import KNNImputer


        from sklearn.linear_model import LinearRegression
        import pandas as pd

            # X_train_features = X_train.drop(columns=['rpr'])  # The 11 features
            # y_train_rpr = X_train['rpr']  # Target column

            # # 2. Train a regression model
            # regressor = LinearRegression()
            # regressor.fit(X_train_features, y_train_rpr)

            
          







        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        # Now scale the cleaned data
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)

        import tensorflow as tf
        from keras.optimizers import SGD
        from keras import regularizers

        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        
        




        model = models.Sequential()

        # Add a Conv1D layer with filters and kernel size suitable for your 14 features
        model.add(layers.Conv1D(32, 3, activation='tanh', input_shape=(12, 1)))  # 32 filters, kernel size of 3
        model.add(layers.MaxPooling1D(2))  # Pooling to reduce dimensions
        model.add(layers.Conv1D(64, 3, activation='tanh'))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Flatten())  # Flatten to feed into Dense layers
        model.add(layers.Dense(64, activation='tanh'))  # Fully connected layer
        model.add(layers.Dense(1))  # Output layer for binary classification

        from keras.callbacks import Callback, EarlyStopping

        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")

        model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='mean_squared_error',
              metrics=['mae'])

        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)

        # Fit the model with both callbacks
        history = model.fit(X_train_scaled, y_train_scaled, epochs=5000, batch_size=32, validation_split=0.2,
                            callbacks=[streamlit_callback, early_stopping], verbose=0)
      
        epochs = range(1, len(history.history['loss']) + 1)


    
        st.title('Loss Curve Plot')

        # Plotting function
        def plot_loss(history):
            plt.figure(figsize=(10, 6))
            plt.scatter(epochs,history.history['loss'], label='Training Loss')
            plt.scatter(epochs,history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        # Display the plot
        st.pyplot(plot_loss(history))

        DL_pred = model.predict(X_test_scaled) #what should k-fold be?
        st.write("Model Predictions:")
        DL_pred_unscaled = scaler_y.inverse_transform(DL_pred)
        df_pred = pd.DataFrame()
        df_pred['Date'] = df_test['date']
        df_pred['Course Name']=df_test['course']
        df_pred['HorseID'] = df_test['horse_name']
        df_pred['RaceID'] = df_test['race_id']
        df_pred['Race Name'] = df_test['race_name']
        df_pred['Horse Number']=df_test['number']
        df_pred['finish_time'] = DL_pred_unscaled
        # df_pred_kempton = df_pred[df_pred['Course Name'] == 'Ayr']
        df_sorted = df_pred.sort_values(by=['RaceID', 'finish_time'])
        df_sorted['HorseWin'] = df_sorted.groupby('RaceID')['finish_time'].transform(lambda x: x == x.min()).astype(int)
        df_sorted['HorseRankTop3'] = df_sorted.groupby('RaceID')['finish_time'].rank(method='min', ascending=True) <= 3
        df_sorted['HorseRankTop3'] = df_sorted['HorseRankTop3'].astype(int)
        winning_horses = df_sorted[df_sorted['HorseWin'] == 1]
        filtered_winning_horses = winning_horses.reset_index(drop=True)
        # Display the filtered DataFrame in Streamlit
        st.title("Bet on the races below")
        st.dataframe(filtered_winning_horses)
    

    if st.button('PREDICT: GB WIN with Multi-Class Deep Learning - RACECARDS'):
        from sklearn.preprocessing import LabelEncoder
        from keras.optimizers import Adam

        selected_bookmaker = 'Bet365'

        # Function to extract decimal odds for the specified bookmaker
        def extract_decimal_odds_for_bookmaker(odds_string, bookmaker):
            try:
                # Convert the string representation of the list to an actual list
                odds_list = ast.literal_eval(odds_string)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing odds string: {e}")
                return np.nan
            
            # Extract the decimal odds for the specified bookmaker
            if isinstance(odds_list, list):
                matched_odds = []
                for odd in odds_list:
                    if odd.get('bookmaker') == bookmaker:
                        decimal_odds = pd.to_numeric(odd.get('decimal', np.nan), errors='coerce')
                        matched_odds.append(decimal_odds)
                
                # Return the first matched decimal odds or np.nan if none found
                return matched_odds[0] if matched_odds else np.nan
            
            return np.nan  # Return NaN if odds_list is not a list





        runs_df=pd.read_csv('./data/df_train_UK_racecards.csv')
        runs_df.reset_index(inplace=True, drop=True)
        df_test=pd.read_csv('./data/df_test_UK.csv')
        label_encoder1 = LabelEncoder()
        label_encoder2 = LabelEncoder()
        # Fit and transform the 'going' column
        runs_df['going'] = label_encoder1.fit_transform(runs_df['going'])
        df_test['going'] = label_encoder1.transform(df_test['going'])
        runs_df['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)

        runs_df['ovr_btn'] = pd.to_numeric(runs_df['ovr_btn'], errors='coerce')

        runs_df = runs_df[pd.to_numeric(runs_df['age'], errors='coerce').notna()]
        runs_df['tote_win']= runs_df['tote_win'].replace('[£,]', '', regex=True).astype(float)
        df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK(row, df_test, runs_df),
            axis=1
)      
        # df_test['sp_dec'] = df_test['odds'].apply(lambda x: extract_decimal_odds_for_bookmaker(x, selected_bookmaker))

        df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
        runs_df['draw'] = pd.to_numeric(runs_df['draw'], errors='coerce')
        runs_df = runs_df[(runs_df['draw'] >= 1) & (runs_df['draw'] <= 14)]
        runs_df = runs_df[runs_df['draw'].between(1, 14, inclusive='both')]
        df_test = df_test.rename(columns={'horse_name': 'horse'})
        df_test['horse'] = df_test['horse'].str.replace(r"\s*\(.*?\)", "", regex=True)
        runs_df = runs_df.rename(columns={'horse_name': 'horse'})
        runs_df['horse'] = runs_df['horse'].str.replace(r"\s*\(.*?\)", "", regex=True)
        runs_df_sorted =runs_df.sort_values(by=['horse', 'date'], ascending=[True, False])

        # 2. Get the most recent `ovr_btn` for each horse by grouping by horse name
        last_ovr_btn = runs_df_sorted.groupby('horse').first()['ovr_btn']

        df_test = df_test.merge(last_ovr_btn, on='horse', how='left', suffixes=('', '_last'))
        df_test['ovr_btn'] = pd.to_numeric(df_test['ovr_btn'], errors='coerce')




        # Check if any values are outside the expected range
        
        df_cleaned = runs_df.dropna(subset=['draw','age',
                            'distance_f','lbs',
                             'going','recent_ave_rank',  
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank','rpr'])
        runs_df = df_cleaned.reset_index(drop=True)
        runs_df= runs_df[['race_id','draw','age',
                            'distance_f','lbs',
                             'going','recent_ave_rank',  
          'jockey_ave_rank',  
          'trainer_ave_rank', 
          'owner_ave_rank','rpr','position']]
        runs_df[['draw', 'age', 'distance_f', 'lbs', 'going', 'recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank', 'owner_ave_rank','rpr']] = runs_df[['draw', 'age', 'distance_f', 'lbs',  'going', 'recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank', 'owner_ave_rank','rpr']].astype(float)
        columns = list(runs_df.columns)
        # Reorder columns putting 'race_id' and 'runners_post' at the beginning
        columns_reordered = ['race_id', 'draw'] + [col for col in columns if col not in ['race_id', 'draw']]
        # Reassign to DataFrame with reordered columns
        runs_df = runs_df[columns_reordered]
        runs_df.to_csv('./data/runs_df.csv')
        # Select the relevant features (exclude 'position' for now)
        features_df = runs_df[['race_id', 'draw', 'age', 'distance_f', 'lbs',  'going', 'recent_ave_rank',  
                            'jockey_ave_rank', 'trainer_ave_rank', 'owner_ave_rank','rpr']]
        # Pivot the features so each row corresponds to a race and each column is a feature for each horse
        features_df = features_df.pivot(index='race_id', columns='draw', values=features_df.columns[2:])
        st.write(features_df)
        # Now pivot the finishing positions (assuming 'position' contains finishing places)
        results_df = runs_df[['race_id', 'draw', 'position']].pivot(index='race_id', columns='draw', values='position')
        st.write(features_df.columns)
        current_columns = features_df.columns
        # Step 2: Extract unique features and horse numbers
        features = sorted(set([col[0] for col in current_columns]))  # First level (features)
        horse_numbers = sorted(set([col[1] for col in current_columns]))  # Second level (horse numbers)
        # Step 3: Rearrange columns to group features by horse number
        rearranged_columns = [(feature, horse) for horse in horse_numbers for feature in features]
        # Step 4: Reorder the DataFrame with the new column arrangement
        features_df = features_df[rearranged_columns]
        # Step 5: Verify the columns
        print(features_df.columns)
        # Merge features and results into a single DataFrame
        runs_df = pd.concat([features_df, results_df], axis=1)
        runs_df = runs_df.fillna(0)
        X = runs_df[runs_df.columns[:-14]]
        st.write("X is",X)
        ss = preprocessing.StandardScaler()
        X = pd.DataFrame(ss.fit_transform(X),columns = X.columns)
        y = runs_df[runs_df.columns[-14:]].applymap(lambda x: 1.0 if 0.5 < x < 1.5 else 0.0)
        print("X shape after preprocessing:", X.shape)
        print("y shape after preprocessing:", y.shape)
        import sklearn.model_selection as model_selection
        # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
        import tensorflow as tf
        # Define the model
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(200, activation='tanh', input_shape=(126,)),  # Hidden layer
            tf.keras.layers.Dense(100, activation='tanh'),
            tf.keras.layers.Dense(50, activation='tanh'),
            tf.keras.layers.Dense(14, activation='softmax')  # Output layer with softmax for multi-class classification
        ])
        from keras.callbacks import Callback, EarlyStopping
        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")

        model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',  # Categorical cross-entropy for multi-class classification
              metrics=['accuracy'])  # Change metrics to accuracy for classification

        import numpy as np

    
        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)

        # Fit the model with both callbacks
        history = model.fit(X, y, epochs=5000, batch_size=16, validation_split=0.2,
                            callbacks=[streamlit_callback, early_stopping], verbose=0)
        epochs = range(1, len(history.history['loss']) + 1)
            # for layer in model.layers:
            #     weights = layer.get_weights()
            #     st.write(f"Layer: {layer.name}")
            #     for weight in weights:
            #         st.write(weight)
            # Streamlit app
        # st.title('Loss Curve Plot')
            # Plotting function
        def plot_loss(history):
                plt.figure(figsize=(10, 6),dpi=150)

                plt.scatter(epochs, history.history['loss'], 
                label='Training Loss', 
                color='blue', 
                s=5)  # 's' controls the size of the dots

                # Scatter plot for Validation Loss with smaller dots
                plt.scatter(epochs, history.history['val_loss'], 
                            label='Validation Loss', 
                            color='orange', 
                            s=5)  # Smaller dots
               
                plt.title('Model Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
            # Display the plot
        # st.pyplot(plot_loss(history))
        df_test['draw'] = pd.to_numeric(df_test['draw'], errors='coerce')
        df_test = df_test[(df_test['draw'] >= 1) & (df_test['draw'] <= 14)]
        df_test = df_test[df_test['draw'].between(1, 14, inclusive='both')]
        all_draws = set(range(1, 15))

        # Step 2: Get the unique draw positions that are present in your dataset
        unique_draws = set(df_test['draw'].unique())

        # Step 3: Identify missing draw positions
        missing_draws = all_draws - unique_draws
        st.write("missing draws are",missing_draws)
        df_cleaned = df_test.dropna(subset=['draw','age',
                            'distance_f','lbs',
                             'going','recent_ave_rank',  
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank','rpr'])
        df_test = df_cleaned.reset_index(drop=True)


################
        df_test[['draw', 'age', 'distance_f', 'lbs', 'going', 'recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank', 'owner_ave_rank','rpr']] = df_test[['draw', 'age', 'distance_f', 'lbs',  'going', 'recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank', 'owner_ave_rank','rpr']].astype(float)
##############



        features_df= df_test[['race_id','draw','age',
                            'distance_f','lbs',
                            'going','recent_ave_rank',  
          'jockey_ave_rank',  
          'trainer_ave_rank', 
          'owner_ave_rank','rpr']]

        st.write("\nUnique values in 'draw':", features_df['draw'].unique())
        st.write("\nUnique values in 'rpr':", features_df['rpr'].unique())
        st.write("\nUnique values in 'age':", features_df['age'].unique())
        st.write("\nUnique values in 'weight':", features_df['lbs'].unique())
        st.write("\nUnique values in 'distance':", features_df['distance_f'].unique())
        st.write("\nUnique values in 'recent_ave_rank':", features_df['recent_ave_rank'].unique())
        st.write("\nUnique values in 'jockey rank':", features_df['jockey_ave_rank'].unique())
        st.write("\nUnique values in 'owner rank':", features_df['owner_ave_rank'].unique())
        columns = list(features_df.columns)
        # Reorder columns putting 'race_id' and 'runners_post' at the beginning
        columns_reordered = ['race_id', 'draw'] + [col for col in columns if col not in ['race_id', 'draw']]
        # Reassign to DataFrame with reordered columns
        features_df = features_df[columns_reordered]
        features_df.to_csv('./data/df_test_multi.csv')

        ######################################
        ######################################
        ######################################
        required_draws = set(range(1, 15))  # Draw positions 1 through 14
        # Step 2: Create a list to store rows with missing draw positions filled with zeros
        missing_rows = []
        for race_id, group in features_df.groupby('race_id'):
            existing_draws = set(group['draw'].unique())
            missing_draws = required_draws - existing_draws  # Identify the missing draw positions
            # For each missing draw, create a row with zeros for all features (age, distance_f, etc.)
            for missing_draw in missing_draws:
                missing_row = {col: 0 for col in features_df.columns[2:]}  # Set features to 0
                missing_row['race_id'] = race_id
                missing_row['draw'] = missing_draw
                missing_rows.append(missing_row)
        # Step 3: Add the missing rows to the original DataFrame
        missing_df = pd.DataFrame(missing_rows)
        features_df = pd.concat([features_df, missing_df], ignore_index=True)
        ##################################
        ##################################
        ##################################






        features_df = features_df.pivot(index='race_id', columns='draw', values=features_df.columns[2:])
        current_columns = features_df.columns

        # Step 2: Extract unique features and horse numbers
        features = sorted(set([col[0] for col in current_columns]))  # First level (features)
        horse_numbers = sorted(set([col[1] for col in current_columns]))  # Second level (horse numbers)

        # Step 3: Rearrange columns to group features by horse number
        rearranged_columns = [(feature, horse) for horse in horse_numbers for feature in features]

        # Step 4: Reorder the DataFrame with the new column arrangement
        features_df = features_df[rearranged_columns]
        
        features_df = features_df.fillna(0)
        # st.write(features_df)








    
        ss = preprocessing.StandardScaler()
        X = pd.DataFrame(ss.fit_transform(features_df),columns = features_df.columns)
        st.write("X IS",X)
        probabilities = model.predict(X)
        max_prob_indices = np.argmax(probabilities, axis=1)
        indices_df = pd.DataFrame({'draw': max_prob_indices+1})
        st.title("Model Predictions")
        features_df = features_df.reset_index()
        predictions= pd.concat([features_df['race_id'], indices_df], axis=1)
        # st.write("features df is",features_df)
        filtered_df_test = df_test.merge(predictions, on=['race_id', 'draw'])
        # Displaying the filtered DataFrame
        final_df = filtered_df_test[['date','race_id', 'course', 'horse']]

        # Displaying the final DataFrame
        st.write(final_df)
    

    if st.button('PREDICT: GB WIN with Multi-Class Deep Learning'):
        from sklearn.preprocessing import LabelEncoder
        from keras.optimizers import Adam
        runs_df=pd.read_csv('./data/df_train_UK.csv')
        runs_df.reset_index(inplace=True, drop=True)
        df_test=pd.read_csv('./data/df_test_UK.csv')
        label_encoder1 = LabelEncoder()
        label_encoder2 = LabelEncoder()
        # Fit and transform the 'going' column
        runs_df['going'] = label_encoder1.fit_transform(runs_df['going'])
        df_test['going'] = label_encoder1.transform(df_test['going'])
        runs_df['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK(row, df_test, runs_df),
            axis=1
)  
        df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
        df_test['distance_f'] = df_test['distance_f'] * 201.168
        runs_df['draw'] = pd.to_numeric(runs_df['draw'], errors='coerce')
        runs_df = runs_df[(runs_df['draw'] >= 1) & (runs_df['draw'] <= 10)]
        runs_df = runs_df[runs_df['draw'].between(1, 10, inclusive='both')]
        # Check if any values are outside the expected range
        print("\nUnique values in 'draw':", runs_df['draw'].unique())
        df_cleaned = runs_df.dropna(subset=['draw','age',
                            'dist_y','weight_lbs',
                             'going','recent_ave_rank',  
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank'])
        runs_df = df_cleaned.reset_index(drop=True)
        runs_df= runs_df[['race_id','draw','age',
                            'dist_y','weight_lbs',
                             'going','recent_ave_rank',  
          'jockey_ave_rank',  
          'trainer_ave_rank', 
          'owner_ave_rank','position']]
        runs_df[['draw', 'age', 'dist_y', 'weight_lbs', 'going', 'recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank', 'owner_ave_rank']] = runs_df[['draw', 'age', 'dist_y', 'weight_lbs',  'going', 'recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank', 'owner_ave_rank']].astype(float)
        runs_df.rename(columns={'dist_y': 'distance_f', 'weight_lbs': 'lbs'}, inplace=True)
        columns = list(runs_df.columns)
        # Reorder columns putting 'race_id' and 'runners_post' at the beginning
        columns_reordered = ['race_id', 'draw'] + [col for col in columns if col not in ['race_id', 'draw']]
        # Reassign to DataFrame with reordered columns
        runs_df = runs_df[columns_reordered]
        runs_df.to_csv('./data/runs_df.csv')
        # Select the relevant features (exclude 'position' for now)
        features_df = runs_df[['race_id', 'draw', 'age', 'distance_f', 'lbs',  'going', 'recent_ave_rank',  
                            'jockey_ave_rank', 'trainer_ave_rank', 'owner_ave_rank']]
        # Pivot the features so each row corresponds to a race and each column is a feature for each horse
        features_df = features_df.pivot(index='race_id', columns='draw', values=features_df.columns[2:])
        st.write(features_df)
        # Now pivot the finishing positions (assuming 'position' contains finishing places)
        results_df = runs_df[['race_id', 'draw', 'position']].pivot(index='race_id', columns='draw', values='position')
        st.write(features_df.columns)
        current_columns = features_df.columns
        # Step 2: Extract unique features and horse numbers
        features = sorted(set([col[0] for col in current_columns]))  # First level (features)
        horse_numbers = sorted(set([col[1] for col in current_columns]))  # Second level (horse numbers)
        # Step 3: Rearrange columns to group features by horse number
        rearranged_columns = [(feature, horse) for horse in horse_numbers for feature in features]
        # Step 4: Reorder the DataFrame with the new column arrangement
        features_df = features_df[rearranged_columns]
        # Step 5: Verify the columns
        print(features_df.columns)
        # Merge features and results into a single DataFrame
        runs_df = pd.concat([features_df, results_df], axis=1)
        runs_df = runs_df.fillna(0)
        X = runs_df[runs_df.columns[:-10]]
        st.write("X is",X)
        ss = preprocessing.StandardScaler()
        X = pd.DataFrame(ss.fit_transform(X),columns = X.columns)
        y = runs_df[runs_df.columns[-10:]].applymap(lambda x: 1.0 if 0.5 < x < 1.5 else 0.0)
        print("X shape after preprocessing:", X.shape)
        print("y shape after preprocessing:", y.shape)
        import sklearn.model_selection as model_selection
        # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
        import tensorflow as tf
        # Define the model
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation='tanh', input_shape=(80,)),  # Hidden layer
            tf.keras.layers.Dense(50, activation='tanh'),
            tf.keras.layers.Dense(10, activation='softmax')  # Output layer with softmax for multi-class classification
        ])
        from keras.callbacks import Callback, EarlyStopping
        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")

        model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',  # Categorical cross-entropy for multi-class classification
              metrics=['accuracy'])  # Change metrics to accuracy for classification

        import numpy as np

    
        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)

        # Fit the model with both callbacks
        history = model.fit(X, y, epochs=5000, batch_size=16, validation_split=0.2,
                            callbacks=[streamlit_callback, early_stopping], verbose=0)
        epochs = range(1, len(history.history['loss']) + 1)
            # for layer in model.layers:
            #     weights = layer.get_weights()
            #     st.write(f"Layer: {layer.name}")
            #     for weight in weights:
            #         st.write(weight)
            # Streamlit app
        st.title('Loss Curve Plot')
            # Plotting function
        def plot_loss(history):
                plt.figure(figsize=(10, 6),dpi=150)

                plt.scatter(epochs, history.history['loss'], 
                label='Training Loss', 
                color='blue', 
                s=5)  # 's' controls the size of the dots

                # Scatter plot for Validation Loss with smaller dots
                plt.scatter(epochs, history.history['val_loss'], 
                            label='Validation Loss', 
                            color='orange', 
                            s=5)  # Smaller dots
               
                plt.title('Model Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
            # Display the plot
        st.pyplot(plot_loss(history))
        df_test['draw'] = pd.to_numeric(df_test['draw'], errors='coerce')
        df_test = df_test[(df_test['draw'] >= 1) & (df_test['draw'] <= 10)]
        df_test = df_test[df_test['draw'].between(1, 10, inclusive='both')]
        print(df_test['draw'].unique())
        df_cleaned = df_test.dropna(subset=['draw','age',
                            'distance_f','lbs',
                             'going','recent_ave_rank',  
          'jockey_ave_rank', 
          'trainer_ave_rank', 
          'owner_ave_rank'])
        df_test = df_cleaned.reset_index(drop=True)
      
        features_df= df_test[['race_id','draw','age',
                            'distance_f','lbs',
                            'going','recent_ave_rank',  
          'jockey_ave_rank',  
          'trainer_ave_rank', 
          'owner_ave_rank']]
        columns = list(features_df.columns)
        # Reorder columns putting 'race_id' and 'runners_post' at the beginning
        columns_reordered = ['race_id', 'draw'] + [col for col in columns if col not in ['race_id', 'draw']]
        # Reassign to DataFrame with reordered columns
        features_df = features_df[columns_reordered]
        features_df.to_csv('./data/df_test_multi.csv')
        features_df = features_df.pivot(index='race_id', columns='draw', values=features_df.columns[2:])
        current_columns = features_df.columns

        # Step 2: Extract unique features and horse numbers
        features = sorted(set([col[0] for col in current_columns]))  # First level (features)
        horse_numbers = sorted(set([col[1] for col in current_columns]))  # Second level (horse numbers)

        # Step 3: Rearrange columns to group features by horse number
        rearranged_columns = [(feature, horse) for horse in horse_numbers for feature in features]

        # Step 4: Reorder the DataFrame with the new column arrangement
        features_df = features_df[rearranged_columns]
        
        features_df = features_df.fillna(0)
        st.write(features_df)
    
        ss = preprocessing.StandardScaler()
        X = pd.DataFrame(ss.fit_transform(features_df),columns = features_df.columns)
        probabilities = model.predict(X)
        max_prob_indices = np.argmax(probabilities, axis=1)
        indices_df = pd.DataFrame({'draw': max_prob_indices+1})
        st.title("Model Predictions")
        features_df = features_df.reset_index()
        predictions= pd.concat([features_df['race_id'], indices_df], axis=1)
        # st.write("features df is",features_df)
        filtered_df_test = df_test.merge(predictions, on=['race_id', 'draw'])
        # Displaying the filtered DataFrame
        final_df = filtered_df_test[['date','race_id', 'course', 'horse_name']]

        # Displaying the final DataFrame
        st.write(final_df)


    if st.button('PREDICT: GB WIN with Pairs Deep Learning - RESULTS'):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        from sklearn.preprocessing import LabelEncoder
        from keras.layers import Dense, Input
        from keras.optimizers import Adam
        from keras.optimizers import SGD
        from keras.layers import Dropout
        from keras.callbacks import Callback, EarlyStopping
        from keras.regularizers import l2
        from sklearn.preprocessing import StandardScaler
        import random
        import pandas as pd
        import networkx as nx
        import matplotlib.pyplot as plt
        from itertools import combinations
        from torch_geometric.data import Data
        df_results = pd.DataFrame(columns=['Model', 'Prediction', 'CV-F1', 'F1 Score', 'AUC', 'Recall', 'Precision'])
        # Get unique race IDs
        df_train=pd.read_csv('./data/df_train_UK_results.csv')
    
         # ################ GRAPH-BASED FEATURES #######################
       
        # edge_weight_diff_df=pd.read_csv('./data/graph/edge_weight_differences_UK.csv')
        df_test=pd.read_csv('./data/df_test_UK.csv')
        df_test = df_test.rename(columns={'horse_name': 'horse'})
        # edge_weight_diff_df['horse'] = edge_weight_diff_df['horse'].str.replace(r' \([A-Z]{3}\)', '', regex=True).str.strip()
        df_train['horse'] = df_train['horse'].str.replace(r' \([A-Z]{3}\)', '', regex=True).str.strip()


        # def create_graph(df):
        #     G = nx.MultiDiGraph()  # Use MultiDiGraph for multiple edges
        #     for race_id, group in df.groupby('race_id'):
        #         # Sort horses by finish position
        #         sorted_horses = group.sort_values('position')['horse'].tolist()
        #         for i in range(len(sorted_horses)):
        #             for j in range(i + 1, len(sorted_horses)):
        #                 u, v = sorted_horses[i], sorted_horses[j]
        #                 # A beats B, directed edge from A to B
        #                 G.add_edge(u, v)
        #                 # Uncomment the next line if you want to keep track of race_id, or other attributes per edge
        #                 # G.add_edge(u, v, race_id=race_id)
        #     return G
        # def plot_graph(G, show_labels=False):
        #     plt.figure(figsize=(12, 10))
            
        #     # Choose a layout for the graph
        #     pos = nx.spring_layout(G, seed=42)  # Using spring layout for better readability
            
        #     # Draw the graph
        #     nx.draw(G, pos, with_labels=show_labels, node_color='red', font_weight='normal', node_size=10, edge_color='gray', arrows=True)
            
        #     # Optionally, draw edge labels if you want to show weights
        #     # if nx.get_edge_attributes(G, 'weight'):
        #     #     edge_labels = nx.get_edge_attributes(G, 'weight')
        #     #     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
        #     plt.title("Directed Graph of Horses by Race Finish Order")
        #     st.pyplot(plt)

        # G = create_graph(df_train)
        # # plot_graph(G, show_labels=False)  # Set `show_labels` to True if you want to display node labels
        # def calculate_degree_ratio(G):
        #     # Calculate in-degrees and out-degrees for each node (horse)
        #     in_degrees = dict(G.in_degree())  # In-degree for each node
        #     out_degrees = dict(G.out_degree())  # Out-degree for each node
            
        #     degree_data = []

        #     for node in G.nodes():
        #         in_deg = in_degrees.get(node, 0)
        #         out_deg = out_degrees.get(node, 0)
                
        #         total_deg = in_deg + out_deg  # Total degree (in-degree + out-degree)
                
        #         if total_deg > 0:
        #             degree_ratio = (in_deg - out_deg) / total_deg
        #         else:
        #             degree_ratio = 0  # Handle isolated nodes with no in/out edges
                
        #         degree_data.append({'horse': node, 'degree_ratio': degree_ratio})
            
        #     # Convert the list of dictionaries to a pandas DataFrame
        #     degree_df = pd.DataFrame(degree_data)
        #     return degree_df

        # # Calculate the degree ratios and get them in a DataFrame
        # degree_df = calculate_degree_ratio(G)

        # # Display the DataFrame with runners_horse_name and degree_ratio
        # print(degree_df)

        # # Save the DataFrame to a CSV file
        # csv_file_path = "./data/graph/horse_degree_ratios_UK.csv"
        # degree_df.to_csv(csv_file_path, index=False)
        # print(f"Degree ratios saved to {csv_file_path}")

        # def calculate_win_loss_spread(G):
        #     win_loss_spreads = []

        #     # Iterate through each pair of nodes with edges between them
        #     for u, v in G.edges():
        #         # Get the number of wins from u to v (how many times u beat v)
        #         u_beats_v = G.number_of_edges(u, v)
        #         # Get the number of wins from v to u (how many times v beat u)
        #         v_beats_u = G.number_of_edges(v, u)
                
        #         # Calculate the win-loss spread (wins - losses)
        #         win_loss_spread = u_beats_v - v_beats_u
                
        #         # Store the result in a list
        #         win_loss_spreads.append({
        #             'horse_1': u,
        #             'horse_2': v,
        #             'horse_1_wins': u_beats_v,
        #             'horse_2_wins': v_beats_u,
        #             'win_loss_spread': win_loss_spread
        #         })

        #     # Convert the result to a DataFrame for easy viewing
        #     win_loss_df = pd.DataFrame(win_loss_spreads).drop_duplicates(subset=['horse_1', 'horse_2'])
            
        #     return win_loss_df

        # # Calculate the win-loss spread for the graph G
        # win_loss_df = calculate_win_loss_spread(G)
        
        # # Display the win-loss spread DataFrame
        # print(win_loss_df)

        # # Save the DataFrame to a CSV file
        # csv_file_path = "./data/graph/horse_win_loss_spread_UK.csv"
        # win_loss_df.to_csv(csv_file_path, index=False)
        # print(f"Win-Loss spread saved to {csv_file_path}")

        # def calculate_edge_weight_difference(G):
        #     edge_weight_diff = []
            
        #     # Get the number of wins and losses for each node
        #     for node in G.nodes():
        #         # Sum weights of incoming edges
        #         in_weights = sum([G.number_of_edges(u, node) for u in G.predecessors(node)])
        #         # Sum weights of outgoing edges
        #         out_weights = sum([G.number_of_edges(node, v) for v in G.successors(node)])
                
        #         # Calculate the edge weight difference
        #         weight_diff = (out_weights - in_weights)/(out_weights + in_weights)
                
        #         edge_weight_diff.append({
        #             'horse': node,
        #             'in_weights': in_weights,
        #             'out_weights': out_weights,
        #             'weight_difference': weight_diff
        #         })
            
        #     weight_diff_df = pd.DataFrame(edge_weight_diff)
        #     return weight_diff_df
        #     # Calculate the edge weight differences for the graph G
        # edge_weight_diff_df = calculate_edge_weight_difference(G)
        # # Display the DataFrame with node, in_weights, out_weights, and weight_difference
        # print(edge_weight_diff_df)
        # # Save the DataFrame to a CSV file
        # csv_file_path = "./data/graph/edge_weight_differences_UK.csv"
        # edge_weight_diff_df.to_csv(csv_file_path, index=False)
        # print(f"Edge weight differences saved to {csv_file_path}")
        
        # hubs, authorities = nx.hits(G)

      
        # hub_data = {
        #     'horse': list(hubs.keys()),
        #     'hub_score': list(hubs.values()),
        #     'authority_score': list(authorities.values())
        # }

        # hub_df = pd.DataFrame(hub_data)
        # hub_df.to_csv('./data/graph/hits_scores_UK.csv', index=False)
        # print("CSV file 'hits_scores.csv' created successfully.")
        
        # df_train.reset_index(inplace=True, drop=True)
        # df_train=pd.merge(df_train,degree_df,on='horse',how='left')
        # df_train=pd.merge(df_train,edge_weight_diff_df,on='horse',how='left')
        # df_train=pd.merge(df_train,hub_df,on='horse',how='left')

        # df_test=pd.merge(df_test,degree_df,on='horse',how='left')
        # df_test=pd.merge(df_test,edge_weight_diff_df,on='horse',how='left')
        # df_test=pd.merge(df_test,hub_df,on='horse',how='left')
    ####################################################################################
        label_encoder = LabelEncoder()
        label_encoder2=LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()

        unique_classes = np.unique(df_train['going'].tolist() + df_test['going'].tolist())
        label_encoder = LabelEncoder()
        label_encoder.fit(unique_classes)  # Ensure all possible values are included

        # Fit and transform the 'going' column
        df_train['going'] = label_encoder.transform(df_train['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['dam']=label_encoder2.fit_transform(df_train['dam'])
        df_train['sire']=label_encoder3.fit_transform(df_train['sire'])
        df_train['course2']=label_encoder4.fit_transform(df_train['course'])
        df_test['going'] = label_encoder.transform(df_test['going'])
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        # df_test['dam']=label_encoder2.transform(df_test['dam'])
        # df_test['sire']=label_encoder3.transform(df_test['sire'])
        df_test['course2']=label_encoder4.transform(df_test['course'])
        df_test['distance_f'] = df_test['distance_f'] * 201.168
        df_train.reset_index(inplace=True, drop=True)
        df_test.reset_index(inplace=True, drop=True)
        df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
        df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK(row, df_test, df_train),
            axis=1)
        df_test.to_csv('./data/df_test_UK2.csv',index=False)
        df_cleaned = df_train.dropna(subset=[  'age','dist_m','weight_lbs',
                             'time','going','recent_ave_rank',  
          'jockey_ave_rank', 
          'trainer_ave_rank',
          'owner_ave_rank','draw','ovr_btn'])
        df_train = df_cleaned.reset_index(drop=True)
        df_cleaned = df_test.dropna(subset=[  'age','distance_f','lbs',
                           'going','recent_ave_rank', 
          'jockey_ave_rank',  
          'trainer_ave_rank', 
          'owner_ave_rank','draw']) 
        df_test = df_cleaned.reset_index(drop=True)
        df_test = df_test.rename(columns={
            'distance_f': 'dist_m',
            'lbs': 'weight_lbs'
        })

        df_train_sorted = df_train.sort_values(by=['horse', 'date'], ascending=[True, False])

            # 2. Get the most recent `ovr_btn` for each horse by grouping by horse name
        last_ovr_btn = df_train_sorted.groupby('horse').first()['ovr_btn']
        df_test = df_test.merge(last_ovr_btn, on='horse', how='left', suffixes=('', '_last'))


        X_train = df_train[['horse','dam','sire','course','course2','race_name','race_id','position','age',
                            'dist_m','weight_lbs',
                           'rpr','going','recent_ave_rank',  
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos','horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos','draw','ovr_btn']]
       
       
        X_test = df_test[[ 'horse','dam','sire','course','course2','race_name','race_id','age',
                            'dist_m','weight_lbs',
                            'rpr','going','recent_ave_rank', 
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos','horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos','draw','ovr_btn']]
        
        
        
        
        
        def create_pairwise_data_by_race(df):
            from itertools import combinations
            import random
            print("processing data into pairs")
            pairs = []
            features = ['draw', 'age', 'dist_m', 'weight_lbs',  'going', 'recent_ave_rank',
                        'jockey_ave_rank',
                        'trainer_ave_rank',
                        'owner_ave_rank','ovr_btn']
            # Sort data by race_id and position (for consistency)
            df = df.sort_values(by=['race_id', 'position'])

            for race_id, race_df in df.groupby('race_id'):
                if len(race_df) < 2:
                    continue  

                # Shuffle horse indices to randomize the order of comparisons
                horse_indices = list(race_df.index)
                random.shuffle(horse_indices)

                for horse1_idx, horse2_idx in combinations(horse_indices, 2):
                    horse1_features = df.loc[horse1_idx, features].values
                    horse2_features = df.loc[horse2_idx, features].values
                    pair_features = list(horse1_features) + list(horse2_features)

                    # Compare positions
                    pos1 = df.loc[horse1_idx, 'position']
                    pos2 = df.loc[horse2_idx, 'position']

                    # Result is 1 if horse1 finished ahead of horse2, else 0
                    result = 1 if pos1 < pos2 else 0
                    
                    # Get horse IDs
                    horse1_id = df.loc[horse1_idx, 'horse']  # Adjust this line based on your DataFrame
                    horse2_id = df.loc[horse2_idx, 'horse']  # Adjust this line based on your DataFrame
                    course=df.loc[horse1_idx, 'course']
                    # if course != "Newcastle" and course != "Newcastle (AW)":
                    #     continue
                    race_name=df.loc[horse1_idx, 'race_name']
                    # Include both horse IDs and race_id with the pair
                    pairs.append((pair_features, result, race_id,course,race_name, horse1_id, horse2_id))

            return pairs
        
        def create_pairwise_data_by_race_test(df):
            from itertools import combinations
            import random
            print("processing data into pairs")
            pairs = []
            features = ['draw', 'age', 'dist_m', 'weight_lbs',  'going', 'recent_ave_rank',
                        'jockey_ave_rank',
                        'trainer_ave_rank',
                        'owner_ave_rank','ovr_btn']

            # Sort data by race_id (for consistency)
            df = df.sort_values(by=['race_id'])

            for race_id, race_df in df.groupby('race_id'):
                if len(race_df) < 2:
                    continue  # Skip races with fewer than 2 horses

                # Shuffle horse indices to randomize the order of comparisons
                horse_indices = list(race_df.index)
                random.shuffle(horse_indices)

                for horse1_idx, horse2_idx in combinations(horse_indices, 2):
                    horse1_features = df.loc[horse1_idx, features].values
                    horse2_features = df.loc[horse2_idx, features].values
                    pair_features = list(horse1_features) + list(horse2_features)

                    # Get horse IDs and race details
                    horse1_id = df.loc[horse1_idx, 'horse']
                    horse2_id = df.loc[horse2_idx, 'horse']
                    course = df.loc[horse1_idx, 'course']
                    # if course != "Newcastle (AW)":
                    #     continue
                    race_name = df.loc[horse1_idx, 'race_name']

                    # Append the pair (no result)
                    pairs.append((pair_features, race_id, course, race_name, horse1_id, horse2_id))

            return pairs
                
        pairwise_data_train = create_pairwise_data_by_race(X_train)
        pairwise_data_test = create_pairwise_data_by_race_test(X_test)
        print("pairwise data is :", pairwise_data_train)

       
        X = np.array([pair[0] for pair in pairwise_data_train])  # Input features (horse1 vs horse2)
        y = np.array([pair[1] for pair in pairwise_data_train])  # Target labels (1 for win, 0 for loss)
                # Ensure X_train and X_test are 2D arrays (with shape (n_samples, n_features))
        y=y.reshape(-1,1)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X)  # Fit and transform the training data
        y_train_scaled = scaler_y.fit_transform(y)
        # print(X_train_scaled[:50])  
        # print(y_train_scaled[:50])  

        import tensorflow as tf
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
      
        # Define the model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(22,)),  # Specify input shape here
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
            

        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")
        model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',  #
              metrics=['accuracy']) 

        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)

        # Fit the model with both callbacks
        history = model.fit(X_train_scaled, y, epochs=5000, batch_size=100, validation_split=0.2,
                            callbacks=[streamlit_callback, early_stopping], verbose=0)
        epochs = range(1, len(history.history['loss']) + 1)

    
        X2 = np.array([pair[0] for pair in pairwise_data_test])  # Input features (horse1 vs horse2)
        print("Size of X2:", X2.shape)
        X_test_scaled = scaler_x.transform(X2)
       
       # Streamlit app
        st.title('Loss Curve Plot')

        # Plotting function
        def plot_loss(history):
            plt.figure(figsize=(10, 6))
            plt.scatter(epochs,history.history['loss'], label='Training Loss')
            plt.scatter(epochs,history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        # Display the plot
        st.pyplot(plot_loss(history))
        DL_pred = model.predict(X_test_scaled) 
        # Convert to binary labels
        pairwise_predictions = (DL_pred > 0.5).astype(int)


        from collections import defaultdict

        # Initialize a dictionary to hold the scores for each horse in each race
        horse_scores = defaultdict(lambda: defaultdict(int))

        # Loop through the pairwise data and predictions to tally wins for each horse
        for pred, (pair_features, race_id, course, race_name, horse1_id, horse2_id) in zip(pairwise_predictions, pairwise_data_test):
            if pred == 1:
                # If horse1 is predicted to beat horse2, increment horse1's score
                horse_scores[race_id][horse1_id] += 1
            else:
                # Otherwise, increment horse2's score
                horse_scores[race_id][horse2_id] += 1

        # Now horse_scores contains the number of wins for each horse in each race


           #Create a DataFrame to hold the predicted winners and their horse IDs
        predicted_winners_dict = {
            'race_id': [],
            'predicted_winner': []
        }

        # Iterate over horse_scores to determine the predicted winners
        for race_id, scores in horse_scores.items():
            if scores:  # Ensure scores is not empty
                predicted_winner = max(scores, key=scores.get)
                # Append details to the dictionary
                predicted_winners_dict['race_id'].append(race_id)
                predicted_winners_dict['predicted_winner'].append(predicted_winner)

        # Convert to DataFrame
        predicted_winners_df = pd.DataFrame(predicted_winners_dict)
        print(df_test.columns)
        predicted_winners_df =predicted_winners_df.rename(columns={'predicted_winner': 'horse'})
        merged_df = predicted_winners_df.merge(df_test[['date','race_id','course','horse', 'race_name']],
                                        on=['race_id', 'horse'],
                                        how='left')
        duplicates = df_test[df_test.duplicated(['race_id'], keep=False)]
        print(duplicates)
        # Display the DataFrame in Streamlit
        st.write(merged_df)
        
                



    if st.button('PREDICT: GB WIN with Pairs Deep Learning - RACECARDS'):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        from sklearn.preprocessing import LabelEncoder
        from keras.layers import Dense, Input
        from keras.optimizers import Adam
        from keras.optimizers import SGD
        from keras.layers import Dropout
        from keras.callbacks import Callback, EarlyStopping
        from keras.regularizers import l2
        from sklearn.preprocessing import StandardScaler
        import random
        import pandas as pd
        import networkx as nx
        import matplotlib.pyplot as plt
        from itertools import combinations
        from torch_geometric.data import Data
        df_results = pd.DataFrame(columns=['Model', 'Prediction', 'CV-F1', 'F1 Score', 'AUC', 'Recall', 'Precision'])
        # Get unique race IDs
        df_train=pd.read_csv('./data/df_train_UK_racecards.csv')


    
         # ################ GRAPH-BASED FEATURES #######################
       
        # edge_weight_diff_df=pd.read_csv('./data/graph/edge_weight_differences_UK.csv')
        df_test=pd.read_csv('./data/df_test_UK.csv')
        df_test = df_test.rename(columns={'horse_name': 'horse'})
        # edge_weight_diff_df['horse'] = edge_weight_diff_df['horse'].str.replace(r' \([A-Z]{3}\)', '', regex=True).str.strip()
        df_train['horse'] = df_train['horse'].str.replace(r' \([A-Z]{3}\)', '', regex=True).str.strip()
    
    ####################################################################################
        label_encoder = LabelEncoder()
        label_encoder2=LabelEncoder()
        label_encoder3=LabelEncoder()
        label_encoder4=LabelEncoder()

        unique_classes = np.unique(df_train['going'].tolist() + df_test['going'].tolist())
        label_encoder = LabelEncoder()
        label_encoder.fit(unique_classes)  # Ensure all possible values are included

        # Fit and transform the 'going' column
        df_train = df_train[pd.to_numeric(df_train['age'], errors='coerce').notna()]

        df_train['going'] = label_encoder.transform(df_train['going'])
        df_train['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['dam']=label_encoder2.fit_transform(df_train['dam'])
        df_train['sire']=label_encoder3.fit_transform(df_train['sire'])
        df_train['course2']=label_encoder4.fit_transform(df_train['course'])
        df_test['going'] = label_encoder.transform(df_test['going'])
        df_test['rpr'].replace(['-', '_', 'N/A', 'null', '–', '—'], np.nan, inplace=True)
        df_train['tote_win']= df_train['tote_win'].replace('[£,]', '', regex=True).astype(float)
        # df_test['dam']=label_encoder2.transform(df_test['dam'])
        # df_test['sire']=label_encoder3.transform(df_test['sire'])
        df_test['course2']=label_encoder4.transform(df_test['course'])
        df_train.reset_index(inplace=True, drop=True)
        df_test.reset_index(inplace=True, drop=True)
        df_test['prize'] = df_test['prize'].replace('[£,]', '', regex=True).astype(float)
        df_test[['recent_ave_rank', 'horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos', 
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos']] = df_test.apply(
            lambda row: compute_average_ranks_all_UK(row, df_test, df_train),
            axis=1)
        df_test.to_csv('./data/df_test_UK2.csv',index=False)
        df_cleaned = df_train.dropna(subset=[  'age','distance_f','lbs',
                             'time','going','recent_ave_rank',  
          'jockey_ave_rank', 
          'trainer_ave_rank',
          'owner_ave_rank','draw','rpr'])
        df_train = df_cleaned.reset_index(drop=True)
        df_cleaned = df_test.dropna(subset=[  'age','distance_f','lbs',
                           'going','recent_ave_rank', 
          'jockey_ave_rank',  
          'trainer_ave_rank', 
          'owner_ave_rank','draw','rpr']) 
        df_test = df_cleaned.reset_index(drop=True)
       

        df_train_sorted = df_train.sort_values(by=['horse', 'date'], ascending=[True, False])

            # 2. Get the most recent `ovr_btn` for each horse by grouping by horse name
        last_ovr_btn = df_train_sorted.groupby('horse').first()['ovr_btn']
        df_test = df_test.merge(last_ovr_btn, on='horse', how='left', suffixes=('', '_last'))


        X_train = df_train[['horse','dam','sire','course','course2','race_name','race_id','position','age',
                            'distance_f','lbs',
                           'rpr','going','recent_ave_rank',  
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos','horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos','draw','ovr_btn']]
       
       
        X_test = df_test[[ 'horse','dam','sire','course','course2','race_name','race_id','age',
                            'distance_f','lbs',
                            'rpr','going','recent_ave_rank', 
          'jockey_ave_rank', 'jockey_last_pos', 'jockey_second_last_pos', 'jockey_third_last_pos', 
          'trainer_ave_rank', 'trainer_last_pos', 'trainer_second_last_pos', 'trainer_third_last_pos', 
          'owner_ave_rank', 'owner_last_pos', 'owner_second_last_pos', 'owner_third_last_pos','horse_last_pos', 'horse_second_last_pos', 'horse_third_last_pos','draw','ovr_btn']]
        
        
        
        
        
        def create_pairwise_data_by_race(df):
            from itertools import combinations
            import random
            print("processing data into pairs")
            pairs = []
            features = ['draw', 'age', 'distance_f', 'lbs',  'going', 'recent_ave_rank',
                        'jockey_ave_rank',
                        'trainer_ave_rank',
                        'owner_ave_rank','rpr']
            # Sort data by race_id and position (for consistency)
            df = df.sort_values(by=['race_id', 'position'])

            for race_id, race_df in df.groupby('race_id'):
                if len(race_df) < 2:
                    continue  

                # Shuffle horse indices to randomize the order of comparisons
                horse_indices = list(race_df.index)
                random.shuffle(horse_indices)

                for horse1_idx, horse2_idx in combinations(horse_indices, 2):
                    horse1_features = df.loc[horse1_idx, features].values
                    horse2_features = df.loc[horse2_idx, features].values
                    pair_features = list(horse1_features) + list(horse2_features)

                    # Compare positions
                    pos1 = df.loc[horse1_idx, 'position']
                    pos2 = df.loc[horse2_idx, 'position']

                    # Result is 1 if horse1 finished ahead of horse2, else 0
                    result = 1 if pos1 < pos2 else 0
                    
                    # Get horse IDs
                    horse1_id = df.loc[horse1_idx, 'horse']  # Adjust this line based on your DataFrame
                    horse2_id = df.loc[horse2_idx, 'horse']  # Adjust this line based on your DataFrame
                    course=df.loc[horse1_idx, 'course']
                    # if course != "Newcastle" and course != "Newcastle (AW)":
                    #     continue
                    race_name=df.loc[horse1_idx, 'race_name']
                    # Include both horse IDs and race_id with the pair
                    pairs.append((pair_features, result, race_id,course,race_name, horse1_id, horse2_id))

            return pairs
        
        def create_pairwise_data_by_race_test(df):
            from itertools import combinations
            import random
            print("processing data into pairs")
            pairs = []
            features = ['draw', 'age', 'distance_f', 'lbs',  'going', 'recent_ave_rank',
                        'jockey_ave_rank',
                        'trainer_ave_rank',
                        'owner_ave_rank','rpr']

            # Sort data by race_id (for consistency)
            df = df.sort_values(by=['race_id'])

            for race_id, race_df in df.groupby('race_id'):
                if len(race_df) < 2:
                    continue  # Skip races with fewer than 2 horses

                # Shuffle horse indices to randomize the order of comparisons
                horse_indices = list(race_df.index)
                random.shuffle(horse_indices)

                for horse1_idx, horse2_idx in combinations(horse_indices, 2):
                    horse1_features = df.loc[horse1_idx, features].values
                    horse2_features = df.loc[horse2_idx, features].values
                    pair_features = list(horse1_features) + list(horse2_features)


                     
                  

                    # Get horse IDs and race details
                    horse1_id = df.loc[horse1_idx, 'horse']
                    horse2_id = df.loc[horse2_idx, 'horse']
                    course = df.loc[horse1_idx, 'course']
                    # if course != "Newcastle (AW)":
                    #     continue
                    race_name = df.loc[horse1_idx, 'race_name']

                    # Append the pair (no result)
                    pairs.append((pair_features, race_id, course, race_name, horse1_id, horse2_id))

            return pairs
                
        pairwise_data_train = create_pairwise_data_by_race(X_train)
        pairwise_data_test = create_pairwise_data_by_race_test(X_test)
        print("pairwise data is :", pairwise_data_train)

       
        X = np.array([pair[0] for pair in pairwise_data_train])  # Input features (horse1 vs horse2)
        y = np.array([pair[1] for pair in pairwise_data_train])  # Target labels (1 for win, 0 for loss)
                # Ensure X_train and X_test are 2D arrays (with shape (n_samples, n_features))
        y=y.reshape(-1,1)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X)  # Fit and transform the training data
        y_train_scaled = scaler_y.fit_transform(y)
        # print(X_train_scaled[:50])  
        # print(y_train_scaled[:50])  

        import tensorflow as tf
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
      
        # Define the model
        model = Sequential([
            Dense(200, activation='relu', input_shape=(20,)),  # Specify input shape here
            Dense(200, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
            

        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")
        model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',  #
              metrics=['accuracy']) 

        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)

        # Fit the model with both callbacks
        history = model.fit(X_train_scaled, y, epochs=5000, batch_size=100, validation_split=0.2,
                            callbacks=[streamlit_callback, early_stopping], verbose=0)
        epochs = range(1, len(history.history['loss']) + 1)

    
        X2 = np.array([pair[0] for pair in pairwise_data_test])  # Input features (horse1 vs horse2)
        print("Size of X2:", X2.shape)
        X_test_scaled = scaler_x.transform(X2)
       
       # Streamlit app
        st.title('Loss Curve Plot')

        # Plotting function
        def plot_loss(history):
            plt.figure(figsize=(10, 6))
            plt.scatter(epochs,history.history['loss'], label='Training Loss')
            plt.scatter(epochs,history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        # Display the plot
        st.pyplot(plot_loss(history))
        DL_pred = model.predict(X_test_scaled) 
        # Convert to binary labels
        pairwise_predictions = (DL_pred > 0.5).astype(int)


        from collections import defaultdict

        # Initialize a dictionary to hold the scores for each horse in each race
        horse_scores = defaultdict(lambda: defaultdict(lambda: {"predicted_score": 0, "actual_result": 0}))


        # Loop through the pairwise data and predictions to tally wins for each horse
        for pred, (pair_features, race_id, course, race_name, horse1_id, horse2_id) in zip(pairwise_predictions, pairwise_data_test):
            if pred == 1:
                # If horse1 is predicted to beat horse2, increment horse1's predicted score
                horse_scores[race_id][horse1_id]["predicted_score"] += 1
            else:
                # Otherwise, increment horse2's predicted score
                horse_scores[race_id][horse2_id]["predicted_score"] += 1

     
        # Create a DataFrame to hold the predicted winners and their horse IDs
        predicted_winners_dict = {
            'race_id': [],
            'predicted_winner': []
            
        }

        # Iterate over horse_scores to determine the predicted and actual winners
        for race_id, scores in horse_scores.items():
            if scores:  # Ensure scores is not empty
                predicted_winner = max(scores, key=lambda horse: scores[horse]["predicted_score"])
                # Append details to the dictionary
                predicted_winners_dict['race_id'].append(race_id)
                predicted_winners_dict['predicted_winner'].append(predicted_winner)

        # Convert to DataFrame
        predicted_winners_df = pd.DataFrame(predicted_winners_dict)

        # Merge with additional test data
        predicted_winners_df = predicted_winners_df.rename(columns={'predicted_winner': 'horse'})
        merged_df = predicted_winners_df.merge(df_test[['date', 'race_id', 'course', 'horse', 'race_name']],
                                            on=['race_id', 'horse'],
                                            how='left')
        st.write(merged_df)
        # Display duplicate races for validation
        duplicates = df_test[df_test.duplicated(['race_id'], keep=False)]
        print(duplicates)
                
                

    st.subheader("PLOT")
    if st.button("PLOT: Time-Series Trajectories of Finishing Time"):
        # Load the data
        df_train = pd.read_csv('./data/df_train_UK.csv')
        df_train.reset_index(inplace=True, drop=True)
        # distance = st.number_input('Enter distance:', min_value=0.0, format="%.2f")
        # Filter data for races with a distance of 1408.0
        df_filtered = df_train[df_train['dist_m'] ==1408]
        
        # Map race IDs to integer values
        df_filtered['race_id_int'] = pd.factorize(df_filtered['race_id'])[0]
        
        # Sort the filtered DataFrame by 'horse_id' and 'race_id_int'
        df_filtered_sorted = df_filtered.sort_values(by=['horse_id', 'race_id_int'])
        df_filtered_sorted['time'] = df_filtered_sorted['time'].apply(time_to_seconds)
        df_filtered_sorted['winning_time_detail'] = df_filtered_sorted['winning_time_detail'].apply(winning_time_to_seconds)
        DF=df_filtered_sorted[['race_id_int','horse_id','time','winning_time_detail','dist_m']]
        DF.to_csv('./data/time_data.csv',index=False)
     
        data = DF.to_numpy()
        num_unique_races = df_filtered_sorted['race_id_int'].nunique()
        num_unique_horses = df_filtered_sorted['horse_id'].nunique()
        
        st.write(f"Number of unique races: {num_unique_races}")
        st.write(f"Number of unique horses: {num_unique_horses}")
        

        # Save NumPy array to file
        np.save('./data/time_data.npy', data)
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Iterate over each unique horse_id and plot their finishing times
        for horse_id in df_filtered_sorted['horse_id'].unique():
            horse_data = df_filtered_sorted[df_filtered_sorted['horse_id'] == horse_id]
            plt.plot(horse_data['race_id_int'], horse_data['time'], marker='o', linestyle='-', alpha=0.6)
        
        plt.xlabel('Race ID')
        plt.ylabel('Finishing Time (seconds)')
        plt.title('Finishing Times Across All Horses for Distance 1658 meters')
        plt.grid(True)
        
        # Add a legend to identify horses
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

        # Display the plot in Streamlit
        st.pyplot(plt)











       
######################## NORTH AMERICAN ##############################
################### UPDATE DATABASE ##################################
with col2:
    st.header("USA")
    st.subheader("UPDATE & PROCESS")
    if st.button('UPDATE: N.A. Database'):
        # start_date = datetime(2023, 7, 1)
        latest_date = session.query(func.max(Entry.date)).scalar()
        start_date=latest_date
        # print("Latest date in the database:", latest_date)
        end_date=datetime.now()
        import sqlalchemy as sa
        dates = get_dates(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        fetch_and_write_db(dates)
        database_url = 'sqlite:///north_american.db'
        # Define unique columns for each model
        entry_unique_columns = ['meet_id', 'track_name', 'race_key_race_number', 'date', 'runners_horse_name']
        result_unique_columns = ['meet_id', 'track_name', 'race_key_race_number', 'date', 'runners_horse_name']
        # Remove duplicates from each table
        remove_duplicates(database_url, Entry, entry_unique_columns)
        remove_duplicates(database_url, Result, result_unique_columns)
        remove_future_dates(database_url, Entry)
        remove_future_dates(database_url, Result)
    ##################  GET UPCOMING RACES #############################
    if st.button('GET: Upcoming N.A. Races'):
        # now = datetime.now()
        # now_str = now.strftime("%Y-%m-%d")
        # dates2 = get_dates(now_str, now_str)
        now = datetime.now()
        # Calculate the next two days
        # next_day_1=now
        next_day_1 = now+timedelta(days=1)
        next_day_2 = now + timedelta(days=2)
        # Convert to string format
        now_str = now.strftime("%Y-%m-%d")
        next_day_1_str = next_day_1.strftime("%Y-%m-%d")
        next_day_2_str = next_day_2.strftime("%Y-%m-%d")
    # Get dates for the two following days
        dates2 = get_dates(next_day_1_str, next_day_2_str)
        fetch_and_write_csv(dates2)
        entries_pred = pd.read_csv('./data/entries_pred.csv')
        results_pred=pd.read_csv('./data/results_pred.csv')
        df_test = pd.merge(entries_pred, results_pred, on=['meet_id', 'track_name', 'race_key_race_number', 'date', 'runners_horse_name'], how='outer')
        df_test['race_id'] = df_test['meet_id'] + '_' + df_test['race_key_race_number'].astype(str)
        from sklearn.preprocessing import LabelEncoder
        le_course_type = LabelEncoder()
        le_course_type_class = LabelEncoder()
        le_surface_description = LabelEncoder()
        le_track_condition = LabelEncoder()
        le_weather=LabelEncoder()

        df_test['course_type'] = le_course_type.fit_transform(df_test['course_type'])
        df_test['course_type_class'] = le_course_type_class.fit_transform(df_test['course_type_class'])
        df_test['surface_description'] = le_surface_description.fit_transform(df_test['surface_description'])
        df_test['track_condition'] = le_track_condition.fit_transform(df_test['track_condition'])
        df_test['forecast'] = le_weather.fit_transform(df_test['forecast'])


        # st.session_state.df_test = df_test
        df_test.to_csv('./data/df_test.csv', index=False)
    #################  PROCESS DATA #####################################
    if st.button('PROCESS: N.A. Races'):
        races = query_and_merge_data()
        if races is not None:
            # st.write(races.head()
            print("processing")
        else:
            st.write("Error occurred while fetching and merging data.")
        #PROCESS TRAIN DATA
        df_horse = races[(races['finish_position'] >= 1) & (races['finish_position'] <= 14)].reset_index(drop=True)
        jockey = df_horse['jockey_full'].unique()
        numJockey = len(jockey)
        jockey_index = range(numJockey)
        trainer = df_horse['trainer_full'].unique()
        numTrainer = len(trainer)
        trainer_index = range(numTrainer)
        # Initialize columns
        df_horse['recent_6_runs'] = '0'
        df_horse['recent_ave_rank'] = '7'
        # Group by 'horse_name' and iterate over each group
        for horse_name, group in df_horse.groupby('runners_horse_name'):
            # Get recent ranks for the horse
            recent_ranks = group['finish_position'].tail(6)  # Get the last 6 ranks
            recent_ranks_list = recent_ranks.astype(str).tolist()  # Convert to list of strings

            # Assign recent ranks to 'recent_6_runs'
            df_horse.loc[group.index, 'recent_6_runs'] = '/'.join(recent_ranks_list)
            # Calculate average rank if recent_ranks is not empty
            if not recent_ranks.empty:
                recent_ave_rank = recent_ranks.mean()
                df_horse.loc[group.index, 'recent_ave_rank'] = recent_ave_rank
        # Convert 'recent_ave_rank' to float
        df_horse['recent_ave_rank'] = df_horse['recent_ave_rank'].astype(float)
        df_horse['finish_position'] = pd.to_numeric(df_horse['finish_position'], errors='coerce')
        # HorseWin
        df_horse['HorseWin'] = (df_horse['finish_position'] == 1).astype(int)
        df_horse['HorseSecond']=(df_horse['finish_position'] == 2).astype(int)
        # HorseRankTop3
        df_horse['HorseRankTop3'] = (df_horse['finish_position'].isin([1, 2, 3])).astype(int)
        df_horse['Win_Percentage'] = df_horse['recent_6_runs'].apply(calculate_win_percentage)
        # HorseRankTop50Percent
        top_finishes = df_horse.loc[df_horse['finish_position'] == 1].index
        top50_finishes = [idx + round((top_finishes[min(i+1, len(top_finishes)-1)] - idx) * 0.5) for i, idx in enumerate(top_finishes)]
        for i in top50_finishes:
            df_horse.loc[i:min(i+5, len(df_horse)-1), 'HorseRankTop50Percent'] = 1
        # Fill remaining NaN values with 0
        df_horse['HorseRankTop50Percent'].fillna(0, inplace=True)
        df_horse.to_csv('./data/df_horse.csv', index=False)
        df_horse.reset_index(drop=True, inplace=True)
        df_horse['date'] = pd.to_datetime(df_horse['date'])
        # Apply your function cal_ave_rank
        df_train = cal_ave_rank(df_horse)
        df_train.to_csv('./data/df_train.csv', index=False)
    st.subheader("BACKTEST")
    if st.button('BACKTEST: N.A. WIN with Logistic Regression'):
        df_results = pd.DataFrame(columns=['Model', 'Prediction', 'CV-F1', 'F1 Score', 'AUC', 'Recall', 'Precision'])
        # Get unique race IDs
        df_horse=pd.read_csv('./data/df_train.csv')

        # threshold = 15 # Set your threshold for high win_payoff
        # df_horse = df_horse[(df_horse['morning_line_odds'] > threshold)]
        unique_race_ids = df_horse['race_id'].unique()
        # Split the race IDs into training, test, and unseen sets
        train_race_ids, test_race_ids = train_test_split(unique_race_ids, test_size=0.2, random_state=42)
        test_race_ids, unseen_race_ids = train_test_split(test_race_ids, test_size=0.5, random_state=42)
        # Split the data based on the race IDs
        df_train_80= df_horse[df_horse['race_id'].isin(train_race_ids)]
        df_test_20 = df_horse[df_horse['race_id'].isin(test_race_ids)]
        df_unseen = df_horse[df_horse['race_id'].isin(unseen_race_ids)]

        df_train_80.to_csv('./data/df_train_80.csv', index=False)
        df_test_20.to_csv('./data/df_test_20.csv', index=False)
        df_unseen.to_csv('./data/df_unseen.csv', index=False)

        df_train_80.reset_index(inplace=True, drop=True)
        df_test_20.reset_index(inplace=True, drop=True)
        
        df_cleaned = df_train_80.dropna(subset=['distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        df_train_80 = df_cleaned.reset_index(drop=True)
        df_cleaned = df_test_20.dropna(subset=['distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        df_test_20 = df_cleaned.reset_index(drop=True)

        df_cleaned = df_unseen.dropna(subset=['distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        df_unseen= df_cleaned.reset_index(drop=True)
        X_train = df_train_80[['runners_weight', 'runners_post_pos',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
        X_test = df_test_20[['runners_weight', 'runners_post_pos',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
        X_test = X_test.rename(columns={'runners_post': 'runners_post_pos'})
        X_unseen=df_unseen[['runners_weight', 'runners_post_pos',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
        X_test = X_test.rename(columns={'runners_post': 'runners_post_pos'})
    
        y_train = df_train_80[['HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']]
        y_test=df_test_20[['HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']]


        y_unseen=df_unseen[['HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']]

        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)
        print("y_unseen shape:", y_unseen.shape)
    
        kfold = KFold(n_splits=5)
        
        lr_pred = run_model_lr(X_train, y_train, X_test, y_test, kfold)
        # rfc=run_model_rfc(X_train, y_train, X_test, y_test, kfold)
        class_pred_dict = {'Logistic Regression': lr_pred}
        strat1_results = pd.DataFrame(columns=['Model', 'Money', 'Bets Made'])
        strat2_results = pd.DataFrame(columns=['Model', 'Money', 'Bets Made'])
        print("lr_pred is:",lr_pred)
        for model_name, class_model in class_pred_dict.items():
                print(model_name)
                money, bets = simple_class_strategy(class_model,df_unseen, graph=True)
                money1,bets1=top3_strategy(class_model,df_unseen,graph=True)
                strat1_results.loc[len(strat1_results)] = [model_name, money, sum(bets)]
                strat2_results.loc[len(strat2_results)] = [model_name, money1, sum(bets1)]


    if st.button('BACKTEST: N.A. WIN with SMOTE+RF'):
        df_results = pd.DataFrame(columns=['Model', 'Prediction', 'CV-F1', 'F1 Score', 'AUC', 'Recall', 'Precision'])
        # Get unique race IDs
        df_horse=pd.read_csv('./data/df_train.csv')

        # threshold = 50  # Set your threshold for high win_payoff
        # df_horse = df_horse[(df_horse['win_payoff'] == 0) | (df_horse['win_payoff'] > threshold)]



        unique_race_ids = df_horse['race_id'].unique()
        # Split the race IDs into training, test, and unseen sets
        train_race_ids, test_race_ids = train_test_split(unique_race_ids, test_size=0.2, random_state=42)
        test_race_ids, unseen_race_ids = train_test_split(test_race_ids, test_size=0.5, random_state=42)
        # Split the data based on the race IDs
        df_train_80= df_horse[df_horse['race_id'].isin(train_race_ids)]
        df_test_20 = df_horse[df_horse['race_id'].isin(test_race_ids)]
        df_unseen = df_horse[df_horse['race_id'].isin(unseen_race_ids)]

        df_train_80.to_csv('./data/df_train_80.csv', index=False)
        df_test_20.to_csv('./data/df_test_20.csv', index=False)
        df_unseen.to_csv('./data/df_unseen.csv', index=False)

        df_train_80.reset_index(inplace=True, drop=True)
        df_test_20.reset_index(inplace=True, drop=True)
        
        df_cleaned = df_train_80.dropna(subset=['distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        df_train_80 = df_cleaned.reset_index(drop=True)
        df_cleaned = df_test_20.dropna(subset=['distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        df_test_20 = df_cleaned.reset_index(drop=True)

        df_cleaned = df_unseen.dropna(subset=['distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        df_unseen= df_cleaned.reset_index(drop=True)
        X_train = df_train_80[['runners_weight', 'runners_post_pos',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
        X_test = df_test_20[['runners_weight', 'runners_post_pos',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
        X_test = X_test.rename(columns={'runners_post': 'runners_post_pos'})
        X_unseen=df_unseen[['runners_weight', 'runners_post_pos',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
        X_test = X_test.rename(columns={'runners_post': 'runners_post_pos'})
    
        y_train = df_train_80[['HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']]
        y_test=df_test_20[['HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']]
        y_unseen=df_unseen[['HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']]

        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)
        print("y_unseen shape:", y_unseen.shape)
    
        kfold = KFold(n_splits=5)
        
        smote_pred = run_model_smote(X_train, y_train, X_test, y_test, kfold)
        # rfc=run_model_rfc(X_train, y_train, X_test, y_test, kfold)
        class_pred_dict = {'SMOTE': smote_pred}
        strat1_results = pd.DataFrame(columns=['Model', 'Money', 'Bets Made'])
        strat2_results = pd.DataFrame(columns=['Model', 'Money', 'Bets Made'])
        print("lr_pred is:",smote_pred)
        for model_name, class_model in class_pred_dict.items():
                print(model_name)
                money, bets = simple_class_strategy(class_model,df_unseen, graph=True)
                money1,bets1=top3_strategy(class_model,df_unseen,graph=True)
                strat1_results.loc[len(strat1_results)] = [model_name, money, sum(bets)]
                strat2_results.loc[len(strat2_results)] = [model_name, money1, sum(bets1)]
    

    if st.button('BACKTEST: N.A. WIN with Pairs Deep Learning'):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        df_results = pd.DataFrame(columns=['Model', 'Prediction', 'CV-F1', 'F1 Score', 'AUC', 'Recall', 'Precision'])
        # Get unique race IDs
        df_horse=pd.read_csv('./data/df_train.csv')
        unique_race_ids = df_horse['race_id'].unique()
        # Split the race IDs into training, test, and unseen sets
        train_race_ids, test_race_ids = train_test_split(unique_race_ids, test_size=0.2, random_state=42)
        test_race_ids, unseen_race_ids = train_test_split(test_race_ids, test_size=0.5, random_state=42)
        # Split the data based on the race IDs
        df_train_80= df_horse[df_horse['race_id'].isin(train_race_ids)]
        df_test_20 = df_horse[df_horse['race_id'].isin(test_race_ids)]
        df_unseen = df_horse[df_horse['race_id'].isin(unseen_race_ids)]

        df_train_80.to_csv('./data/df_train_80.csv', index=False)
        df_test_20.to_csv('./data/df_test_20.csv', index=False)
        df_unseen.to_csv('./data/df_unseen.csv', index=False)

        df_train_80.reset_index(inplace=True, drop=True)
        df_test_20.reset_index(inplace=True, drop=True)
        
        df_cleaned = df_train_80.dropna(subset=['distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        df_train_80 = df_cleaned.reset_index(drop=True)
        df_cleaned = df_test_20.dropna(subset=['distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        df_test_20 = df_cleaned.reset_index(drop=True)

        df_cleaned = df_unseen.dropna(subset=['distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        df_unseen= df_cleaned.reset_index(drop=True)
        X_train = df_train_80[['race_id','finish_position','runners_weight', 'runners_post_pos',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
        X_test = df_test_20[['race_id','finish_position','runners_weight', 'runners_post_pos',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
        X_test = X_test.rename(columns={'runners_post': 'runners_post_pos'})
        X_unseen=df_unseen[['race_id','finish_position','runners_weight', 'runners_post_pos',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
        X_test = X_test.rename(columns={'runners_post': 'runners_post_pos'})
        y_train = df_train_80[['HorseWin']]
        y_test=df_test_20[['HorseWin']]
        y_unseen=df_unseen[['HorseWin']]


        def create_pairwise_data_by_race(df):
            from itertools import combinations
            print("processing data into pairs")
            pairs = []
            features = ['runners_weight', 'runners_post_pos', 'jockey_ave_rank', 
                        'trainer_ave_rank', 'recent_ave_rank', 'distance_value']
            for race_id, race_df in df.groupby('race_id'):
                if len(race_df) < 2:
                    continue  
                for horse1_idx, horse2_idx in combinations(race_df.index, 2):
                    horse1_features = df.loc[horse1_idx, features].values
                    horse2_features = df.loc[horse2_idx, features].values
                    pair_features = list(horse1_features) + list(horse2_features)
                    result = 1 if df.loc[horse1_idx, 'finish_position'] < df.loc[horse2_idx, 'finish_position'] else 0
                    pairs.append((pair_features, result))
            return pairs

        pairwise_data_train = create_pairwise_data_by_race(X_train)
        pairwise_data_test = create_pairwise_data_by_race(X_test)

        from sklearn.preprocessing import StandardScaler
        X = np.array([pair[0] for pair in pairwise_data_train])  # Input features (horse1 vs horse2)
        y = np.array([pair[1] for pair in pairwise_data_train])  # Target labels (1 for win, 0 for loss)
                # Ensure X_train and X_test are 2D arrays (with shape (n_samples, n_features))
        y=y.reshape(-1,1)

        print(X.shape)  # Should print something like (n_pairs, n_features)
        print(y.shape)


        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)  # Fit and transform the training data
      
        print(X_train_scaled[:5])  
        print(y[:5])  
        from keras.layers import Dense, Input
        from keras.optimizers import Adam
        from keras.optimizers import SGD
        from keras.layers import Dropout



        # model = Sequential([
        #     Input(shape=(12,)),  # Specify input shape
        #     Dense(5, activation='relu'),
        #     Dense(7, activation='relu'),
        #     Dense(1, activation='sigmoid')
        # ])
        from sklearn.utils import class_weight
        # y_train_flat = np.array(y).flatten()

        # Compute class weights for imbalanced data
        # class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)
      


        from keras.regularizers import l2

        # Define the model
        model = Sequential([
            Input(shape=(12,)),  
            Dense(10, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid') 
        ])

        # sgd_optimizer = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',  #
              metrics=['accuracy']) 
        model.fit(X_train_scaled, y, epochs=400, batch_size=1000)
        
        X2 = np.array([pair[0] for pair in pairwise_data_test])  # Input features (horse1 vs horse2)
        y2 = np.array([pair[1] for pair in pairwise_data_test])  # Target labels (1 for win, 0 for loss)
        # y2=np.array(y2).flatten()
        X_test_scaled = scaler.transform(X2)   
        predictions=model.predict(X_test_scaled)
        print(predictions)
        binary_predictions = (predictions > 0.5).astype(int)
        print(binary_predictions)














        # def predict_race_outcome(model, race_df):
        #     pairs = create_pairwise_data_by_race(race_df)
        #     X_new = np.array([pair[0] for pair in pairs])  # Input features
        #     # Predict the probability that horse1 wins in each pair
        #     predictions = model.predict(X_new)
        #     return predictions
        


        # def rank_horses(predictions, race_df):
        #     horse_names = race_df['runners_horse_name'].values
        #     win_counts = {horse: 0 for horse in horse_names}
        #     pair_idx = 0
        #     for horse1, horse2 in combinations(horse_names, 2):
        #         if predictions[pair_idx] > 0.5:  # horse1 is predicted to win
        #             win_counts[horse1] += 1
        #         else:  # horse2 is predicted to win
        #             win_counts[horse2] += 1
        #         pair_idx += 1
        #     # Sort horses by the number of predicted wins
        #     ranked_horses = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
        #     return ranked_horses



        # # Example: Rank horses based on predictions
        # predictions = predict_race_outcome(model, df_train)
        # ranked_horses = rank_horses(predictions, df_train)
        # print(ranked_horses)
        


    # if st.button('PREDICT: North American WIN with Multi-Class Deep Learning'):
    #     runs_df=pd.read_csv('./data/df_train.csv')
    #     runs_df['runners_post_pos'] = pd.to_numeric(runs_df['runners_post_pos'], errors='coerce')
    #     runs_df = runs_df[(runs_df['runners_post_pos'] >= 1) & (runs_df['runners_post_pos'] <= 14)]
    #     runs_df = runs_df[runs_df['runners_post_pos'].between(1, 14, inclusive='both')]
    #     # Check if any values are outside the expected range
    #     print("\nUnique values in 'runners_post_pos':", runs_df['runners_post_pos'].unique())
    #     df_train = pd.read_csv('./data/df_train.csv')
    #     df_train.reset_index(inplace=True, drop=True)
    #     df_cleaned = runs_df.dropna(subset=['morning_line_odds','distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
    #                         'trainer_ave_rank', 'recent_ave_rank'])
    #     # Reset index
    #     runs_df = df_cleaned.reset_index(drop=True)
    #     runs_df = runs_df[['race_id','runners_weight', 'runners_post_pos','morning_line_odds',
    #                         #'win_odds',
    #                         'jockey_ave_rank',
    #                         'trainer_ave_rank', 'recent_ave_rank', 'distance_value','finish_position']]
    #     # Get a list of all current columns
    #     columns = list(runs_df.columns)
    #     # Reorder columns putting 'race_id' and 'runners_post' at the beginning
    #     columns_reordered = ['race_id', 'runners_post_pos'] + [col for col in columns if col not in ['race_id', 'runners_post_pos']]
    #     # Reassign to DataFrame with reordered columns
    #     runs_df = runs_df[columns_reordered]
    #     runs_df.to_csv('./data/runs_df.csv')
    #     runs_df = runs_df.pivot(index='race_id', columns='runners_post_pos', values=runs_df.columns[2:])
    #     rearranged_columns = sorted(list(runs_df.columns.values), key=group_horse_and_result)
    #     runs_df = runs_df[rearranged_columns]
    #     runs_df = runs_df.fillna(0)
    #     print(runs_df.head())
    #     X = runs_df[runs_df.columns[:-14]]
    #     ss = preprocessing.StandardScaler()
    #     X = pd.DataFrame(ss.fit_transform(X),columns = X.columns)
    #     X.to_csv('./data/X_multi.csv', index=False)
    #     y = runs_df[runs_df.columns[-14:]].applymap(lambda x: 1.0 if 0.5 < x < 1.5 else 0.0) 
    #     y.to_csv('./data/y_multi.csv', index=False)
    #     print(X.shape)
    #     print(y.shape)
    #     print("X shape after preprocessing:", X.shape)
    #     print("y shape after preprocessing:", y.shape)
    #     import sklearn.model_selection as model_selection
    #     X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
    #     model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(50, activation='relu', input_shape=(84,)),
    #     tf.keras.layers.Dense(14, activation='softmax')
    #     ])
    #     model.compile(optimizer=tf.keras.optimizers.Adam(5e-04),
    #                 loss=tf.keras.losses.CategoricalCrossentropy(),
    #                 metrics=[tf.keras.metrics.Precision(name='precision')])
    #     print(model.summary())
    #     dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
    #     train_dataset = dataset.shuffle(len(X_train)).batch(500)
    #     dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
    #     validation_dataset = dataset.shuffle(len(X_test)).batch(500)
    #     print("Start training..\n")
    #     history = model.fit(train_dataset, epochs=100, validation_data=validation_dataset)
    #     print("Done.")
    #     df_test=pd.read_csv('./data/df_test.csv')
    #     df_test[['recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank']] = df_test.apply(
    #     lambda row: compute_average_ranks(row, df_test, df_train),
    #     axis=1)
    #     df_test['runners_post'] = pd.to_numeric(df_test['runners_post'], errors='coerce')
    #     df_test = df_test[(df_test['runners_post'] >= 1) & (df_test['runners_post'] <= 14)]
    #     df_test = df_test[df_test['runners_post'].between(1, 14, inclusive='both')]
    #     df_cleaned = df_test.dropna(subset=['morning_line_odds','distance_value', 'runners_post', 'runners_weight','jockey_ave_rank',
    #                         'trainer_ave_rank', 'recent_ave_rank'])
    #     # Reset index
    #     df_test = df_cleaned.reset_index(drop=True)
    #     df_test = df_test[['race_id','runners_weight', 'runners_post','morning_line_odds',
    #                         #'win_odds',
    #                         'jockey_ave_rank',
    #                         'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
    #     # Get a list of all current columns
    #     columns = list(df_test.columns)
    #     # Reorder columns putting 'race_id' and 'runners_post' at the beginning
    #     columns_reordered = ['race_id', 'runners_post'] + [col for col in columns if col not in ['race_id', 'runners_post']]
    #     # Reassign to DataFrame with reordered columns
    #     df_test = df_test[columns_reordered]
    #     print(df_test.columns)
    #     df_test.to_csv('./data/df_test_multi.csv')
    #     df_test = df_test.pivot(index='race_id', columns='runners_post', values=df_test.columns[2:])
    #     df_test = df_test.fillna(0)
    #     df_test.to_csv(('./data/df_test_multi_pivot.csv'))
    #     # Make predictions
    #     probabilities = model.predict(df_test)
    #     # Find the index of the maximum probability for each row
    #     max_prob_indices = np.argmax(probabilities, axis=1)
    #     # Create a DataFrame to store the indices
    #     indices_df = pd.DataFrame({'PredictedClass': max_prob_indices+1})
    #     # Display the DataFrame in Streamlit
    #     st.title("Model Predictions")
    #     st.dataframe(indices_df)


    st.subheader("PREDICT")
    if st.button('PREDICT: N.A. WIN with Logistic Regression'):
        df_train = pd.read_csv('./data/df_train.csv')
        df_train.reset_index(inplace=True, drop=True)
        df_test=pd.read_csv('./data/df_test.csv')

        # df_train= df_train[df_train['track_name'] == 'Thistledown']
        # df_test = df_test[df_test['track_name'] == 'Thistledown']            

        df_test.drop(columns=['owner_full', 'owner_l_name', 'owner_f_name'], inplace=True)
        # Select the first instance of each runners_horse_name in df_train
        df_train_unique = df_train.drop_duplicates(subset='runners_horse_name', keep='first')

        # Merge with df_test
        df_test = pd.merge(
            df_test, 
            df_train_unique[['runners_horse_name', 'owner_full']], 
            on='runners_horse_name', 
            how='left')
        print(df_test.head())
       
        df_test[['recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank']] = df_test.apply(
        lambda row: compute_average_ranks(row, df_test, df_train),
        axis=1)
        df_test.to_csv('./data/df_test_2.csv')
        # df_test.to_csv('./data/df_test.csv',index=False)
        df_test.reset_index(inplace=True, drop=True)
        df_cleaned = df_train.dropna(subset=['morning_line_odds','distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        # Reset index
        df_train = df_cleaned.reset_index(drop=True)

        df_cleaned = df_test.dropna(subset=['morning_line_odds','distance_value', 'runners_post', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)
        X_train = df_train[['runners_weight', 'runners_post_pos',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank',  'recent_ave_rank', 'distance_value']]
        X_test = df_test[['runners_weight', 'runners_post',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
        X_test = X_test.rename(columns={'runners_post': 'runners_post_pos'})
        print("X test is",X_test)
        y_train = df_train[['HorseWin']]
        lr =  LogisticRegression(solver='lbfgs', penalty='l2', C=1, max_iter=500)
        # Do hyperparameter tuning
        param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                        'penalty': [ 'l2']}
        kfold = KFold(n_splits=5)
        grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5)
        # find the best parameters
        grid.fit(X_train, y_train['HorseWin'].to_numpy())
        # Print the best parameters
        print(grid.best_params_)
        print(grid.best_score_)
        # Initialize the model using best parameters
        lr = grid.best_estimator_
        print("Model coefficients:",lr.coef_)
        # Create a dataframe to store the predictions
        df_pred = pd.DataFrame()
        df_pred['Date']=df_test['date']
        df_pred['Horse Name'] = df_test['runners_horse_name']
        df_pred['Horse Number'] = df_test['runners_program_number']
        df_pred['Track']=df_test['track_name']
        # df_pred['RaceID'] = df_test['race_id']
        df_pred['Race Number']=df_test['race_key_race_number']
        df_pred['Jockey'] = df_test['jockey_full']
        df_pred['Trainer'] = df_test['trainer_full']
        df_pred['Owner']=df_test['owner_full']
        df_pred['M/L'] = df_test['morning_line_odds']
        df_pred['Live Odds'] = df_test['live_odds']
        # df_pred['Win Percentage'] = df_test['Win_Percentage']
        # df_pred['Final Odds']=df_test['l']
        # Make predictions
        y_test_pred = lr.predict(X_test)
        # Store the predictions in the dataframe
        df_pred['HorseWin'] = y_test_pred
        # Save predictions to csv
        pd.DataFrame(df_pred).to_csv('./predictions/deploy_pred.csv')
        # Filter the DataFrame to include only rows where HorseWin = 1
        winning_horses = df_pred[df_pred['HorseWin'] == 1]

        
        # race_id_counts = winning_horses['RaceID'].value_counts()
        # duplicate_race_ids = race_id_counts[race_id_counts > 1].index

        # Step 3: Exclude races with duplicate race_ids
        # filtered_winning_horses = winning_horses[~winning_horses['RaceID'].isin(duplicate_race_ids)]

        # Reset the index of the filtered DataFrame
        filtered_winning_horses = winning_horses.reset_index(drop=True)

        # Display the filtered DataFrame in Streamlit
        st.title("Bet on the races below")
        st.dataframe(filtered_winning_horses)

      # Predict probabilities
        y_pred_probs = lr.predict_proba(X_test)[:, 1]  # Probability of the horse winning

        # Compare with market odds
        # Copy X_test to a new DataFrame
        df_test['model_prob'] = y_pred_probs
        df_test['implied_prob'] = 1 / df_test['decimal_odds']

        # Identify undervalued horses
        undervalued_horses = df_test[df_test['model_prob'] > df_test['implied_prob']]

        # Streamlit display
        st.write("Undervalued Horses")
        st.dataframe(undervalued_horses)

    if st.button('PREDICT: N.A. EXACTAS with Logistic Regression'):
        # Load and prepare data
        df_train = pd.read_csv('./data/df_train.csv')
        df_train.reset_index(inplace=True, drop=True)
        df_test = pd.read_csv('./data/df_test.csv')
        df_test[['recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank']] = df_test.apply(
            lambda row: compute_average_ranks(row, df_test, df_train),
            axis=1)
        
        df_test.to_csv('./data/df_test2.csv', index=False)
        df_test.reset_index(inplace=True, drop=True)

        # Clean the data
        df_cleaned_train = df_train.dropna(subset=['morning_line_odds', 'distance_value', 'runners_post_pos', 'runners_weight', 'jockey_ave_rank', 'trainer_ave_rank', 'recent_ave_rank'])
        df_train = df_cleaned_train.reset_index(drop=True)

        df_cleaned_test = df_test.dropna(subset=['morning_line_odds', 'distance_value', 'runners_post', 'runners_weight', 'jockey_ave_rank', 'trainer_ave_rank', 'recent_ave_rank'])
        df_test = df_cleaned_test.reset_index(drop=True)

        # Features
        X_train = df_train[['runners_weight', 'runners_post_pos',  'jockey_ave_rank', 'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
        X_test = df_test[['runners_weight', 'runners_post',  'jockey_ave_rank', 'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
        X_test = X_test.rename(columns={'runners_post': 'runners_post_pos'})

        # Define targets
        y_train_win = df_train['HorseWin']
        y_train_second = df_train['HorseSecond']

        # Initialize models
        lr_win = LogisticRegression(solver='lbfgs', penalty='l2', C=1, max_iter=500)
        lr_second = LogisticRegression(solver='lbfgs', penalty='l2', C=1, max_iter=500)

        # Hyperparameter tuning
        param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'penalty': ['l2']}
        kfold = KFold(n_splits=5)
        
        # Model for predicting the first place
        grid_win = GridSearchCV(estimator=lr_win, param_grid=param_grid, cv=kfold)
        grid_win.fit(X_train, y_train_win)
        lr_win = grid_win.best_estimator_
        
        # Model for predicting the second place
        grid_second = GridSearchCV(estimator=lr_second, param_grid=param_grid, cv=kfold)
        grid_second.fit(X_train, y_train_second)
        lr_second = grid_second.best_estimator_

        # Print the best parameters
        print("Win Model - Best Params:", grid_win.best_params_)
        print("Second Place Model - Best Params:", grid_second.best_params_)
        
        # Create a dataframe to store the predictions
        df_pred = pd.DataFrame()
        df_pred['Date'] = df_test['date']
        df_pred['RaceID'] = df_test['race_id']
        df_pred['Track'] = df_test['track_name']
        df_pred['HorseID'] = df_test['runners_horse_name']

        # Make predictions
        y_test_pred_win = lr_win.predict(X_test)
        y_test_pred_second = lr_second.predict(X_test)

        # Store the predictions in the dataframe
        df_pred['HorseWin'] = y_test_pred_win
        df_pred['HorseSecond'] = y_test_pred_second

        # Filter for exacta predictions
        # Ensure that for each race_id, there is one horse predicted as first and a different horse predicted as second
        exacta_predictions = []
        for race_id in df_pred['RaceID'].unique():
            df_race = df_pred[df_pred['RaceID'] == race_id]
            if df_race['HorseWin'].sum() == 1 and df_race['HorseSecond'].sum() == 1:
                if not df_race[df_race['HorseWin'] == 1]['HorseID'].values[0] == df_race[df_race['HorseSecond'] == 1]['HorseID'].values[0]:
                    exacta_predictions.append(df_race)

        # Concatenate all exacta predictions into a single DataFrame
        df_exacta = pd.concat(exacta_predictions) if exacta_predictions else pd.DataFrame(columns=df_pred.columns)

        # Save predictions to csv
        df_exacta.to_csv('./predictions/deploy_pred_exactas.csv', index=False)
        
        # Display the predictions
        st.title("Bet Exactas on the races below")
        st.dataframe(df_exacta)

    if st.button('PREDICT: N.A. WIN with Deep Learning'):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        df_train = pd.read_csv('./data/df_train.csv')
        df_train.reset_index(inplace=True, drop=True)
        df_test=pd.read_csv('./data/df_test.csv')
        df_test.drop(columns=['owner_full', 'owner_l_name', 'owner_f_name'], inplace=True)
        # Select the first instance of each runners_horse_name in df_train
        df_train_unique = df_train.drop_duplicates(subset='runners_horse_name', keep='first')

        # Merge with df_test
        df_test = pd.merge(
            df_test, 
            df_train_unique[['runners_horse_name', 'owner_full']], 
            on='runners_horse_name', 
            how='left')
        print(df_test.head())
       
        df_test[['recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank']] = df_test.apply(
        lambda row: compute_average_ranks(row, df_test, df_train),
        axis=1)
        df_test.to_csv('./data/df_test_2.csv')
        # df_test.to_csv('./data/df_test.csv',index=False)
        df_test.reset_index(inplace=True, drop=True)
        df_cleaned = df_train.dropna(subset=['morning_line_odds','distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        # Reset index
        df_train = df_cleaned.reset_index(drop=True)

        df_cleaned = df_test.dropna(subset=['morning_line_odds','distance_value', 'runners_post', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)
        X_train = df_train[['runners_weight', 'runners_post_pos','morning_line_odds',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank',  'recent_ave_rank', 'distance_value']]
        X_test = df_test[['runners_weight', 'runners_post','morning_line_odds',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
        X_test = X_test.rename(columns={'runners_post': 'runners_post_pos'})
        # print("X test is",X_test)
        y_train = df_train[['HorseWin']]
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)  # Fit and transform the training data
        X_test_scaled=scaler_X.fit_transform(X_test)
      
        import tensorflow as tf
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
         # Define the model
        model = Sequential([
            Input(shape=(14,)),  # Input layer with 20 features
            Dense(60, activation='tanh'),  # First hidden layer with 10 neurons
            Dense(60, activation='tanh'),
            Dense(1, activation='sigmoid'),])  # Output layer
        from keras.callbacks import Callback, EarlyStopping

        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")
        model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['mae'])

        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)

        # Fit the model with both callbacks
        history = model.fit(X_train_scaled, y_train, epochs=5000, batch_size=128, validation_split=0.2,
                            callbacks=[streamlit_callback, early_stopping], verbose=0)
      
        epochs = range(1, len(history.history['loss']) + 1)


    
        st.title('Loss Curve Plot')

        # Plotting function
        def plot_loss(history):
            plt.figure(figsize=(10, 6))
            plt.scatter(epochs,history.history['loss'], label='Training Loss')
            plt.scatter(epochs,history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        # Display the plot
        st.pyplot(plot_loss(history))

        DL_pred = model.predict(X_test_scaled) #what should k-fold be?
        binary_predictions = (predictions >= 0.5).astype(int)
        st.write("Model Predictions:")

        df_pred = pd.DataFrame()
        df_pred['Date'] = df_test['date']
        df_pred['Course Name']=df_test['course']
        df_pred['HorseID'] = df_test['runners_horse_name']
        df_pred['RaceID'] = df_test['race_id']
        df_pred['Race Name'] = df_test['race_name']
        df_pred['Horse Number']=df_test['runners_program_numb']
        df_pred['HorseWin'] = binary_predictions

        winning_horses = df_sorted[df_sorted['HorseWin'] == 1]
        filtered_winning_horses = winning_horses.reset_index(drop=True)
        # Display the filtered DataFrame in Streamlit
        st.title("Bet on the races below")
        st.dataframe(filtered_winning_horses)

    if st.button("PREDICT: US WIN with Pairs Deep Learning"):
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras.models import Sequential
        from keras.layers import Dense,Input,Dropout
        from keras.optimizers import Adam
        from sklearn.utils import class_weight
        from keras.regularizers import l2
        from sklearn.preprocessing import LabelEncoder
        import tensorflow as tf
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        from keras.callbacks import Callback, EarlyStopping
        df_train = pd.read_csv('./data/df_train.csv')
        df_train.reset_index(inplace=True, drop=True)
        df_test=pd.read_csv('./data/df_test.csv')
        df_test.drop(columns=['owner_full', 'owner_l_name', 'owner_f_name'], inplace=True)
        # Select the first instance of each runners_horse_name in df_train
        df_train_unique = df_train.drop_duplicates(subset='runners_horse_name', keep='first')

        # Merge with df_test
        df_test = pd.merge(
            df_test, 
            df_train_unique[['runners_horse_name', 'owner_full']], 
            on='runners_horse_name', 
            how='left')
        print(df_test.head())
       
        df_test[['recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank']] = df_test.apply(
        lambda row: compute_average_ranks(row, df_test, df_train),
        axis=1)
        df_test.to_csv('./data/df_test_2.csv')
        # df_test.to_csv('./data/df_test.csv',index=False)
        df_test.reset_index(inplace=True, drop=True)
        df_cleaned = df_train.dropna(subset=['morning_line_odds','distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        # Reset index
        df_train = df_cleaned.reset_index(drop=True)

        df_cleaned = df_test.dropna(subset=['morning_line_odds','distance_value', 'runners_post', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)
        X_train = df_train[['race_id','runners_horse_name','track_name','race_key_race_number','finish_position','runners_weight', 'runners_post_pos','morning_line_odds',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank',  'recent_ave_rank', 'distance_value']]
        X_test = df_test[['race_id','runners_horse_name','track_name','race_key_race_number','runners_weight', 'runners_post','morning_line_odds',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
        X_test = X_test.rename(columns={'runners_post': 'runners_post_pos'})
        print("X test is",X_test)
        y_train = df_train[['HorseWin']]
        # scaler_X = StandardScaler()
        # scaler_y = StandardScaler()
        # X_train_scaled = scaler_X.fit_transform(X_train)  # Fit and transform the training data
        # X_test_scaled=scaler_X.fit_transform(X_test)

        def create_pairwise_data_by_race(df):
            from itertools import combinations
            import random
            print("processing data into pairs")
            pairs = []
            features = ['runners_weight', 'runners_post_pos','morning_line_odds',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank',  'recent_ave_rank', 'distance_value']
    

            # Sort data by race_id and position (for consistency)
            df = df.sort_values(by=['race_id', 'finish_position'])

            for race_id, race_df in df.groupby('race_id'):
                if len(race_df) < 2:
                    continue  

                # Shuffle horse indices to randomize the order of comparisons
                horse_indices = list(race_df.index)
                random.shuffle(horse_indices)

                for horse1_idx, horse2_idx in combinations(horse_indices, 2):
                    horse1_features = df.loc[horse1_idx, features].values
                    horse2_features = df.loc[horse2_idx, features].values
                    pair_features = list(horse1_features) + list(horse2_features)

                    # Compare positions
                    pos1 = df.loc[horse1_idx, 'finish_position']
                    pos2 = df.loc[horse2_idx, 'finish_position']

                    # Result is 1 if horse1 finished ahead of horse2, else 0
                    result = 1 if pos1 < pos2 else 0
                    
                    # Get horse IDs
                    horse1_id = df.loc[horse1_idx, 'runners_horse_name']  # Adjust this line based on your DataFrame
                    horse2_id = df.loc[horse2_idx, 'runners_horse_name']  # Adjust this line based on your DataFrame
                    course=df.loc[horse1_idx, 'track_name']
                    race_name=df.loc[horse1_idx, 'race_key_race_number']
                    # Include both horse IDs and race_id with the pair
                    pairs.append((pair_features, result, race_id,course,race_name, horse1_id, horse2_id))

            return pairs
        
        def create_pairwise_data_by_race_test(df):
            from itertools import combinations
            import random
            print("processing data into pairs")
            pairs = []
            features = ['runners_weight', 'runners_post_pos','morning_line_odds',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank',  'recent_ave_rank', 'distance_value']

            # Sort data by race_id (for consistency)
            df = df.sort_values(by=['race_id'])

            for race_id, race_df in df.groupby('race_id'):
                if len(race_df) < 2:
                    continue  # Skip races with fewer than 2 horses

                # Shuffle horse indices to randomize the order of comparisons
                horse_indices = list(race_df.index)
                random.shuffle(horse_indices)

                for horse1_idx, horse2_idx in combinations(horse_indices, 2):
                    horse1_features = df.loc[horse1_idx, features].values
                    horse2_features = df.loc[horse2_idx, features].values
                    pair_features = list(horse1_features) + list(horse2_features)

                    # Get horse IDs and race details
                    horse1_id = df.loc[horse1_idx, 'runners_horse_name']
                    horse2_id = df.loc[horse2_idx, 'runners_horse_name']
                    course = df.loc[horse1_idx, 'track_name']
                    race_name = df.loc[horse1_idx, 'race_key_race_number']

                    # Append the pair (no result)
                    pairs.append((pair_features, race_id, course, race_name, horse1_id, horse2_id))

            return pairs
                
        pairwise_data_train = create_pairwise_data_by_race(X_train)
        pairwise_data_test = create_pairwise_data_by_race_test(X_test)
        print("pairwise data is :", pairwise_data_train)

       
        X = np.array([pair[0] for pair in pairwise_data_train])  # Input features (horse1 vs horse2)
        y = np.array([pair[1] for pair in pairwise_data_train])  # Target labels (1 for win, 0 for loss)
                # Ensure X_train and X_test are 2D arrays (with shape (n_samples, n_features))
        y=y.reshape(-1,1)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X)  # Fit and transform the training data
        y_train_scaled = scaler_y.fit_transform(y)
        # print(X_train_scaled[:50])  
        # print(y_train_scaled[:50])  
      
        # Define the model
        model = Sequential([
            Dense(128, activation='tanh', input_shape=(14,)),  # Specify input shape here
            Dense(128, activation='tanh'),
            Dense(128, activation='tanh'),
            Dense(1, activation='sigmoid')
        ])
            

        # Define a custom callback to print loss at the end of each epoch
        class StreamlitCallback(Callback):
            def __init__(self, placeholder):
                super(StreamlitCallback, self).__init__()
                self.placeholder = placeholder

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.placeholder.write(f"Epoch {epoch + 1}: Loss = {loss}")
        model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',  #
              metrics=['accuracy']) 

        # Create a placeholder in Streamlit
        placeholder = st.empty()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the Streamlit callback
        streamlit_callback = StreamlitCallback(placeholder)

        # Fit the model with both callbacks
        history = model.fit(X_train_scaled, y, epochs=5000, batch_size=500, validation_split=0.2,
                            callbacks=[streamlit_callback, early_stopping], verbose=0)
        epochs = range(1, len(history.history['loss']) + 1)

    
        X2 = np.array([pair[0] for pair in pairwise_data_test])  # Input features (horse1 vs horse2)
        print("Size of X2:", X2.shape)
        X_test_scaled = scaler_x.fit_transform(X2)
       
       # Streamlit app
        st.title('Loss Curve Plot')

        # Plotting function
        def plot_loss(history):
            plt.figure(figsize=(10, 6))
            plt.scatter(epochs,history.history['loss'], label='Training Loss')
            plt.scatter(epochs,history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        # Display the plot
        st.pyplot(plot_loss(history))
        DL_pred = model.predict(X_test_scaled) 
        # Convert to binary labels
        pairwise_predictions = (DL_pred > 0.5).astype(int)


        from collections import defaultdict

        # Initialize a dictionary to hold the scores for each horse in each race
        horse_scores = defaultdict(lambda: defaultdict(int))

        # Loop through the pairwise data and predictions to tally wins for each horse
        for pred, (pair_features, race_id, course, race_name, horse1_id, horse2_id) in zip(pairwise_predictions, pairwise_data_test):
            if pred == 1:
                # If horse1 is predicted to beat horse2, increment horse1's score
                horse_scores[race_id][horse1_id] += 1
            else:
                # Otherwise, increment horse2's score
                horse_scores[race_id][horse2_id] += 1

        # Now horse_scores contains the number of wins for each horse in each race


           #Create a DataFrame to hold the predicted winners and their horse IDs
        predicted_winners_dict = {
            'race_id': [],
            'predicted_winner': []
        }

        # Iterate over horse_scores to determine the predicted winners
        for race_id, scores in horse_scores.items():
            if scores:  # Ensure scores is not empty
                predicted_winner = max(scores, key=scores.get)
                # Append details to the dictionary
                predicted_winners_dict['race_id'].append(race_id)
                predicted_winners_dict['predicted_winner'].append(predicted_winner)

        # Convert to DataFrame
        predicted_winners_df = pd.DataFrame(predicted_winners_dict)
        print(df_test.columns)
        predicted_winners_df =predicted_winners_df.rename(columns={'predicted_winner': 'runners_horse_name'})
        merged_df = predicted_winners_df.merge(df_test[['date','race_id','track_name','runners_horse_name',  'race_key_race_number']],
                                        on=['race_id', 'runners_horse_name'],
                                        how='left')
        duplicates = df_test[df_test.duplicated(['race_id'], keep=False)]
        print(duplicates)
        # Display the DataFrame in Streamlit
        st.write(merged_df)


    if st.button('PREDICT: HK WIN with LSTM'):
        races= pd.read_csv('./data/races_HK_kaggle.csv')
        races.reset_index(inplace=True, drop=True)
        runs=pd.read_csv('./data/runs_HK_kaggle.csv')
        runs.reset_index(inplace=True, drop=True)
        df_runners = pd.merge(races, runs, on='race_id', how='left')
        st.write(df_runners.head())
        # Specify the horse_id you are interested in
        horse_id = 3917  # Replace with the actual horse_id

        # Filter the DataFrame for this particular horse_id
        filtered_df = df_runners[df_runners['horse_id'] == horse_id]

        # Count the number of entries in the 'results' column for this horse_id
        results_count = filtered_df['result'].count()

        # Display the count in Streamlit
        st.write(f'The number of result entries for horse_id {horse_id} is: {results_count}')
        n_ts = 10  ## Length of each time sequence or the forcast window
        n_f = 1
        n_prediction = 10
        def prepare_time_seqs(data):
            output_df = pd.DataFrame()
            for i in range(n_ts):
                output_df[i] = data.shift(n_ts-i)
            output_df['target'] = data
            return output_df
        data=filtered_df['result']
        max_value = filtered_df['result'].max()
        df = prepare_time_seqs(data)
        st.dataframe(df.head(20))
        df.dropna(axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.dataframe(df.head(20))  
        Y = df.target.values
        # Y = Y/Y.max()
        X = df.drop('target', axis=1).values
        print(X.shape)
        X = X.reshape(-1, n_ts, n_f)
        X.shape     
                # Assuming X is your reshaped array
        np.save('./data/X.npy', X)

        X_future = X[-n_prediction:]
        Y_future = Y[-n_prediction:]

        X = X[:-n_prediction]
        Y = Y[:-n_prediction]

        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        import keras
        from keras import layers, models
        from keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout
        from keras.layers import Masking, LSTM, Dense
        from keras.preprocessing.sequence import pad_sequences
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0)
        print(X_train.shape)

        model = models.Sequential()
        model.add(LSTM(500, input_shape =(n_ts, n_f) ,
                    activation='tanh',
                    return_sequences=True)
                )
        model.add(LSTM(32,  activation='tanh')  )
        model.add(Dense(32,  activation='relu')  )
        model.add(Dense(1,  activation='linear')  )
        model.compile(loss = keras.losses.mean_squared_error,
                    optimizer=keras.optimizers.Adam(learning_rate = .0001) )

        history = model.fit(X_train, Y_train,
                    epochs=2000, batch_size=4,
                    verbose=1, validation_data=(X_test, Y_test))
        

        Y_fut_predicted = model.predict(X_future)

        def extract_history_data(history):
            # Ensure 'loss' and 'val_loss' keys exist in the history object
            loss = history.history.get('loss', [])
            val_loss = history.history.get('val_loss', [])
            return loss, val_loss
        st.title('Model Loss and Prediction Plots')
        loss, val_loss = extract_history_data(history)
        # Plot model loss
        fig1 = plt.figure(figsize=(7, 5))
        ax1 = fig1.add_subplot(1, 1, 1, title='Model loss', ylabel='Loss')
        plt.plot(loss[2:], label='Train')
        plt.plot(val_loss[2:], label='Validation')
        plt.legend(loc='upper left')
        plt.tight_layout()

        # Display model loss plot
        st.subheader('Model Loss Plot')
        st.pyplot(fig1)

        # Plot real vs. predicted values
        fig2 = plt.figure(figsize=(7, 5))
        ax2 = fig2.add_subplot(1, 1, 1, xlabel='Time', ylabel='Loss')
        plt.plot(Y_future, label='Real Data')
        plt.plot(Y_fut_predicted, label='Predicted values')
        plt.xlabel('Time')
        plt.legend(loc='best')
        plt.tight_layout()

        # Display real vs. predicted values plot
        st.subheader('Real vs. Predicted Values Plot')
        st.pyplot(fig2)
        
    if st.button('PREDICT: N.A. Logistic Regression with Graph-Based Features'):
        import pandas as pd
        import networkx as nx
        import matplotlib.pyplot as plt
        from itertools import combinations
        import torch
        from torch_geometric.data import Data

        data=pd.read_csv('./data/df_train.csv')

        df = pd.DataFrame(data)

        # Limit to the first 100 races
        unique_races = df['race_id'].unique()
        df_limited = df[df['race_id'].isin(unique_races)]

        def create_graph(df):
            G = nx.MultiDiGraph()  # Use MultiDiGraph for multiple edges
            for race_id, group in df.groupby('race_id'):
                # Sort horses by finish position
                sorted_horses = group.sort_values('finish_position')['runners_horse_name'].tolist()
                for i in range(len(sorted_horses)):
                    for j in range(i + 1, len(sorted_horses)):
                        u, v = sorted_horses[i], sorted_horses[j]
                        # A beats B, directed edge from A to B
                        G.add_edge(u, v)
                        # Uncomment the next line if you want to keep track of race_id, or other attributes per edge
                        # G.add_edge(u, v, race_id=race_id)
            return G
        def plot_graph(G, show_labels=False):
            plt.figure(figsize=(12, 10))
            
            # Choose a layout for the graph
            pos = nx.spring_layout(G, seed=42)  # Using spring layout for better readability
            
            # Draw the graph
            nx.draw(G, pos, with_labels=show_labels, node_color='red', font_weight='normal', node_size=10, edge_color='gray', arrows=True)
            
            # Optionally, draw edge labels if you want to show weights
            # if nx.get_edge_attributes(G, 'weight'):
            #     edge_labels = nx.get_edge_attributes(G, 'weight')
            #     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
            plt.title("Directed Graph of Horses by Race Finish Order")
            st.pyplot(plt)

        G = create_graph(df_limited)
        # plot_graph(G, show_labels=False)  # Set `show_labels` to True if you want to display node labels


        def calculate_degree_ratio(G):
            # Calculate in-degrees and out-degrees for each node (horse)
            in_degrees = dict(G.in_degree())  # In-degree for each node
            out_degrees = dict(G.out_degree())  # Out-degree for each node
            
            degree_data = []

            for node in G.nodes():
                in_deg = in_degrees.get(node, 0)
                out_deg = out_degrees.get(node, 0)
                
                total_deg = in_deg + out_deg  # Total degree (in-degree + out-degree)
                
                if total_deg > 0:
                    degree_ratio = (in_deg - out_deg) / total_deg
                else:
                    degree_ratio = 0  # Handle isolated nodes with no in/out edges
                
                degree_data.append({'runners_horse_name': node, 'degree_ratio': degree_ratio})
            
            # Convert the list of dictionaries to a pandas DataFrame
            degree_df = pd.DataFrame(degree_data)
            return degree_df

        # Calculate the degree ratios and get them in a DataFrame
        degree_df = calculate_degree_ratio(G)

        # Display the DataFrame with runners_horse_name and degree_ratio
        print(degree_df)

        # Save the DataFrame to a CSV file
        csv_file_path = "./data/graph/horse_degree_ratios.csv"
        degree_df.to_csv(csv_file_path, index=False)
        print(f"Degree ratios saved to {csv_file_path}")

        def calculate_win_loss_spread(G):
            win_loss_spreads = []

            # Iterate through each pair of nodes with edges between them
            for u, v in G.edges():
                # Get the number of wins from u to v (how many times u beat v)
                u_beats_v = G.number_of_edges(u, v)
                # Get the number of wins from v to u (how many times v beat u)
                v_beats_u = G.number_of_edges(v, u)
                
                # Calculate the win-loss spread (wins - losses)
                win_loss_spread = u_beats_v - v_beats_u
                
                # Store the result in a list
                win_loss_spreads.append({
                    'horse_1': u,
                    'horse_2': v,
                    'horse_1_wins': u_beats_v,
                    'horse_2_wins': v_beats_u,
                    'win_loss_spread': win_loss_spread
                })

            # Convert the result to a DataFrame for easy viewing
            win_loss_df = pd.DataFrame(win_loss_spreads).drop_duplicates(subset=['horse_1', 'horse_2'])
            
            return win_loss_df

        # Calculate the win-loss spread for the graph G
        win_loss_df = calculate_win_loss_spread(G)
        
        # Display the win-loss spread DataFrame
        print(win_loss_df)

        # Save the DataFrame to a CSV file
        csv_file_path = "./data/graph/horse_win_loss_spread.csv"
        win_loss_df.to_csv(csv_file_path, index=False)
        print(f"Win-Loss spread saved to {csv_file_path}")

        def calculate_edge_weight_difference(G):
            edge_weight_diff = []
            
            # Get the number of wins and losses for each node
            for node in G.nodes():
                # Sum weights of incoming edges
                in_weights = sum([G.number_of_edges(u, node) for u in G.predecessors(node)])
                # Sum weights of outgoing edges
                out_weights = sum([G.number_of_edges(node, v) for v in G.successors(node)])
                
                # Calculate the edge weight difference
                weight_diff = (out_weights - in_weights)/(out_weights + in_weights)
                
                edge_weight_diff.append({
                    'runners_horse_name': node,
                    'in_weights': in_weights,
                    'out_weights': out_weights,
                    'weight_difference': weight_diff
                })
            
            weight_diff_df = pd.DataFrame(edge_weight_diff)
            return weight_diff_df
            # Calculate the edge weight differences for the graph G
        edge_weight_diff_df = calculate_edge_weight_difference(G)
        # Display the DataFrame with node, in_weights, out_weights, and weight_difference
        print(edge_weight_diff_df)

        # Save the DataFrame to a CSV file
        csv_file_path = "./data/graph/edge_weight_differences.csv"
        edge_weight_diff_df.to_csv(csv_file_path, index=False)
        print(f"Edge weight differences saved to {csv_file_path}")
        
        hubs, authorities = nx.hits(G)

      
        hub_data = {
            'runners_horse_name': list(hubs.keys()),
            'hub_score': list(hubs.values()),
            'authority_score': list(authorities.values())
        }

        hub_df = pd.DataFrame(hub_data)

        hub_df.to_csv('./data/graph/hits_scores.csv', index=False)
        print("CSV file 'hits_scores.csv' created successfully.")
            

        df_train = pd.read_csv('./data/df_train.csv')
        df_train.reset_index(inplace=True, drop=True)
        df_train=pd.merge(df_train,degree_df,on='runners_horse_name',how='left')
        df_train=pd.merge(df_train,edge_weight_diff_df,on='runners_horse_name',how='left')
        df_train=pd.merge(df_train,hub_df,on='runners_horse_name',how='left')


        df_test=pd.read_csv('./data/df_test.csv')
        df_test.drop(columns=['owner_full', 'owner_l_name', 'owner_f_name'], inplace=True)
        # Select the first instance of each runners_horse_name in df_train

        df_train= df_train[df_train['track_name'] == 'Thistledown']
        df_test = df_test[df_test['track_name'] == 'Thistledown']       


        df_train_unique = df_train.drop_duplicates(subset='runners_horse_name', keep='first')

        # Merge with df_test
        df_test = pd.merge(
            df_test, 
            df_train_unique[['runners_horse_name', 'owner_full']], 
            on='runners_horse_name', 
            how='left')
        print(df_test.head())
       
        df_test=pd.merge(df_test,degree_df,on='runners_horse_name',how='left')
        df_test=pd.merge(df_test,edge_weight_diff_df,on='runners_horse_name',how='left')
        df_test=pd.merge(df_test,hub_df,on='runners_horse_name',how='left')
        df_test[['recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank']] = df_test.apply(
        lambda row: compute_average_ranks(row, df_test, df_train),
        axis=1)
        df_test.to_csv('./data/df_test_2.csv')
        # df_test.to_csv('./data/df_test.csv',index=False)
        df_test.reset_index(inplace=True, drop=True)
        df_cleaned = df_train.dropna(subset=['distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        # Reset index
        df_train = df_cleaned.reset_index(drop=True)

        # threshold = 15 # Set your threshold for high win_payoff
        # df_train = df_train[(df_train['morning_line_odds'] > threshold)]


        df_cleaned = df_test.dropna(subset=['morning_line_odds','distance_value', 'runners_post', 'runners_weight','jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank'])
        # Reset index
        df_test = df_cleaned.reset_index(drop=True)
        # threshold = 15 # Set your threshold for high win_payoff
        # df_test = df_test[(df_test['morning_line_odds'] > threshold)]
        X_train = df_train[['runners_weight', 'runners_post_pos',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank',  'recent_ave_rank', 'distance_value','degree_ratio']]
        
      
        X_test = df_test[['runners_weight', 'runners_post',
                            #'win_odds',
                            'jockey_ave_rank',
                            'trainer_ave_rank', 'recent_ave_rank', 'distance_value','degree_ratio']]
        X_test = X_test.rename(columns={'runners_post': 'runners_post_pos'})
        print("X test is",X_test)

        from sklearn.preprocessing import MinMaxScaler
        # Initialize the scaler
        scaler = MinMaxScaler()
        # Fit and transform the training data
        X_train_scaled = scaler.fit_transform(X_train)
        # Transform the test data using the same scaler
        X_test_scaled = scaler.transform(X_test)
        # Convert scaled data back to DataFrame for easier handling
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        y_train = df_train[['HorseWin']]

        lr =  LogisticRegression(solver='lbfgs', penalty='l2', C=1, max_iter=500)
        # Do hyperparameter tuning
        param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                        'penalty': [ 'l2']}
        kfold = KFold(n_splits=5)
        grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5)
        # find the best parameters
        grid.fit(X_train, y_train['HorseWin'].to_numpy())
        # Print the best parameters
        print(grid.best_params_)
        print(grid.best_score_)
        # Initialize the model using best parameters
        lr = grid.best_estimator_
        import shap

        # Initialize the SHAP explainer
        explainer = shap.Explainer(lr, X_train)

        # Calculate SHAP values
        shap_values = explainer(X_train)

        
        # Create a Streamlit app
        st.title("SHAP Feature Importance - Bar Plot")

        # Generate SHAP bar plot
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, show=False)  # Prevent SHAP from displaying the plot automatically

        # Display the bar plot in Streamlit
        st.pyplot(fig)

        print("Model coefficients:",lr.coef_)
        # Create a dataframe to store the predictions
        df_pred = pd.DataFrame()
        df_pred['Date']=df_test['date']
        df_pred['Horse Name'] = df_test['runners_horse_name']
        df_pred['Horse Number'] = df_test['runners_program_number']
        df_pred['Track']=df_test['track_name']
        df_pred['Race Number']=df_test['race_key_race_number']
        # df_pred['RaceID'] = df_test['race_id']
        df_pred['Jockey'] = df_test['jockey_full']
        df_pred['Trainer'] = df_test['trainer_full']
        df_pred['Owner']=df_test['owner_full']
        df_pred['M/L'] = df_test['morning_line_odds']
        df_pred['Live Odds'] = df_test['live_odds']
        # df_pred['Win Percentage'] = df_test['Win_Percentage']
        # df_pred['Final Odds']=df_test['l']
        # Make predictions
        y_test_pred = lr.predict(X_test)
        # Store the predictions in the dataframe
        df_pred['HorseWin'] = y_test_pred
        # Save predictions to csv
        pd.DataFrame(df_pred).to_csv('./predictions/deploy_pred.csv')
        # Filter the DataFrame to include only rows where HorseWin = 1
        winning_horses = df_pred[df_pred['HorseWin'] == 1]

        
        # race_id_counts = winning_horses['RaceID'].value_counts()
        # duplicate_race_ids = race_id_counts[race_id_counts > 1].index

        # Step 3: Exclude races with duplicate race_ids
        # filtered_winning_horses = winning_horses[~winning_horses['RaceID'].isin(duplicate_race_ids)]

        # Reset the index of the filtered DataFrame
        filtered_winning_horses = winning_horses.reset_index(drop=True)

        # Display the filtered DataFrame in Streamlit
        st.title("Bet on the races below")
        st.dataframe(filtered_winning_horses)

    # if st.button('PREDICT: N.A. WIN with Q-Learning'):
    #     df_train = pd.read_csv('./data/df_train.csv')
    #     df_train.reset_index(inplace=True, drop=True)
    #     df_test=pd.read_csv('./data/df_test.csv')
    #     df_test.drop(columns=['owner_full', 'owner_l_name', 'owner_f_name'], inplace=True)
    #     # Select the first instance of each runners_horse_name in df_train
    #     df_train_unique = df_train.drop_duplicates(subset='runners_horse_name', keep='first')

    #     # Merge with df_test
    #     df_test = pd.merge(
    #         df_test, 
    #         df_train_unique[['runners_horse_name', 'owner_full']], 
    #         on='runners_horse_name', 
    #         how='left')
    #     print(df_test.head())
       
    #     df_test[['recent_ave_rank', 'jockey_ave_rank', 'trainer_ave_rank']] = df_test.apply(
    #     lambda row: compute_average_ranks(row, df_test, df_train),
    #     axis=1)
    #     df_test.to_csv('./data/df_test_2.csv')
    #     # df_test.to_csv('./data/df_test.csv',index=False)
    #     df_test.reset_index(inplace=True, drop=True)
    #     df_cleaned = df_train.dropna(subset=['morning_line_odds','distance_value', 'runners_post_pos', 'runners_weight','jockey_ave_rank',
    #                         'trainer_ave_rank', 'recent_ave_rank'])
    #     # Reset index
    #     df_train = df_cleaned.reset_index(drop=True)

    #     df_cleaned = df_test.dropna(subset=['morning_line_odds','distance_value', 'runners_post', 'runners_weight','jockey_ave_rank',
    #                         'trainer_ave_rank', 'recent_ave_rank'])
    #     # Reset index
    #     df_test = df_cleaned.reset_index(drop=True)
    #     X_train = df_train[['runners_weight', 'runners_post_pos',
    #                         #'win_odds',
    #                         'jockey_ave_rank',
    #                         'trainer_ave_rank',  'recent_ave_rank', 'distance_value']]
    #     X_test = df_test[['runners_weight', 'runners_post',
    #                         #'win_odds',
    #                         'jockey_ave_rank',
    #                         'trainer_ave_rank', 'recent_ave_rank', 'distance_value']]
    #     X_test = X_test.rename(columns={'runners_post': 'runners_post_pos'})
    #     print("X test is",X_test)
    #     y_train = df_train[['HorseWin']]










