from sqlalchemy import Column, Integer, ForeignKey,Float, String,Boolean,Text, Date, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import Column, Integer, Float, String, Date, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()
Base2=declarative_base()
Base3=declarative_base()
Base4=declarative_base()

class Horse(Base4):
    __tablename__ = 'horses'
    
    horse_id = Column(String, primary_key=True)  # Assuming horse_id is a string
    horse_name = Column(String, nullable=False)
    sire = Column(String)
    sire_id = Column(String)
    dam = Column(String)
    dam_id = Column(String)
    damsire = Column(String)
    damsire_id = Column(String)
    total_runs = Column(Integer, default=0)

    # One-to-many relationship with Distance
    distances = relationship('Distance', back_populates='horse', cascade='all, delete-orphan')

class Time(Base4):
    __tablename__ = 'times'
    
    id = Column(Integer, primary_key=True)
    horse_id = Column(String, ForeignKey('horses.horse_id'), nullable=False)
    distance_id = Column(Integer, ForeignKey('distances.id'), nullable=False)  # Link to Distance model
    date = Column(Date)
    region = Column(String)
    course = Column(String)
    time = Column(String)
    going = Column(String)
    position = Column(String)

    # Relationships to Horse and Distance
    horse = relationship('Horse', back_populates='times')
    distance = relationship('Distance', back_populates='times')


class Distance(Base4):
    __tablename__ = 'distances'
    
    id = Column(Integer, primary_key=True)
    horse_id = Column(String, ForeignKey('horses.horse_id'), nullable=False)
    dist = Column(String)
    dist_y = Column(String)
    dist_m = Column(String)
    dist_f = Column(String)
    runs = Column(Integer, default=0)
    first_place = Column(Integer, default=0)
    second_place = Column(Integer, default=0)
    third_place = Column(Integer, default=0)
    fourth_place = Column(Integer, default=0)
    ae = Column(Integer, default=0)
    win_percentage = Column(Integer, default=0)
    first_place_or_more = Column(Integer, default=0)

    # Relationships
    horse = relationship('Horse', back_populates='distances')
    times = relationship('Time', back_populates='distance', cascade='all, delete-orphan')



class Entry(Base):
    __tablename__ = 'entries'
    id = Column(Integer, primary_key=True)
    meet_id = Column(String)
    track_name = Column(String)
    race_key_race_number = Column(Integer)
    date = Column(Date)
    distance_value = Column(Float)
    decimal_odds = Column(Float)
    morning_line_odds = Column(Float)
    live_odds = Column(Float)
    runners_horse_name = Column(String)
    jockey_f_name = Column(String)
    jockey_l_name = Column(String)
    jockey_full = Column(String)
    trainer_f_name = Column(String)
    trainer_l_name = Column(String)
    trainer_full = Column(String)
    runners_post_pos = Column(String)
    runners_program_numb = Column(String)
    runners_weight = Column(Float)
    __table_args__ = (
        UniqueConstraint('meet_id', 'track_name', 'race_key_race_number', 'date', 'runners_horse_name', name='unique_entry'),
    )
    # timestamp = Column(DateTime, default=func.now(), onupdate=func.now())


class Result(Base):
    __tablename__ = 'results'
    id = Column(Integer, primary_key=True)
    meet_id = Column(String)
    track_name = Column(String)
    race_key_race_number = Column(Integer)
    date = Column(Date)
    owner_f_name=Column(String)
    owner_l_name=Column(String)
    owner_full=Column(String)
    runners_horse_name = Column(String)
    finish_position = Column(Integer)
    place_payoff = Column(Float)
    show_payoff = Column(Float)
    win_payoff = Column(Float)
    __table_args__ = (
        UniqueConstraint('meet_id', 'track_name', 'race_key_race_number', 'date', 'runners_horse_name', name='unique_entry'),
    )
    # timestamp = Column(DateTime, default=func.now(), onupdate=func.now())


class MergedData(Base):
    __tablename__ = 'merged_data'
    id = Column(Integer, primary_key=True)
    meet_id = Column(String)
    track_name = Column(String)
    race_key_race_number = Column(Integer)
    date = Column(Date)
    distance_value = Column(Float)
    decimal_odds = Column(Float)
    morning_line_odds = Column(Float)
    live_odds = Column(Float)
    runners_horse_name = Column(String)
    jockey_full = Column(String)
    trainer_full = Column(String)
    finish_position = Column(Integer)
    place_payoff = Column(Float)
    show_payoff = Column(Float)
    win_payoff = Column(Float)
    # timestamp = Column(DateTime, default=func.now(), onupdate=func.now())




#FOR RESULTS API ######################################
class Runner(Base2):
    __tablename__ = 'runners'
    id = Column(Integer, primary_key=True)
    horse_id = Column(String, nullable=True)  # Can be empty or null
    horse = Column(String, nullable=True)  # Renamed from 'horse'
    sp = Column(String, nullable=True)
    sp_dec = Column(String, nullable=True)  # Kept as String
    number = Column(String, nullable=True)
    position = Column(String, nullable=True)  # Keeping this as String
    draw = Column(String, nullable=True)
    btn = Column(String, nullable=True)
    ovr_btn = Column(String, nullable=True)
    age = Column(String, nullable=True)  # Consider using Integer if it's always a number
    sex = Column(String, nullable=True)
    weight = Column(String, nullable=True)  # Changed to Float for numeric operations
    weight_lbs = Column(String, nullable=True)  # Changed to Float
    headgear = Column(String, nullable=True)
    time = Column(String, nullable=True)
    or_rating = Column('or',String, nullable=True)  # Changed to Integer
    rpr = Column(String, nullable=True)  # Changed to Integer
    tsr = Column(String, nullable=True)  # Changed to Integer
    prize = Column(String, nullable=True)  # Changed to Float
    jockey = Column(String, nullable=True)
    jockey_claim_lbs = Column(Float, nullable=True)  # Changed to Float
    jockey_id = Column(String, nullable=True)
    trainer = Column(String, nullable=True)
    trainer_id = Column(String, nullable=True)
    owner = Column(String, nullable=True)
    owner_id = Column(String, nullable=True)
    sire = Column(String, nullable=True)
    sire_id = Column(String, nullable=True)
    dam = Column(String, nullable=True)
    dam_id = Column(String, nullable=True)
    damsire = Column(String, nullable=True)
    damsire_id = Column(String, nullable=True)
    silk_url = Column(String, nullable=True)

    # Foreign key relationship
    race_id = Column(Integer, ForeignKey('races.race_id'), nullable=True)
    race = relationship("Race", back_populates="runners")
    __table_args__ = (
        UniqueConstraint('horse_id', 'race_id', name='uq_horse_race'),
    )

class Race(Base2):
    __tablename__ = 'races'
    id = Column(Integer, primary_key=True)
    race_id = Column(String, nullable=True)  # Keeping as String
    date = Column(String, nullable=True)  # Changed to String
    region = Column(String, nullable=True)
    course = Column(String, nullable=True)
    course_id = Column(String, nullable=True)
    off = Column(String, nullable=True)
    off_dt = Column(String, nullable=True)
    race_name = Column(String, nullable=True)
    race_type = Column('type', String, nullable=True)  # Renamed from 'type'
    race_class = Column('class', String, nullable=True)
    pattern = Column(String, nullable=True)
    rating_band = Column(String, nullable=True)
    age_band = Column(String, nullable=True)
    sex_rest = Column(String, nullable=True)
    dist = Column(String, nullable=True)  # Changed to Float
    dist_y = Column(String, nullable=True)  # Changed to Float
    dist_m = Column(String, nullable=True)  # Changed to Float
    dist_f = Column(String, nullable=True)  # Changed to Float
    going = Column(String, nullable=True)
    jumps = Column(String, nullable=True)  # Assuming string representation
    comments = Column(String, nullable=True)
    # Additional fields based on the JSON structure
    winning_time_detail = Column(String, nullable=True)
    non_runners = Column(String, nullable=True)
    tote_win = Column(String, nullable=True)  # Changed to Float
    tote_pl = Column(String, nullable=True)  # Changed to Float
    tote_ex = Column(String, nullable=True)  # Changed to Float
    tote_csf = Column(String, nullable=True)  # Changed to Float
    tote_tricast = Column(String, nullable=True)  # Changed to Float
    tote_trifecta = Column(String, nullable=True)  # Changed to Float

    runners = relationship("Runner", back_populates="race")


############ FOR RACECARD API #########################

class Racecard(Base3):
    __tablename__ = 'racecards'

    id = Column(Integer, primary_key=True)
    race_id = Column(String, nullable=False)
    course = Column(String)
    course_id = Column(String)
    date = Column(Date)  # Use Date for date values
    off_time = Column(String)
    off_dt = Column(String)
    race_name = Column(String)
    distance_round = Column(String)
    distance = Column(String)  # Consider Numeric type if needed
    distance_f = Column(String)  # Consider Numeric type if needed
    region = Column(String)
    pattern = Column(String)
    race_class = Column(String)
    type = Column(String)
    age_band = Column(String)
    rating_band = Column(String)
    prize = Column(String)  # Consider Numeric type if needed
    field_size = Column(String)  # Consider Numeric type if needed
    going_detailed = Column(String)
    rail_movements = Column(String)
    stalls = Column(String)
    weather = Column(String)
    going = Column(String)
    surface = Column(String)
    jumps = Column(String)  # Adjust type as needed
    big_race = Column(Boolean, default=False)
    is_abandoned = Column(Boolean, default=False)

    runners = relationship('Runners', back_populates='racecard')


class Runners(Base3):
    __tablename__ = 'runners2'

    id = Column(Integer, primary_key=True)
    horse_id = Column(String)
    horse = Column(String)
    dob = Column(String)
    age = Column(String)
    sex = Column(String)
    sex_code = Column(String)
    colour = Column(String)
    region = Column(String)
    breeder = Column(String)
    dam = Column(String)
    dam_id = Column(String)
    dam_region = Column(String)
    sire = Column(String)
    sire_id = Column(String)
    sire_region = Column(String)
    damsire = Column(String)
    damsire_id = Column(String)
    damsire_region = Column(String)
    trainer = Column(String)
    trainer_id = Column(String)
    trainer_location = Column(String)
    trainer_14_days = Column(String)  # Consider JSON type if supported
    owner = Column(String)
    owner_id = Column(String)
    comment = Column(String)
    spotlight = Column(String)
    number = Column(String)
    draw = Column(String)
    headgear = Column(String)
    headgear_run = Column(String)
    wind_surgery = Column(String)
    wind_surgery_run = Column(String)
    lbs = Column(String)
    ofr = Column(String)
    rpr = Column(String)
    ts = Column(String)
    jockey = Column(String)
    jockey_id = Column(String)
    silk_url = Column(String)
    last_run = Column(String)
    form = Column(String)
    trainer_rtf = Column(String)
    
    race_id = Column(String, ForeignKey('racecards.race_id'))
    racecard = relationship('Racecard', back_populates='runners')



