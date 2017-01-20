import random
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn import utils as sklearn_utils
from constants import (
    MONTH_TOTAL_DAYS,
    MONTH_TO_SEASON, 
    WORKING_TIME_START,
    WORKING_TIME_END,
    WORKING_TIME_TOTAL_MINUTES,
    SEASON_TO_MONTHS,
    WEEKDAYS,
)


# ===========
# Auxiliaries
# ===========

def parse_dates(lines):
    return [l.split(',')[0].split(' ')[0] for l in lines if l.startswith('2')]


def normalize_time(hour, minutes=0):
    return (
        ((hour - WORKING_TIME_START) * 60 + minutes) / 
        WORKING_TIME_TOTAL_MINUTES
    )


def date_to_datetime(date):
    year, month, day = date.split('-')
    year, month, day = int(year), int(month), int(day)
    return datetime(year, month, day)


# =======
# Parsers
# =======

def parse_and_hot_encode_season(date):
    """Parse season from date and return hot-encoded season vector"""
    splits = date.split('-')
    _, month, _, *rest = splits
    lb = LabelBinarizer()
    lb.fit(list(SEASON_TO_MONTHS.keys()))
    return lb.transform([MONTH_TO_SEASON[month]])[0]


def parse_and_normalize_day(date):
    """Parse day of month from date and normalize to real value in [0, 1]"""
    splits = date.split('-')
    _, _, day, *rest = splits
    return int(day) / MONTH_TOTAL_DAYS


def parse_and_normalize_time(noon, time):
    """Parse minutes of day from time and normalize to real value in [0, 1]"""
    splits = time.split(':')
    hours, minutes, *rest = splits
    hours, minutes = int(hours), int(minutes)
    if noon == '오후' and hours != 12:
        hours += 12
    return normalize_time(hours, minutes)


def hot_encode_weekday(weekday):
    """Returns a hot-encoded weekday vector"""
    lb = LabelBinarizer()
    lb.fit(WEEKDAYS)
    return lb.transform([weekday])[0]


def make_feature(date, noon, time, weekday, dates=None, adjacency=4):
    return np.array([        
        *parse_and_hot_encode_season(date),
        parse_and_normalize_day(date),
        parse_and_normalize_time(noon, time),
        *hot_encode_weekday(weekday),
        get_normalized_adjecent_visits_in(date, dates, adjacency=adjacency)
    ])


def make_feature_from_normalized_time(date, time, weekday, dates, adjacency=4):
    return np.array([        
        *parse_and_hot_encode_season(date),
        parse_and_normalize_day(date), time,
        *hot_encode_weekday(weekday),
        get_normalized_adjecent_visits_in(date, dates, adjacency=adjacency)
    ])


def parse_line(l, dates=None, adjacency=4):
    """Parse features from a valid csv record line. 
    
    Parsed features will be:

    - hot-encoded season: R^4
    - normalized date: [0, 1] R^1
    - normalized time: [0, 1] R^1
    - hot-encoded weekday: R^7
    - normalized adjecent visits: [0, 1] R^1

    Note that these features will be rolled into a single vector of R^14.

    """
    splits = l.split(',')
    datetime, weekday, *rest = splits
    date, noon, time, *rest = datetime.split(' ')
    return make_feature(date, noon, time, weekday, dates=dates, 
                        adjacency=adjacency)


# ===============
# Data utilities
# ===============

def get_normalized_adjecent_visits_in(date, dates=None, adjacency=4):
    date = date_to_datetime(date)
    dates = [date_to_datetime(d) for d in dates or []]
    deltas = [date - d for d in dates]
    valid_delta = (
        lambda delta: (
            delta >= timedelta(days=1) and 
            delta <= timedelta(days=adjacency)
        )
    )
    return len([delta for delta in deltas if valid_delta(delta)]) / adjacency


def generate_positive_training_set(lines, dates=None, adjacency=4):
    X = np.array([parse_line(l, dates, adjacency=adjacency) for l in lines])
    y = np.ones(X.shape[0])
    return X, y


def generate_negative_training_set(positive_dates, adjacency=4, sample_size=3):
    start, *rest, end = [date_to_datetime(d) for d in positive_dates]
    delta = timedelta(days=1)

    # start finding missing dates and generate negative training instance.
    X = None
    y = None
    dt = start + delta
    while dt <= end:
        date = '{}-{}-{}'.format(dt.year, dt.month, dt.day)
        weekday = WEEKDAYS[dt.weekday()]
        if date not in positive_dates:
            for _ in range(sample_size):
                hours = random.randint(WORKING_TIME_START, WORKING_TIME_END)
                normalized_time = normalize_time(hours, 0)
                features = make_feature_from_normalized_time(
                    date, normalized_time, weekday, dates=positive_dates, 
                    adjacency=adjacency
                )
                X = np.vstack([X, features]) if X is not None else np.array(features)
                y = np.append(y, 0) if y is not None else np.array([0])
        dt += delta
    return X, y


def read_data():
    with open('./data.csv', 'r', encoding='utf-8') as f:
        return f.read()

def training_set(adjacency=4, negative_sample_size=1, shuffle=True):
    # load data
    data = read_data()
    lines = [l for l in data.splitlines() if l.startswith('2')]
    dates = parse_dates(lines)

    positive_X, positive_y = generate_positive_training_set(
        lines, dates=dates, adjacency=adjacency
    )

    negative_X, negative_y = generate_negative_training_set(
        dates, adjacency=adjacency, sample_size=negative_sample_size
    )

    X = np.vstack([positive_X, negative_X])
    y = np.append(positive_y,  negative_y)

    return sklearn_utils.shuffle(X, y)
