from time import sleep
from datetime import datetime
from sklearn import tree
from utils import *
from constants import WORKING_TIME_START, WORKING_TIME_END, WEEKDAYS


DELAY = 5

data = read_data()
lines = data.splitlines()
dates = parse_dates(lines)

X, y = training_set()
model = tree.DecisionTreeClassifier()
model = model.fit(X, y)


def delay(n, dots=3):
    for _ in range(n):
        sleep(1)
        print('.' * dots)


if __name__ == '__main__':
    # create test instance from user input
    today = datetime.now()
    date = '{}-{}-{}'.format(today.year, today.month, today.day)
    hour = int(input(
        '오늘 걱정되는 시간대가 언제입니까? [{}, {}]: '.format(
        WORKING_TIME_START, WORKING_TIME_END
    )))
    time = normalize_time(hour)
    weekday = WEEKDAYS[today.weekday()]

    # receive oracle
    x = make_feature_from_normalized_time(date, time, weekday, dates)
    print('신탁을 받는 중입니다...')
    delay(DELAY)
    oracle = model.predict([x])[0]

    # tell user the oracle
    print('오늘 당신의 상사는 {}'.format(
        '출근합니다' if oracle else '출근하지 않습니다'
    ))
