MONTH_TOTAL_DAYS = 31

SEASON_TO_MONTHS = {
    'SPRING': ['3', '4', '5'],
    'SUMMER': ['6', '7', '8'],
    'FALL': ['9', '10', '11'],
    'WINTER': ['12', '1', '2'],
}

MONTH_TO_SEASON = {}
for season, months in SEASON_TO_MONTHS.items():
    for month in months:
        MONTH_TO_SEASON[month] = season

WEEKDAYS = ['월', '화', '수', '목', '금', '토', '일']

WORKING_TIME_START = 9
WORKING_TIME_END = 18
WORKING_TIME_TOTAL_MINUTES = (WORKING_TIME_END - WORKING_TIME_START) * 60
