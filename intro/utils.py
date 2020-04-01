import re


def extract_number(string):
    return int(re.search(r'\d+', string).group())
