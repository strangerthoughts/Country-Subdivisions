import os
import pandas
import time

from fuzzywuzzy import process
from unidecode import unidecode

REGION_NAME_TABLE = pandas.read_excel("data/Subdivision ISO Codes.xlsx")
ASCII_REGION_NAMES = [unidecode(s) for s in REGION_NAME_TABLE['regionName'].values]
COUNTRY_GROUPS = REGION_NAME_TABLE.groupby(by = 'countryCode')
COUNTRY_GROUPS = {code: [unidecode(s) for s in group['regionName'].values] 
					for code, group in COUNTRY_GROUPS}

def searchForRegion(region_name, country_code = None):
	region_name = unidecode(region_name)

	if country_code is None:
		region_names = ASCII_REGION_NAMES
	else:
		region_names = COUNTRY_GROUPS.get(country_code)
	
	if region_names is None:
		return None
	
	match = process.extractOne(region_name, region_names, limit = 5)

	return match

if __name__ == "__main__":
	test_region = 'Ouémé'
	start = time.time()
	_iter = 10
	for i in range(_iter):
		result = searchForRegion(test_region, 'BEN')
		#result = searchForRegion("Oueme")
	stop = time.time()
	print(stop - start)

	print(result)
