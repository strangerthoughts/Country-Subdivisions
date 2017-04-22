import os
import pandas
import math
import csv
from pprint import pprint
from prettytable import PrettyTable
DATA_FOLDER = os.path.join(os.getcwd(), 'data')
DATA_FOLDER = os.getcwd()

def readTSV(filename, headers = False):

	with open(filename, 'r') as file1:
		reader = csv.DictReader(file1, delimiter = "\t")
		fieldnames = reader.fieldnames
		reader = list(reader)

	if headers:
		return reader, fieldnames
	else:
		return reader
def writeTSV(table, filename, fieldnames = None):
	if fieldnames is None: fieldnames = sorted(table[0].keys())
	with open(filename, 'w', newline = "") as file1:
		writer = csv.DictWriter(file1, delimiter = '\t', fieldnames = fieldnames)
		writer.writeheader()
		writer.writerows(table)
def tonum(x):
	if isinstance(x, str) and x != "":
		if '.' in x: value = float(x)
		else:
			try: value = int(x)
			except: value = None
	elif isinstance(x, float) and not math.isnan(x):
		value = x
	elif isinstance(x, int):
		value = x
	else:
		value = None

	return value
#https://en.wikipedia.org/wiki/Gini_coefficient

class SubdivisionDatabase:
	def __init__(self):
		self.datasets = self._loadDatasets()
		self.metadata = self._get_metadata(self.datasets)
		
	def __call__(self, country = None, region_type = None, splitsets = False):
		if country is None: df = self.datasets
		else:
			df = [i for i in self.datasets if i['countryName'] == country]
		if region_type is not None:
			df = [i for i in df if i['regionType'] == region_type]


		if splitsets:
			df = self.splitSets(df)


		return df
		
	def splitSets(self, dataset):
		regions = list()
		unique_regions = set((row['countryName'], row['regionType']) for row in dataset)
		for country, region_type in unique_regions:
			subdivision = [row for row in dataset if (row['countryName'] == country and row['regionType'] == region_type)]
			regions.append(subdivision)
		return regions

	def _get_metadata(self, datasets):
		regions = dict()
		for row in datasets:
			country = row['countryName']
			region_type = row['regionType']
			key = country + '|' + region_type
			if key not in regions:
				regions[key] = 0
			regions[key] += 1
		unique_regions = list()
		for region, count in regions.items():
			country, region_type = region.split('|')
			m = {
				'countryName': country,
				'regionType': region_type,
				'count': count
			}
			unique_regions.append(m)
		return unique_regions

	def _loadDatasets(self):
		configs = self._retrieveDatasetConfiguration()

		datasets = list()
		for config in configs:
			datasets += self.parseDataset(config)
		return datasets
	
	def _openDataset(self, filename, configuration):
		ext = os.path.splitext(filename)[1]

		if ext in {'.csv', '.tsv'}:
			data = readTSV(filename)
		elif ext in {'.xls', '.xlsx'}:
			data = pandas.read_excel(filename, sheetname = configuration['sheetname'])
			data = [row for index, row in data.iterrows()]
		else:
			message = "Unsupported extension: " + ext
			raise ValueError(message)

		return data
	def _getDefaultConfiguration(self, kwargs):
		kwargs['area_col'] 			= kwargs.get('area_col', 	'totalArea')
		kwargs['area_multiplier'] 	= kwargs.get('area_multiplier', 1)
		kwargs['country_name'] 		= kwargs.get('country_name', 'countryName')
		kwargs['country_code'] 		= kwargs.get('country_code', 'countryCode')
		kwargs['density_col'] 		= kwargs.get('density_col', 'totalPopulationDensity')
		kwargs['ensemble'] = kwargs.get('ensemble', False)
		kwargs['exclude']			= kwargs.get('exclude', [])
		kwargs['pop_col'] 			= kwargs.get('pop_col', 'totalPopulation')
		kwargs['region_code'] 		= kwargs.get('region_code', 'regionCode')
		kwargs['region_name']		= kwargs.get('region_name', 'regionName')
		kwargs['region_type'] 		= kwargs.get('region_type', 'regionType')
		kwargs['sheetname'] 		= kwargs.get("sheetname", 0)
		kwargs['skiprows'] 			= kwargs.get('skiprows', 0)
		kwargs['source']			= kwargs.get('source', "")
		kwargs['url']				= kwargs.get("url", "")
		return kwargs
	def parseDataset(self, configuration):
		""" Opens and parses a dataset.
			Required Keys
			-------------
				filename: string
					The filename of the dataset.

			Optional Keys
			-------------
				url: string; default ""
					A url that points to the specific website the data
					was downloaded from.
				source: string; default ""
					The institution that provided the data.
				area_col: string; default 'totalArea'
					The column containing the area of each subdivision.
					km2 is preferred, but other units may be used in the 
					'area_multiplier' column is included.
				area_multiplayer: number; default 1
					Required to convert the areas in the area column to km2
				base_year: int, string; default 'baseYear'
					Either the year the data represents, or a column containing
					that information.
				country_name: string; default 'countryName'
					This should be eithr the name of the country for the entire dataset,
					or a column containing the parent country for each subdivision.
				country_code: string; default 'countryCode'
					Behaves similarly to country_name, but with each country's ISO code.
				ensemble: bool; default False
					Indicates whether the dataset contains subdivisions from a single
					country or multiple countries.
				pop_col: string; default 'totalPopulation'
					Column containing the total population for the region.
				pop_density_col: string; default 'totalPopulationDensity'
					If not provided or the dataset is missing the column,
					the column will be generated based on the population and area
					columns.
				region_code: string; default 'regionCode'
				region_name: string; default 'regionName'
				region_type: string; default 'regionType'
					The local name for the subdivision type.
					Ex. County, Commune, NUTS-2
				sheetname: int, string; default 0
					Indicates which sheet to use when the dataset is an excel spreadsheet.
					Required if the data is not in the first sheet.
		"""
		configuration = self._getDefaultConfiguration(configuration)
		dataset = self._openDataset(configuration['filename'], configuration)
		dataset = self._cleanDataset(dataset, configuration)
		return dataset
	def _cleanDataset(self, dataset, variables):
		"""	Harmonizes the available columns in the dataset.
		"""
		regions = list()
		for row in dataset:

			if len(variables['exclude']) > 0:
				if row['regionType'] in variables['exclude']: continue
			pop = tonum(row[variables['pop_col']])
			area= tonum(row[variables['area_col']])

			if area is None or area == 0.0: continue
			else: area = area * variables['area_multiplier']
			if pop is None: pop = 0

			density = row.get(variables['density_col'], pop / area)
			density = tonum(density)
			if 'base_year' in variables.keys():
				base_year = row.get(variables['base_year'], variables['base_year'])
			else:
				base_year = ""
			element = {
				'totalPopulation': pop,
				'totalArea': 	area,
				'totalDensity':  density,
				'year': 	base_year,
				'regionCode': 	row.get(variables['region_code'], variables['region_code']),
				'regionType': 	row.get(variables['region_type'], variables['region_type']),
				'countryCode': 	row.get(variables['country_code'], variables['country_code']),
				'countryName': 	row.get(variables['country_name'], variables['country_name']),
				'regionName': 	row.get(variables['region_name'], variables['region_name']),
				'url': 		variables['url'],
				'source': variables['source']
			}
			regions.append(element)
		return regions

	def _retrieveDatasetConfiguration(self):
		dataset_configurations = [
			{
				'filename': "data/Canadian Census Areas.tsv",
				'url': "",
				'source': "",

				'area_col': "Land area in square kilometres, 2011",
				'country_code': "CAN",
				'country_name': "Canada",
				'pop_col': "Population, 2011",
				'region_code': "Geographic code",
				'region_name': "Geographic name",
				'region_type': "Census Tract"
			},
			{
				'filename': "data/United States/Population by State.tsv",
				'url': "https://www.census.gov/geo/maps-data/data/tiger-data.html",
				'source': "US Census Bureau",

				'area_col': "landArea",
				'region_type': "State"
			},
			{
				'filename': "data/United States/Population by 111th Congressional Districts.tsv",
				'url': "https://www.census.gov/geo/maps-data/data/tiger-data.html",
				'source': "US Census Bureau",

				'area_col': "landArea",
				'region_type': "Congressional District"
			},
			{
				'filename': "data/United States/County Subdivisions.xlsx",
				'source': "US Census Bureau",
				'url': "",

				'area_multiplier': 1.602**2,
				'country_code': 'USA',
				'country_name': 'United States',
				'region_type': "County"
			},
			{
				'filename': "data/United States/Population by ZIP Code Dataset.tsv",
				'url': "https://www.census.gov/geo/maps-data/data/tiger-data.html",
				'source': "US Census Bureau",
				'area_col': 'landArea',
				'region_type': 'ZIP Code'
			},
			{
				'filename': "data/United States/Population by Census Tract.tsv",
				'url': "https://www.census.gov/geo/maps-data/data/tiger-data.html",
				'source': "US Census Bureau",
				'area_col': 'landArea',
				'region_type': "Census Tract"
			},
			{
				'filename': "data/French Commune Database.tsv",
				'url': "http://professionnels.ign.fr/geofla#tab-3",
				'source': "L'information Grandeur Nature",
				
				'area_col': "SUPERFICIE",
				'area_multiplier': .01,
				'country_name': "France",
				'country_code': "FRA",
				'base_year': 2016,
				'pop_col': 'POPULATION',
				'region_code': "INSEE_CODE",
				'region_name': "NOM_COM",
				'region_type': "Commune"
			},
			{
				'filename': "data/United Kingdom/UK Localities.tsv",
				'url': "https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland",
				'source': "Office for National Statistics",

				'base_year': 2015,
				'countryCode': "GBR",
				'countryName': "United Kingdom",
				'exclude': ['metropolitan county']
			},
			{
			'filename': 'data/Combined Country Subdivision Table.xlsx',
			'source': '',
			'url': '',
			'split_region_types': True,
			'ensemble': True
			}
		]

		return dataset_configurations


class PopulationDensity:
	def __init__(self, regions, region_type = None):
		"""
			Parameters
			----------
				regions: list<dict<>>
					list of regions.
					* 'totalPopulation': int
					* 'totalArea': float
					* 'countryName': string
					* 'regionCode': string
					* 'countryCode'
		"""

		if isinstance(regions, pandas.DataFrame):
			regions = list(regions.T.to_dict().values())
		#pprint(regions)
		if region_type is None:
			region_type = regions[0]['regionType']
		
		self.country = regions[0]['countryName']
		
		self.population_column = 'totalPopulation'
		self.area_column = 'totalArea'
		self.density_column = 'totalDensity'

		#regions = list(self.addDensity(regions))

		self.population_density   = self.calculatePopulationDensity(regions, region_type = regions[0]['regionType'])
		#pprint(self.population_density)


	def calculateGini(self, seq):
		""" Requires equally-sized bins (i.e. equal area counties)
		"""
		sorted_list = sorted(seq)
		height, area = 0, 0
		for value in sorted_list:
			height += value
			area += height - value / 2.
		fair_area = height * len(seq) / 2.
		return (fair_area - area) / fair_area
	def addDensity(self, regions):
		for row in regions:
			#print(row)
			if self.density_column not in row.items():
				pop = tonum(row[self.population_column])
				area = tonum(row[self.area_column])
				if area is None: continue
				if pop is None: pop = 0
				density = pop / area
				row[self.density_column] = density
				yield row
	def calculatePopulationDensity(self, regions, region_type, threshold = .90, top = True, DEBUG = False):
		""" Retrieves the top n % of regions by population, area, or density
			Parameters
			----------
				regions: list<dict<>>
					list of region values
				threshold: float [0,1]
				top: bool; default True
					Whether to grab the densist or least dense regions.
		"""
		regions = sorted(regions, key = lambda s: s[self.density_column])
		#regions = sorted(regions, key = lambda r: r[self.population_column] / r[self.area_column])
		if top:
			regions = list(reversed(regions))

		total_population = int(sum([r[self.population_column] for r in regions]))
		total_area = sum(r[self.area_column] for r in regions)
		population_limit = total_population * threshold
		
		if DEBUG:
			print("\t#Num Regions: ", len(regions))
			print("\tTotal: ", total)
			print("\tLimit: ", population_limit)
			
		
		subdivision_population = 0
		subdivision_area = 0.0
		subdivisions = list()
		previous_difference = population_limit
		for index, region in enumerate(regions):
			region_population = region[self.population_column]
			region_area = region[self.area_column]
			
			current_population = subdivision_population + region_population
			current_area = subdivision_area + region_area
			current_difference = abs(population_limit - current_population)
			#print(current_ratio, previous_ratio_difference, current_ratio_difference)
			if current_difference < previous_difference:
				subdivisions.append(region)

				subdivision_population = current_population
				subdivision_area = current_area
				previous_difference = current_difference
			else:
				break
			#if subdivision_population >= population_limit: break

		total_density = total_population / total_area
		subdivision_density = subdivision_population / subdivision_area
		mean_area = subdivision_area / len(subdivisions)

		result = {
			'country': self.country,
			'meanArea': mean_area,
			'regionType': region_type,
			'ratioDensity': subdivision_density / total_density,
			'ratioPopulation': 100 * subdivision_population / total_population,
			'totalPopulation': int(total_population),
			'totalPopulationDensity': total_density,
			'totalRegions': len(regions),
			'totalArea': total_area,
			'subPopulation': int(subdivision_population),
			'subRegions': len(subdivisions),
			'subArea': subdivision_area,
			'subPopulationDensity': subdivision_density
		}

		return result

	def export(self):

		rows = [v for k, v in sorted(self.population_density.items())]

		 #rows.append([self.partial_density[i] for i in ['country', 'regionType', 'totalPopulation', 'totalRegions', 'populationDensity']])
		return rows

class ParseDataset:
	def __init__(self, **kwargs):
		"""
			Required Keywords
			-----------------
				filename: string

			Optional Keywords
			-----------------
				area_col: string; default 'totalArea'
				area_multiplier: number; default 1
				country_name: string; default 'countryName'
				country_code: string; default 'countryCode'
				pop_col: string; default 'totalPopulation'
				region_code: string; default 'regionCode'
				region_name: string; default 'regionName'
				region_type: string; default 'regionType'
				skiprows: int; default 0
				split_region_types: bool; default False
				url: string; default ""
				source: string; default ""
		"""

		if 'kwargs' in kwargs.keys(): kwargs = kwargs['kwargs']
		kwargs = self.setDefaultVars(kwargs)

		#pprint(kwargs)
		kwargs['filename'] = os.path.join(DATA_FOLDER, kwargs['filename'])
		print(kwargs['filename'])
		self.variables = kwargs
		#self.setVars()
		ext = os.path.splitext(kwargs['filename'])[1].lower()
		if ext == '.xls' or ext == '.xlsx':
			data = pandas.read_excel(kwargs['filename'], skiprows = kwargs['skiprows'], sheetname = kwargs['sheetname'])
		elif ext == '.tsv':
			data = readTSV(kwargs['filename'])
			data = pandas.DataFrame(data)
		elif ext == '.csv':
			data = pandas.read_csv(kwargs['filename'], skiprows = kwargs['skiprows'])

		self.data = self.parse(data, self.variables)

	def parse(self, data, variables):
		regions = list()
		for index, row in data.iterrows():
			pop = tonum(row[variables['pop_col']])
			area= tonum(row[variables['area_col']])

			if area is None or area == 0.0: continue
			else: area = area * variables['area_multiplier']
			if pop is None: pop = 0
			density = pop / area

			try: base_year = variables['base_year']
			except: base_year = None

			element = {
				'population': pop,
				'area': 	area,
				'density':  density,
				'year': 	base_year,
				'regionCode': 	row.get(variables['region_code'], variables['region_code']),
				'regionType': 	row.get(variables['region_type'], variables['region_type']),
				'countryCode': 	row.get(variables['country_code'], variables['country_code']),
				'countryName': 	row.get(variables['country_name'], variables['country_name']),
				'regionName': 	row.get(variables['region_name'], variables['region_name']),
				'url': 		variables['url'],
				'source': variables['source']
			}
			regions.append(element)
		return regions
	def export(self, region_types = None):
		#Split countries and region types
		if self.variables['split_region_types']:
			result = list()
			unique_countries = set(i['countryName'] for i in self.data)
			if region_types is None: region_types = set(i['regionType'] for i in self.data)
			elif isinstance(region_types, str): region_types = [region_types]
			
			for country in unique_countries:
				for rt in region_types:
					subregions = [i for i in self.data if 
						(i['countryName'] == country and i['regionType'] == rt)]
					result.append(subregions)
		else:
			result = self.data
		return result
	def setDefaultVars(self, kwargs):

		kwargs['skiprows'] 			= kwargs.get('skiprows', 0)
		kwargs['pop_col'] 			= kwargs.get('pop_col', 'totalPopulation')
		kwargs['region_code'] 		= kwargs.get('region_code', 'regionCode')
		kwargs['region_name']		= kwargs.get('region_name', 'regionName')
		kwargs['region_type'] 		= kwargs.get('region_type', 'regionType')
		kwargs['area_col'] 			= kwargs.get('area_col', 'totalArea')
		kwargs['country_name'] 		= kwargs.get('country_name', 'countryName')
		kwargs['country_code'] 		= kwargs.get('country_code', 'countryCode')
		kwargs['area_multiplier'] 	= kwargs.get('area_multiplier', 1)
		kwargs['source']			= kwargs.get('source', "")
		kwargs['split_region_types'] = kwargs.get('split_region_types', False)
		kwargs['url']				= kwargs.get("url", "")
		kwargs['sheetname'] 		= kwargs.get("sheetname", 0)

		return kwargs

def mapIso(filename):
	""" """
	from fuzzywuzzy import process
	df = pandas.read_excel(filename)
	candidates = list()
	matches = list()
	for index, row in df.iterrows():
		name = row['regionName']
		match = process.extractOne(name, candidates)
		match = {
			'originalName': name,
			'matchedName': match[0],
			'score': match[1]
		}
		matches.append(match)



if __name__ == "__main__" and True:
	if True:
		datasets = SubdivisionDatabase()
	else:

		datasets = [ParseDataset(kwargs = d).export() for d in single_datasets]
		datasets += ParseDataset(kwargs = multiple_datasets[0]).export()

	densities = [PopulationDensity(i) for i in datasets(splitsets = True) if len(i) > 10]


	fieldnames = sorted(densities[0].population_density.keys())
	table = PrettyTable(field_names = fieldnames)
	for col in fieldnames:
		table.align[col] = 'r'
	table.float_format = ".2"

	print("Creating table for {0} rows".format(len(densities)))
	for row in densities:
		r = row.export()
		try:
			print("Adding ", r[0])
			table.add_row(r)
		except Exception as exception:
			print("Exception: ", exception)
	table.sortby = "country"
	print(str(table))

elif False:

	def shapefileToTable(folder, filename, sortby = None):
		""" Converts the fields of a shapefile into a tsv table """
		import shapefile
		shape_file = shapefile.Reader(os.path.join(folder, os.path.basename(folder)))

		shapes = sf.shapes()
		records = sf.records()
		table = list()
		for index, record in enumerate(records):
			row = {f:v for f, v in zip(df.fields[1:], record)}
			table.append(row)

		if sortby is not None:
			table = sorted(table, key = lambda s: s[sortby])
		writeTSV(table, filename = filename)

	import shapefile
	folder 		= ""
	filename 	= ""
	shapefileToTable(folder, filename)












