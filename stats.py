class StatsMan(object):
	pass

class BMStatsMan(StatsMan):
	def __init__(self):
		self.early_stops = 0
		self.discrete_runs = 0

	def info(self):
		print('early stops |', 'discrete runs |', 'quotient', end=': ')
		print(self.early_stops, '|', self.discrete_runs, '|', self.early_stops/self.discrete_runs)