class SOFTParser(object):
	def __init__(self, fname):
		self.header = {}

		with open(fname, 'r') as fd:
			self.content = fd.read()

	def parse(self):
		section = None
		dataset = None

		for line in self.content.split('\n'):
			fields = line.split('=')

			if line.startswith('^'):
				section = fields[0].strip().lower()[1:]

				if section == 'dataset':
					dataset = fields[1].strip()
					
					if not 'dataset' in self.header:
						self.header['dataset'] = {}
						self.header['dataset']['name'] = dataset
						self.header['dataset']['subsets'] = []
				elif section == 'subset':
					self.header['dataset']['subsets'].append({})
				else:
					self.header[section] = {}
			elif line.startswith('!'):
				if '=' in line:
					key = fields[0].strip()[1:]
					value = fields[1].strip()

					if section == 'subset':
						access = self.header['dataset']['subsets'][-1]
					else:
						access = self.header[section]

					access[key] = value
			else:
				break

		#from pprint import pprint
		#pprint(self.header)


if __name__ == '__main__':
	parser = SOFTParser('../data/concentrations/GDS2590.soft')
	parser.parse()