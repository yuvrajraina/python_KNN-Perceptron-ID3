import numpy as np


class KNN:
	def __init__(self, k):
		#KNN state here
		#Feel free to add methods
		self.k = k

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		#training logic here
		#input is an array of features and labels
		self.features = X
		self.labels = y
		None

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		
		prediction = np.array([])
		for feature in X:
			distance = np.array([self.distance(feature,f) for f in self.features])
			best = sorted(self.zipper(distance), key=lambda x: x[0])[:self.k]
			predict = self.select([x[1] for x in best])
			prediction = np.append(prediction, [predict])	
			
		return prediction
	
	def zipper(self, distance):
		return zip(distance, self.labels)
		
	def select(self, neighbours):
		total = 0
		best = None
		for n in neighbours:
			if total == 0:
				best = n
			if n == best:
				total += 1 
			else:
				total -= 1
		return best


class Perceptron:
	def __init__(self, w, b, lr):
		#Perceptron state here, input initial weight matrix
		#Feel free to add methods
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		#training logic here
		#input is array of features and labels
		for step in range(steps):
			temp = step % len(y)
			#print('activation', activation)
			if np.dot(self.w,X[temp]) + (self.b)>0:
				prob = 1
			else:
				prob = 0
			
			error= y[temp] - prob
			self.w += X[temp] * error * self.lr 
		#None

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		predictions = []
		for step in range(len(X)):
			
			if np.dot(self.w,X[step]) + (self.b)>0:
				prob = 1
			else:
				prob = 0
			predictions.append(prob)
		return np.ravel(predictions)


class ID3:
	
	class Tree(object):
		def __init__(self,attr):
			self.attr = attr
			self.parent = None
			self.child = {}
		def addChild(self,key,child):
			self.child[key] = child
			if type(child) == type(self):
				self.child[key].parent = self
		

	class Node(object):
		def __init__(self,id,data,value):
			self.id = id
			self.data = data
			self.value = value
		
		def __str__(self):
			string = " "
			if self.parent != None:
				string += str(self.parent.attr)+" ====>> "+str(self.attr)
			else:
				string += str(self.attr)
			for branch in self.branches:
				if type(self.branches[branch]) == type(self):
					string += "\n  branch->: "+str(branch)+"  attr->: "+str(self.branches[branch].attr)
				else:
					string += "\n  branch->: "+str(branch)+"  out->: "+str(self.branches[branch])
			return string

	def __init__(self, nbins, data_range):
		#Decision tree state here
		#Feel free to add methods
		self.bin_size = nbins
		self.range = data_range

	def preprocess(self, data):
		#Our dataset only has continuous data
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data

	def train(self, X, y):
		#training logic here
		#input is array of features and labels
		categorical_data = self.preprocess(X)
		attributes = np.arange(np.size(categorical_data,1))
		box = []
		for d in range(len(categorical_data)):
			box.append(self.Node(d,categorical_data[d],y[d]))
		self.tree = self.tree(box,attributes,None)

	def tree(self,box,attributes,parent_samples):
		if box == None:
			return self.getPlurality(parent_samples)
		lastValue = box[0].value
		flag =True
		for i in box:
			if i.value != lastValue:
				flag = False	
		if flag ==True:
			return box[-1].value
		
			
			
		elif len(attributes)==0:
			return self.getPlurality(box)
		else:
			best = None
			for attr in attributes:
				current = self.getGain(attr,box)
				if best == None or current > best[1]:
					best = (attr,current)
			attr = best[0]
			
			new_attr = attributes[attributes!=attr]
			n_tree = self.Tree(attr)
			
			for v in self.getValues(attr,box):
				sample = []
				for i in box:
					if i.data[attr] == v:
						sample.append(i)
					
				
				subtree = self.tree(sample,new_attr,box)
				
				n_tree.addChild(v,subtree)
			
		return n_tree

	def getValues(self,attr,box):
		totaliample = len(box)
		values = {}
		for i in box:
			if i.data[attr] not in values:
				values[i.data[attr]] = [i]
			else:
				values[i.data[attr]].append(i)
		return values

	def getInfo(self,p_value,n_value):
		ans = 0
		total = p_value + n_value
		if total != 0:
			p_d = p_value/total
			n_d = n_value/total
			if p_d != 0:
				ans = -(p_d)*np.log2(p_d)
			if n_d != 0:
				ans = -(n_d)*np.log2(n_d)
		return ans

	def getPNValue(self,box):
		P, N = 0, 0
		for i in range(len(box)):
			if box[i].value == 1:
				P +=1
			else:
				N +=1
		return [P,N]

	def sumOfInformation(self,box,attr):
		length_container= len(box)
		values = self.getValues(attr,box)
		Total = 0
		for v in values:
			output = self.getPNValue(values[v])			
			Total += output[0]/length_container*self.getInfo(output[1],output[1])
		return Total

	def getGain(self,attr,box):
		output = self.getPNValue(box)
		ans = self.getInfo(output[0],output[1]) - self.sumOfInformation(box,attr)
		return ans

	def getPlurality(self,box):
		
		cont = {}
		for i in box:
			if i.value not in cont:
				cont[i.value] = 1
			else:
				cont[i.value] += 1
		res = max(cont,key=lambda x:cont[x])
		
		return res

	def readData(self,t,data):
		data_v = data[t.attr]
		if data_v not in t.child:
			return self.getMajorityOutcome(t)
		
		if t.child[data_v] == type(t):
			return self.readData(t.child[data_v],data)
		else:
			return t.child[data_v]

	def getOutcome(self,t):
		outcome = []
		for i in t.child:
			if type(t.child[i]) == type(self.tree):
				_out = self.getOutcome(t.child[i])
				outcome += _out
			else:
				outcome.append(t.child[i])
		
		return outcome

	def getMajorityOutcome(self,t):
		outcome = self.getOutcome(t)
		Value0, Value1 = 0,0
		for out in outcome:
			if out == 0:
				Value0 += 1
			else:
				Value1 += 1
		if Value1 > Value0:
			return 1
		elif Value0 < Value1:
			return 0
		else:
			return np.random.randint(0,1)

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)
		prediction = np.array([])
		
		for row in categorical_data:
			
			prediction = np.append(prediction,self.readData(self.tree,row))
			
		return prediction


