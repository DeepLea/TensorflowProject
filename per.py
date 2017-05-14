# My file for reading in images and shuffling them
import random

namefile = open("testfile50.txt","r")

text = namefile.readlines()

testData = list()
validationData = list()
trainingData = list()

print("************************\nPeople in this file:")

for number in range(0,len(text)):
			temp = text[number]
			t = temp.split()
			tt = t[0].split("_")
			print(tt[0] + " " + tt[1])
			for image in range(1,3):
						i = open("lfw/" + t[0] + "/" + t[0] + "_000" + str(image) + ".jpg")
						if image == 1:
									trainingData.append(i)
						if image == 2:
									validationData.append(i)
						if image == 3:
									testData.append(i)

print("************************")

namefile.close()

random.shuffle(trainingData)
random.shuffle(validationData)
random.shuffle(testData)

