# My file for reading in images and shuffling them
import random

namefile = open("testfile50.txt","r")

text = namefile.readlines()

testData = list()
validationData = list()
trainingData = list()
names = list()
onehot = list()
print("************************\nPeople in this file:")

for number in range(0,len(text)):
	onehot.append(0);
onehot[0] = 1

for number in range(0,len(text)):
			
			temp = text[number]
			t = temp.split()
			names.append(t)
			tt = t[0].split("_")
			print(tt[0] + " " + tt[1] + " as: \n" + str(onehot))
			for image in range(1,int(t[1])):
						if image <= 9:
							i = open("lfw/" + t[0] + "/" + t[0] + "_000" + str(image) + ".jpg")
						elif image >= 10 and image <= 99:
							i = open("lfw/" + t[0] + "/" + t[0] + "_00" + str(image) + ".jpg")
						else:
							i = open("lfw/" + t[0] + "/" + t[0] + "_0" + str(image) + ".jpg")
						
						if image >= int(t[1])*0.5:
									trainingData.append([i,list(onehot)])
						elif image >= int(t[1])*0.25:
									validationData.append([i,list(onehot)])
						else:
									testData.append([i,list(onehot)])
			onehot.pop()
			onehot.insert(0,0)

print("************************")

namefile.close()

random.shuffle(trainingData)
random.shuffle(validationData)
random.shuffle(testData)

print("Length trainingData:   " + str(len(trainingData)))
print("Length validationData: " + str(len(validationData)))
print("Length testData:       " + str(len(testData)))
print(testData[0])

print("************************")

print("Not training yet")

print("************************")

accuracy = 0;
print("Accuracy is: " + str(accuracy) + " %")

print("************************")







