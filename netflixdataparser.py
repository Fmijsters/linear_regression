import csv
# # userdict = {}
# # with open('data/users.csv') as csvfile:
# # 	smartreader = csv.DictReader(csvfile)
# # 	for row in smartreader:
# # 		userdict[row["id_user"]] = {"sex":row["sex"],"age":row["age"],"proffesion":row["proffesion"]}



# # moviedict={}
# # with open('data/movies.csv') as csvfile:
# # 	spamreader = csv.DictReader(csvfile)
# # 	for row in spamreader:
# # 		moviedict[row["id_movie"]] = {"year_of_release":row["year_of_release"],"title":row["title"]}

# with open('data/predictions.csv') as csvfile:
# 	spamreader = csv.DictReader(csvfile)
# 	f = open("needpredicting.txt", "x")
# 	for row in spamreader:
# 		sex = -1
# 		if userdict[row["id_user"]]["sex"] == "F":
# 			sex = 1
# 		else:
# 			sex = 0
# 		f.write("1.0,"+str(sex) + "," + userdict[row["id_user"]]["proffesion"] + "," + userdict[row["id_user"]]["age"] + "," + moviedict[row["id_movie"]]["year_of_release"] +"\n")

names = ['CM', 'KG', 'Apps', 'Mins', 'Goals', 'Assists', 'Yel', 'Red', 'SpG', 'PS%', 'AerialsWon', 'MotM','Tackles', 'Inter', 'Fouls', 'Offsides', 'Clear', 'Drb', 'Blocks', 'OwnG', 'KeyP', 'Fouled', 'Off', 'Disp', 'UnsTch', 'AvgP', 'Crosses', 'LongB', 'ThrB','Rating']
def fixAndPrintData(filename):
	with open('data/'+filename) as csvfile:
		smartreader = csv.DictReader(csvfile)
		for row in smartreader:
			row_string = ""
			for name in names:
				if row[name].find('(') is not -1:
					row[name] = row[name][:row[name].find("(")]
				if row[name] is "-":
					row[name] = 0
				if row_string is "":
					row_string = str(row[name])
				else:
					row_string = row_string + "," + str(row[name])
			print(row_string)

fixAndPrintData("ajax1718real.csv")
fixAndPrintData("ajax1617real.csv")
fixAndPrintData("ajax1516real.csv")
fixAndPrintData("ajax1415real.csv")