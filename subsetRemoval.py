Try it with this: this script will check the coordinates of a sorted file.
############## rangeCheck.py ###############
y_66 = []
concat = []
dummy = ''
result = ''
input_file = "test.txt"
with open(input_file,"r") as f1:
  a = f1.readlines()
  for y in range(0,len(a)):
    concat.append(a[y].split('\t'))


for col1 in range(0,len(concat)):
  y_66.append(range(int(concat[col1][1]), int(concat[col1][2])))

for y in range(1,len(y_66)):
  if set((y_66[y-1])).issubset(y_66[y]):
    pass
  else:
    result += a[y-1]


result += a[len(y_66)-1]
print(result)
print("\n")
print("Running Tests with: " + input_file)
print("#########################################\n")
#### For Testing Purposes ######################
for l in range(1,len(y_66)):
  flag = False
  testing = y_66[l-1]
  #print("Testing: " + str(testing))
  while l < len(y_66):
    if set((testing)).issubset(y_66[l]):
      flag = True
      print("flagged: " + str(testing) + " is in : " + str(y_66[l]) + "\n")
      l = 7777
    else :
      l += 1
  if flag == True:
    pass
  else:
    dummy += str(testing) + '\n'
#################################################
