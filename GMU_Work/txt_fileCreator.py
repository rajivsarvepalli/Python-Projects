
def create(location, name,size):
    f = open(location+name,"w+")
    b= ""
    for i in range(0,size):
        b+=str(i)+ "\t"
    f.write(b)
    f.close()
def create_multiple(location,startName,size,numberOfFiles):
    startNum=int(''.join(list(filter(str.isdigit, startName))))
    cop = startNum
    print(startName.find(".")-len(str(startNum))-1)
    name1 = startName[0:startName.find(".")-len(str(startNum))]
    name2 = startName[startName.find("."):len(startName)]
    while startNum <numberOfFiles+cop:
        create(location,name1+str(startNum)+name2,size)
        startNum+=1
if __name__=="__main__":
    create_multiple("C:/USers/Rajiv SArvepalli/PRojects/GMU Work/AllData/fakedata/Combination_of_pfp2_and_txt/","fake14.txt",2000,100)