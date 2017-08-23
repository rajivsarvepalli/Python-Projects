import numpy as np
class pfp2_files_data:
    def __init__(self,year,month,day,unused1,hour,minute,second,unused2,DAQMode,Duration,TriggerSampleIdx,
    SequenceNumber,ClassDescriptor,PFPInternalUse1,PFPInternalUse2,PFPInternalUse3,PFPInternalUse4,data):
        self.year =year
        self.month =month
        self.day =day
        self.unused1 = unused1
        self.hour = hour
        self.minute = minute
        self.second = second
        self.unused2 =unused2
        self.DAQMode = DAQMode
        self.Duration = Duration
        self.TriggerSampleIdx = TriggerSampleIdx
        self.SequenceNumber = SequenceNumber
        self.ClassDescriptor = ClassDescriptor
        self.PFPInternalUse1 = PFPInternalUse1
        self.PFPInternalUse2 = PFPInternalUse2
        self.PFPInternalUse3 = PFPInternalUse3
        self.PFPInternalUse4 = PFPInternalUse4
        self.data = data
def getData(file_name):
    f= open(file_name,"r")
    a = np.fromfile(f,dtype=np.uint32)

    year = a[1] & 0xfff
    month=a[1]&0xffff&0x000f
    day = (a[1]&0xf0000)>>16
    unused1 = a[1] >>21

    hour = a[2] & 0x0000f
    minute = ((a[2] <<1)&0x00ff0) >>6 #wrong
    second = (a[2]& 0xfff00) >>11
    unused2 = (a[2]&0x0ffff)>>16

    DAQMode = a[3]&0x000f
    Duration = (a[3] & 0xfff0)>>2

    Fs = a[4]

    SDRL = a[5]

    TriggerRec = (a[6] &0xf)>>31     #1bit****not right
    Unused3 = a[6]>>1   #31bit

    TriggerSampleIdx=a[7]

    SequenceNumber = a[7]

    ClassDescriptor = a[8]

    PFPInternalUse1 = a[9]
    PFPInternalUse2 = a[10]
    PFPInternalUse3 =a[11]
    PFPInternalUse4 = a[12]
    f.close()
    f= open(file_name,"r")
    a = np.fromfile(f,dtype=np.single)
    data =a[14:len(a)]
    dictionary = pfp2_files_data(year,month,day,unused1,hour,minute,second,unused2,DAQMode,Duration,TriggerSampleIdx,
    SequenceNumber,ClassDescriptor,PFPInternalUse1,PFPInternalUse2,PFPInternalUse3,PFPInternalUse1,data)
    f.close()
    return dictionary
#end
if __name__ == "__main__":
    x=getData("C:/Users/Rajiv Sarvepalli/Projects/Python-Projects/GMU Work/AllData/PFP2Files/20170110_093852_034.pfp2")
    for key in vars(x):
        s= vars(x)[key]
        print(str(key) + ": "+ str(s))
    print("Done")



