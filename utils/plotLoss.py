import matplotlib.pyplot as plt

filePath = "../logs/2021-05-05-1526.log"

f = open(filePath, "r")

bAP = list()
mAP = list()

for line in f.readlines():
    if "AP" in line:
        bbox_AP = float(line[line.index("bbox AP") + 9 : line.index(";")])
        mask_AP = float(line[line.index("mask AP") + 9 :])
        bAP.append(bbox_AP)
        mAP.append(mask_AP)

plt.plot(range(len(bAP)), bAP, label="bbox AP")
plt.plot(range(len(mAP)), mAP, label="mask AP")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("AP")
plt.show()
