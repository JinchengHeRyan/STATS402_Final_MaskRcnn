import matplotlib.pyplot as plt

# filePath = "../logs/2021-05-05-1526.log"
filePath = "../logs/2021-05-06-2036_VOC.log"

f = open(filePath, "r")

trainLossExist = True

dataset = "VOC"

bAP = list()
mAP = list()


class Losses:
    def __init__(self, name: str):
        self.name = name
        self.val = list()

    def getName(self):
        return self.name


rpn_objectness_losses = Losses("rpn objectness loss")
rpn_box_losses = Losses("rpn box loss")
roi_classifier_losses = Losses("roi classifier loss")
roi_box_losses = Losses("roi box loss")
roi_mask_losses = Losses("roi mask loss")

for line in f.readlines():
    if "AP" in line:
        bbox_AP = float(line[line.index("bbox AP") + 9 : line.index(";")])
        mask_AP = float(line[line.index("mask AP") + 9 :])
        bAP.append(bbox_AP)
        mAP.append(mask_AP)

    if trainLossExist:
        if "loss" in line:
            rpn_obj_loss = float(
                line[
                    line.index("rpn_objectness_loss")
                    + 20 : line.index("rpn_objectness_loss")
                    + 26
                ]
            )
            rpn_box_loss = float(
                line[line.index("rpn_box_loss") + 13 : line.index("rpn_box_loss") + 19]
            )
            roi_classifier_loss = float(
                line[
                    line.index("roi_classifier_loss")
                    + 20 : line.index("roi_classifier_loss")
                    + 26
                ]
            )
            roi_box_loss = float(
                line[line.index("roi_box_loss") + 13 : line.index("roi_box_loss") + 19]
            )
            roi_mask_loss = float(
                line[
                    line.index("roi_mask_loss") + 14 : line.index("roi_mask_loss") + 20
                ]
            )
            rpn_objectness_losses.val.append(rpn_obj_loss)
            rpn_box_losses.val.append(rpn_box_loss)
            roi_classifier_losses.val.append(roi_classifier_loss)
            roi_box_losses.val.append(roi_box_loss)
            roi_mask_losses.val.append(roi_mask_loss)

plt.plot(range(len(bAP)), bAP, label="bbox AP")
plt.plot(range(len(mAP)), mAP, label="mask AP")
plt.legend()
plt.title("{} dataset".format(dataset))
plt.xlabel("epochs")
plt.ylabel("AP")
plt.show()

losses_list = [
    rpn_objectness_losses,
    rpn_box_losses,
    roi_classifier_losses,
    roi_box_losses,
    roi_mask_losses,
]

for lossList in losses_list:
    plt.plot(range(len(lossList.val)), lossList.val, label=lossList.name)

plt.legend()
plt.ylim(0, 0.4)
plt.title("{} dataset".format(dataset))
plt.show()
