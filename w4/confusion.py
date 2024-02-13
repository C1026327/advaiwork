TP = 114
FP = 14
TN = 0
FN = 12

TF = 7 # we set the True false values to 5 %
print("  FN    FP   TP     pre   acc   rec   f1")
for FN in range(0, 7):
    for FP in range(0, FN+1):
        # the sum of FN, FP, TF and TP will be 100:
        TP = 100 - FN - FP - TF
        #print(FN, FP, TP, FN+FP+TP+TF)
        precision = TP / (TP + FP)
        accuracy = (TP + TN)/(TP + TN + FP + FN)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall / (precision + recall)
        print(f"{FN:6.2f}{FP:6.2f}{TP:6.2f}", end="")
        print(f"{precision:6.2f}{accuracy:6.2f}{recall:6.2f}{f1_score:6.2f}")
