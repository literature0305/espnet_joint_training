import numpy as np
import torch
import sys
import re
import matplotlib.pyplot as plt

# Step 1: Read data and extract acc and conf values
def plot_stepped_area_chart(input_file, acc_list, conf_list, location="lower right"):
    # input_file = 'errlog005-3_decode-TF_none2' #'errlog005-4_decode_none2'
    output_file = input_file + '.png'
    # with open(input_file, 'r') as f:
    #     lines = f.readlines()

    # acc_list = []
    # conf_list = []
    # # label_list = []

    # for line in lines:
    #     # Extract acc and conf values using regular expressions
    #     acc = re.findall(r'acc: (.*?)[\s]', line)[0]
    #     conf = re.findall(r'conf: (.*?)[\s]', line)[0]
        
    #     # Append to respective lists
    #     acc_list.append(float(acc))
    #     conf_list.append(float(conf))
        
    x_label=[]
    for i in range(len(acc_list)):
        x_label.append(str(i+1))

    # Step 2: Create a stepped area chart
    fig, ax = plt.subplots()
    # ax.fill_between(range(1, len(acc_list)+1), conf_list, step='pre', alpha=0.5, color='blue')
    # ax.fill_between(range(1, len(acc_list)+1), acc_list, step='pre', alpha=0.5, color='red')
    ax.fill_between(x_label, conf_list, step='pre', alpha=0.5, color='red')
    ax.fill_between(x_label, acc_list, step='pre', alpha=0.5, color='blue')
    ax.set_xlabel('Bins')
    ax.set_ylabel('Accuracy/Confidence')
    ax.set_ylim([0, 1])
    ax.legend(['Confidence', 'Accuracy'], loc=location, fontsize = 20)
    plt.savefig(output_file)
    plt.show()


# conf
top_n_classes_to_print = 3
num_bins_list = [2, 10, 20, 40, 80]
num_bins_for_class_wise_ece = 20
output_path = sys.argv[1]
if len(sys.argv) != 2:
    print("Insufficient arguments")
    sys.exit()
eps = 0.00000000000001

char_list=[
    "<blank>",
    "<unk>",
    "'",
    "A",
    "ABLE",
    "AD",
    "AGE",
    "AK",
    "AL",
    "AM",
    "AN",
    "ANCE",
    "ANT",
    "AR",
    "AS",
    "AT",
    "ATE",
    "ATION",
    "B",
    "C",
    "CE",
    "CH",
    "CK",
    "CT",
    "D",
    "DER",
    "E",
    "ED",
    "EL",
    "EN",
    "ENCE",
    "ENT",
    "ER",
    "ERS",
    "ES",
    "ET",
    "F",
    "FUL",
    "G",
    "GE",
    "H",
    "I",
    "IBLE",
    "IC",
    "ID",
    "IES",
    "IG",
    "IGHT",
    "IL",
    "IM",
    "IN",
    "ING",
    "ION",
    "IOUS",
    "IR",
    "IS",
    "ISH",
    "IST",
    "IT",
    "ITY",
    "IVE",
    "J",
    "K",
    "L",
    "LA",
    "LAND",
    "LE",
    "LESS",
    "LI",
    "LL",
    "LO",
    "LU",
    "LY",
    "M",
    "MENT",
    "N",
    "NE",
    "NESS",
    "NG",
    "O",
    "OL",
    "OM",
    "ON",
    "OOK",
    "OR",
    "OUGH",
    "OW",
    "P",
    "Q",
    "R",
    "RA",
    "RE",
    "RESS",
    "RI",
    "RY",
    "S",
    "SIDE",
    "ST",
    "T",
    "TAIN",
    "TER",
    "TH",
    "THER",
    "TION",
    "U",
    "UL",
    "UN",
    "UND",
    "UR",
    "URE",
    "US",
    "V",
    "VE",
    "VER",
    "W",
    "WARD",
    "X",
    "Y",
    "Z",
    "ZZ",
    "▁",
    "▁A",
    "▁ABOUT",
    "▁AFTER",
    "▁AGAIN",
    "▁ALL",
    "▁AN",
    "▁AND",
    "▁ANY",
    "▁ARE",
    "▁AS",
    "▁AT",
    "▁B",
    "▁BA",
    "▁BACK",
    "▁BE",
    "▁BEEN",
    "▁BEFORE",
    "▁BO",
    "▁BR",
    "▁BUT",
    "▁BY",
    "▁C",
    "▁CA",
    "▁CAME",
    "▁CAN",
    "▁CAR",
    "▁CH",
    "▁CHA",
    "▁CO",
    "▁COM",
    "▁CON",
    "▁COULD",
    "▁CR",
    "▁D",
    "▁DAY",
    "▁DE",
    "▁DI",
    "▁DID",
    "▁DIS",
    "▁DO",
    "▁DOWN",
    "▁E",
    "▁EN",
    "▁EVEN",
    "▁EVERY",
    "▁EX",
    "▁F",
    "▁FA",
    "▁FIRST",
    "▁FOR",
    "▁FROM",
    "▁G",
    "▁GO",
    "▁GOOD",
    "▁GRA",
    "▁GREAT",
    "▁GRO",
    "▁HA",
    "▁HAD",
    "▁HAND",
    "▁HAVE",
    "▁HE",
    "▁HER",
    "▁HIM",
    "▁HIMSELF",
    "▁HIS",
    "▁HO",
    "▁HOUSE",
    "▁HOW",
    "▁HU",
    "▁I",
    "▁IF",
    "▁IMP",
    "▁IN",
    "▁INTO",
    "▁IS",
    "▁IT",
    "▁JO",
    "▁K",
    "▁KNOW",
    "▁LA",
    "▁LE",
    "▁LI",
    "▁LIKE",
    "▁LITTLE",
    "▁LO",
    "▁LONG",
    "▁LOOK",
    "▁MA",
    "▁MADE",
    "▁MAN",
    "▁ME",
    "▁MI",
    "▁MIGHT",
    "▁MISS",
    "▁MISTER",
    "▁MO",
    "▁MORE",
    "▁MU",
    "▁MUCH",
    "▁MY",
    "▁NA",
    "▁NE",
    "▁NO",
    "▁NOT",
    "▁NOW",
    "▁O",
    "▁OF",
    "▁OLD",
    "▁ON",
    "▁ONE",
    "▁OR",
    "▁OTHER",
    "▁OUR",
    "▁OUT",
    "▁OVER",
    "▁OWN",
    "▁P",
    "▁PA",
    "▁PART",
    "▁PO",
    "▁PRE",
    "▁PRO",
    "▁R",
    "▁RA",
    "▁RE",
    "▁RO",
    "▁S",
    "▁SA",
    "▁SAID",
    "▁SEE",
    "▁SH",
    "▁SHE",
    "▁SHOULD",
    "▁SO",
    "▁SOME",
    "▁SP",
    "▁ST",
    "▁SU",
    "▁T",
    "▁TAKE",
    "▁TH",
    "▁THAN",
    "▁THAT",
    "▁THE",
    "▁THEIR",
    "▁THEM",
    "▁THERE",
    "▁THEY",
    "▁THINK",
    "▁THIS",
    "▁THOUGH",
    "▁THROUGH",
    "▁TIME",
    "▁TO",
    "▁TWO",
    "▁UN",
    "▁UNDER",
    "▁UP",
    "▁UPON",
    "▁US",
    "▁VA",
    "▁VERY",
    "▁VI",
    "▁W",
    "▁WAS",
    "▁WAY",
    "▁WE",
    "▁WELL",
    "▁WERE",
    "▁WHAT",
    "▁WHEN",
    "▁WHERE",
    "▁WHICH",
    "▁WHO",
    "▁WILL",
    "▁WITH",
    "▁WOULD",
    "▁YOU",
    "▁YOUR",
    "<eos>"
]

# read
with open('am_lev_log.txt', 'r') as f:
    lines_lev = f.readlines()

count = torch.zeros(999)
insertion = torch.zeros(999)
deletion = torch.zeros(999)
substitution = torch.zeros(999)
correction = torch.zeros(999)

for line in lines_lev:
    line = line.split(' ')
    for idx, token in enumerate(line):
        count[idx] = count[idx] + 1
        if token == "S":
            substitution[idx] = substitution[idx] + 1
        elif token =='C':
            correction[idx] = correction[idx] + 1
        elif token =='I':
            insertion[idx] = insertion[idx] + 1
        elif token =='D':
            deletion[idx] = deletion[idx] + 1

print('ACC by len (COR, INS, DEL, SUB, #sample)')
accuracy=[]
del_plot=[]
ins_plot=[]
sub_plot=[]
for i in range(len(count)):
    if count[i] ==0:
        break
    # print(correction[i]/count[i], insertion[i]/count[i], deletion[i]/count[i], substitution[i]/count[i])
    cor = round((correction[i]/count[i]).item(), 4)
    del2 = round((deletion[i]/count[i]).item(), 4)
    ins = round((insertion[i]/count[i]).item(), 4)
    sub = round((substitution[i]/count[i]).item(), 4)
    if count[i]> 10:
        accuracy.append(cor)
        del_plot.append(del2)
        ins_plot.append(ins)
        sub_plot.append(sub)
    print("{:.4f} {:.4f} {:.4f} {:.4f}".format(cor, ins, del2, sub), int(count[i].item()))

print('')
print('ACC before 50:', (correction[:50].sum()/count[:50].sum()).item())
print('ACC 50:100:', (correction[50:100].sum()/count[50:100].sum()).item())
print('ACC 100:150:', (correction[100:150].sum()/count[100:150].sum()).item())
print('ACC after 150:', (correction[150:].sum()/count[150:].sum()).item())
# generate x-axis values
x_axis = list(range(len(accuracy)))
# create a new figure and axis
fig, ax = plt.subplots()

# plot the accuracy values as a line graph
ax.plot(x_axis, accuracy, label='Accuracy')
ax.plot(x_axis, del_plot, label='Deletion')
ax.plot(x_axis, ins_plot, label='Insertion')
ax.plot(x_axis, sub_plot, label='Substitution')

# set the x-axis label
ax.set_xlabel('Sequence Length')

# set the y-axis label
ax.set_ylabel('Accuracy/Error')

# set the title of the plot
ax.set_title('Accuracy/Error by Length')
ax.legend(fontsize = 16)
# display the plot
plt.savefig(output_path + '_acc_len.png')
plt.show()

print('')

# read
pdf = torch.Tensor(np.load('ece_ctc.npy'))
label = torch.Tensor(np.load('ece_am_target.npy'))
label = torch.nn.functional.one_hot(label.to(torch.long), num_classes=pdf.size(-1))

# get NLL, mean-probabiliy, Brier score
nll = - (torch.log(pdf+eps) * label).sum(-1).mean()
mean_prob =(pdf*label).sum(-1).mean()
bier = ((label-pdf)*(label-pdf)).mean()
print('NLL:', nll.item())
print('BRIER:', bier.item())
print('MEAN_Prob:', mean_prob.item())

# get sorted pdf, labels (classwise)
sorted_pdf, indices = torch.sort(pdf)
sorted_label = label.clone()
for i in range(len(label)):
    sorted_label[i] = sorted_label[i][indices[i]]

# get sorted pdf, labels (samplewise after classwise)
sorted_pdf = sorted_pdf.transpose(0,1) # B,V -> V,B
sorted_label = sorted_label.transpose(0,1)
sorted_pdf2, indices2 = torch.sort(sorted_pdf)
sorted_label2 = sorted_label.clone()
for i in range(len(sorted_label2)):
    sorted_label2[i] = sorted_label2[i][indices2[i]]

sorted_pdf = sorted_pdf.transpose(0,1) # V,B -> B,V
sorted_label = sorted_label.transpose(0,1)
sorted_pdf2 = sorted_pdf2.transpose(0,1) # V,B -> B,V
sorted_label2 = sorted_label2.transpose(0,1)

# check if well sorted
# nll_after = - (torch.log(sorted_pdf+eps) * sorted_label).sum(-1).mean()
# nll_after2 = - (torch.log(sorted_pdf2+eps) * sorted_label2).sum(-1).mean()
# assert int(nll*100000) == int(nll_after*100000)
# assert int(nll*100000) == int(nll_after2*100000)

# calculate top-k ECE (equal mass)
for num_bins in num_bins_list:
    print('')
    print('num_bins (equ-mass):', num_bins)
    per_bin = float(int(len(pdf) / num_bins))
    for k in range(top_n_classes_to_print):
        print('')
        print('top-k:', k+1)
        top_k_conf = sorted_pdf2[:,-1-k]
        top_k_cor = sorted_label2[:,-1-k]

        ece = 0
        conf_bin = 0
        cor_bin = 0
        count_bin = 0

        acc_list_to_plot=[]
        conf_list_to_plot=[]
        for i in range(len(top_k_conf)):
            conf_bin = conf_bin + top_k_conf[i]
            cor_bin = cor_bin + top_k_cor[i]
            count_bin = count_bin + 1

            iii = (i+1)%per_bin
            if iii==0:
                conf_mean = conf_bin / count_bin
                acc_mean = cor_bin / count_bin
                acc_list_to_plot.append(acc_mean)
                conf_list_to_plot.append(conf_mean)

                if conf_mean > acc_mean:
                    print('num', int((i+1)//per_bin), 'acc:', acc_mean.item(), 'conf:', conf_mean.item(), 'OVER_CONF')
                else:
                    print('num', int((i+1)//per_bin), 'acc:', acc_mean.item(), 'conf:', conf_mean.item(), 'UNDER_CONF')
                ece = ece + abs(acc_mean - conf_mean)
                conf_bin=0
                cor_bin=0
                count_bin = 0
        print('ECE for top', k+1, 'num_bins (equal mass):', num_bins, (ece/((i+1)//per_bin)).item())
        if k == 0 and num_bins == 20:
            name_figure = output_path + str(k) + 'k_' + str(num_bins) + 'bins_' + 'eq-mass'
            plot_stepped_area_chart(name_figure, acc_list_to_plot, conf_list_to_plot)
        elif k == 0 and num_bins == 10:
            name_figure = output_path + str(k) + 'k_' + str(num_bins) + 'bins_' + 'eq-mass'
            plot_stepped_area_chart(name_figure, acc_list_to_plot, conf_list_to_plot)
        elif k == 1 and num_bins == 20:
            name_figure = output_path + str(k) + 'k_' + str(num_bins) + 'bins_' + 'eq-mass'
            plot_stepped_area_chart(name_figure, acc_list_to_plot, conf_list_to_plot, location='upper left')
        elif k == 1 and num_bins == 10:
            name_figure = output_path + str(k) + 'k_' + str(num_bins) + 'bins_' + 'eq-mass'
            plot_stepped_area_chart(name_figure, acc_list_to_plot, conf_list_to_plot, location='upper left')
        elif k == 2 and num_bins == 20:
            name_figure = output_path + str(k) + 'k_' + str(num_bins) + 'bins_' + 'eq-mass'
            plot_stepped_area_chart(name_figure, acc_list_to_plot, conf_list_to_plot, location='upper left')
        elif k == 2 and num_bins == 10:
            name_figure = output_path + str(k) + 'k_' + str(num_bins) + 'bins_' + 'eq-mass'
            plot_stepped_area_chart(name_figure, acc_list_to_plot, conf_list_to_plot, location='upper left')

# calculate top-k ECE (equal width)
for num_bins in num_bins_list:
    print('')
    print('num_bins (equ-width):', num_bins)
    width = 100 / num_bins

    for k in range(top_n_classes_to_print):
        print('')
        print('top-k:', k+1)
        top_k_conf = sorted_pdf2[:,-1-k]
        top_k_cor = sorted_label2[:,-1-k]

        ece=0
        sum1_to_check=0
        acc_binwise=np.zeros(num_bins)
        conf_binwise=np.zeros(num_bins)
        num_samples_binwise=np.zeros(num_bins)
        acc_list_to_plot=[]
        conf_list_to_plot=[]
        for i in range(len(top_k_conf)):
            bin_idx = int((top_k_conf[i] * 100) // width)
            if bin_idx == num_bins:
                bin_idx = bin_idx - 1

            acc_binwise[bin_idx] = acc_binwise[bin_idx] + top_k_cor[i]
            conf_binwise[bin_idx] = conf_binwise[bin_idx] + top_k_conf[i]
            num_samples_binwise[bin_idx] = num_samples_binwise[bin_idx] + 1

        for bin_idx in range(num_bins):
            if num_samples_binwise[bin_idx] == 0:
                print('num', bin_idx, 'acc:', None, 'conf:', None, 'OVER_CONF', '#sample_bin:', int(num_samples_binwise[bin_idx]))
                acc_list_to_plot.append(0)
                conf_list_to_plot.append(0)
                ece_bin = 0
            else:
                acc_bin = acc_binwise[bin_idx] / num_samples_binwise[bin_idx]
                conf_bin = conf_binwise[bin_idx] / num_samples_binwise[bin_idx]
                acc_list_to_plot.append(acc_bin)
                conf_list_to_plot.append(conf_bin)
                ece_bin = abs(acc_bin-conf_bin)
                if conf_bin > acc_bin:
                    print('num', bin_idx, 'acc:', acc_bin, 'conf:', conf_bin, 'OVER_CONF', '#sample_bin:', int(num_samples_binwise[bin_idx]))
                else:
                    print('num', bin_idx, 'acc:', acc_bin, 'conf:', conf_bin, 'UNDER_CONF', '#sample_bin:', int(num_samples_binwise[bin_idx]))

            ece = ece + ece_bin * num_samples_binwise[bin_idx] / len(top_k_conf)
            sum1_to_check = sum1_to_check+num_samples_binwise[bin_idx] / len(top_k_conf)

        print('K:', k, 'num_bins:', num_bins, 'ECE (equal width):', ece.item(), 'sum1_to_check:', sum1_to_check.item())
        if k == 0 and num_bins == 20:
            name_figure = output_path + str(k) + 'k_' + str(num_bins) + 'bins_' + 'eq-width'
            plot_stepped_area_chart(name_figure, acc_list_to_plot, conf_list_to_plot)
        elif k == 0 and num_bins == 10:
            name_figure = output_path + str(k) + 'k_' + str(num_bins) + 'bins_' + 'eq-width'
            plot_stepped_area_chart(name_figure, acc_list_to_plot, conf_list_to_plot)


# get sorted pdf, labels (samplewise)
pdf = pdf.transpose(0,1) # B,V -> V,B
label = label.transpose(0,1)
sorted_pdf3, indices3 = torch.sort(pdf)
sorted_label3 = label.clone()
for i in range(len(sorted_label3)):
    sorted_label3[i] = sorted_label3[i][indices3[i]]
pdf = pdf.transpose(0,1) # V,B -> B,V
label = label.transpose(0,1)
sorted_pdf3 = sorted_pdf3.transpose(0,1) # V,B -> B,V
sorted_label3 = sorted_label3.transpose(0,1)

# check if well sorted
# nll_after3 = - (torch.log(sorted_pdf3+eps) * sorted_label3).sum(-1).mean()
# assert int(nll*100000) == int(nll_after3*100000)

# calculate class-wise ECE
ece_cw = torch.zeros(pdf.size(-1))
conf_bin_cw = torch.zeros(pdf.size(-1))
cor_bin_cw = torch.zeros(pdf.size(-1))
count_bin_cw = 0

conf_mean_accum=torch.zeros(pdf.size(-1))
acc_mean_accum=torch.zeros(pdf.size(-1))
count_mean_accum = 0

per_bin = float(int(len(pdf) / num_bins_for_class_wise_ece))
for i in range(len(sorted_pdf3)):
    conf_bin_cw=conf_bin_cw+sorted_pdf3[i]
    cor_bin_cw=conf_bin_cw+sorted_label3[i]
    count_bin_cw=count_bin_cw+1
    iii = (i+1)%per_bin
    
    if iii==0:
        conf_mean = conf_bin_cw / count_bin_cw
        acc_mean = cor_bin_cw / count_bin_cw
        ece_cw = ece_cw + abs(acc_mean - conf_mean)

        conf_bin_cw=0
        cor_bin_cw=0
        count_bin_cw = 0

        conf_mean_accum = conf_mean_accum + conf_mean
        acc_mean_accum = acc_mean_accum + acc_mean
        count_mean_accum = count_mean_accum + 1

ece_cw = ece_cw / ((i+1)//per_bin)
sorted_ece_cw, indices_ece_cw = torch.sort(ece_cw)

print('class wise ECE:', ece_cw.mean().item())
print('class wise ECE 1st-highest:', sorted_ece_cw[-1].item(), 'class:', indices_ece_cw[-1].item(), char_list[indices_ece_cw[-1].item()])
print('class wise ECE 2nd-highest:', sorted_ece_cw[-2].item(), 'class:', indices_ece_cw[-2].item(), char_list[indices_ece_cw[-2].item()])
print('class wise ECE 3rd-highest:', sorted_ece_cw[-3].item(), 'class:', indices_ece_cw[-3].item(), char_list[indices_ece_cw[-3].item()])
print('')
for idx, label_idx in enumerate(indices_ece_cw):
    idx_new = -1 -idx
    if conf_mean_accum[idx_new] > acc_mean_accum[idx_new]:
        print('class wise ECE of ', idx+1, 'th-highest:', ece_cw[label_idx], ' conf:', conf_mean_accum[label_idx], ' acc:', acc_mean_accum[label_idx], ' class:', label_idx.item(), char_list[label_idx.item()], 'OVER_CONF')
    else:
        print('class wise ECE of ', idx+1, 'th-highest:', ece_cw[label_idx], ' conf:', conf_mean_accum[label_idx], ' acc:', acc_mean_accum[label_idx], ' class:', label_idx.item(), char_list[label_idx.item()], 'UNDER_CONF')

# calculate distribution shift
pdf_argmax = torch.argmax(pdf, dim=-1)
pdf_argmax = torch.nn.functional.one_hot(pdf_argmax.to(torch.long), num_classes=pdf.size(-1))

true_dist = label.sum(0) / label.sum()
output_dist = pdf.sum(0) / pdf.sum()
output_dist_sharp = pdf_argmax.sum(0) / pdf_argmax.sum()

dist_shift = abs(true_dist - output_dist)
_, indices_dist_shift = torch.sort(dist_shift)

print('indices_dist_shift:', indices_dist_shift)
print(output_dist.size())
print(true_dist.size())
print(output_dist_sharp.size())
for idx, label_idx in enumerate(indices_dist_shift):
    if output_dist[label_idx] > true_dist[label_idx]:
        print('Distribution shift of ', idx+1, 'th-highest:', dist_shift[label_idx], ' true:', true_dist[label_idx], ' estimated:', output_dist[label_idx], ' estimated_sharpened:', output_dist_sharp[label_idx], ' class:', label_idx.item(), char_list[label_idx.item()], 'OVER_ESTIMATE')
    else:
        print('Distribution shift of ', idx+1, 'th-highest:', dist_shift[label_idx], ' true:', true_dist[label_idx], ' estimated:', output_dist[label_idx], ' estimated_sharpened:', output_dist_sharp[label_idx], ' class:', label_idx.item(), char_list[label_idx.item()], 'UNDER_ESTIMATE')
print('pdf.sum()',pdf.sum(), 'label.sum()', label.sum())
print('Total distribution shift:', dist_shift.sum())
