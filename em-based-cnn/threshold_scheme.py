


'''
'''


selected_num_all=0
discri_num_all=0
before_num_all=0

for slide_, threshf in final_thresh_dict.items():
    
    # print(slide_, threshf)
    selected_num=0
    discri_num=0
    before_num=0

    for i, v in threshS_img_pred_dict[slide_].items():

        before_num+=1

        if slides_GT[slide_] == 'negative':
            slide_label=0
        else:
            slide_label=1

        if v > threshf[0]:
            selected_num+=1

            if training_groundTruth[i] == slide_label:
                discri_num+=1
    
    selected_num_all += selected_num
    discri_num_all += discri_num
    before_num_all += before_num
    
    # print(discri_num, selected_num, before_num)
    
print(discri_num_all, selected_num_all, before_num_all)
print(discri_num_all / selected_num_all)