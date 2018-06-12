'''
'''
import pickle 
import os

def Percentile_search():


    '''
        Search for the best P1, P2.

        Args:
            threshS_img_pred_dict: 
            slides_GT: 

        Returns:

    '''

    step=0 # for display use

    prange = np.arange(20, 99, 0.5)

    dmax=0.0

    for p1 in prange:
        for p2 in prange:
            
            step += 1
            
            # {slide: threshold} 
            image_level_thresh=dict()

            for i in list(threshS_img_pred_dict.keys()):
                '''
                    Loop over prediction results of each slides.
                '''

                predvalues=sorted(list(threshS_img_pred_dict[i].values()))

                if slides_GT[i] == 'negative':
                    curlabel=0
                else:
                    curlabel=1

                # image-level threshold
                chosen_thresh=np.percentile(predvalues, p1)
                image_level_thresh[i] = (chosen_thresh, curlabel)
            
            negnum=0
            posnum=0

            pospred=list()
            negpred=list()

            for slide_ in list(image_labels.keys())[:]:

                tmppred = threshS_img_pred_dict[slide_]

                if image_labels[slide_]:
                    posnum += len(tmppred)

                    pospred+=list(tmppred.values())

                else:
                    negnum += len(tmppred)
                    negpred+=list(tmppred.values())

            # class-level thresholds
            negthresh = np.percentile(negpred, p2)
            posthresh = np.percentile(pospred, p2)
            thresholds=(negthresh, posthresh)

            num=0

            final_thresh_dict=dict()

            for i,v in image_level_thresh.items():

                num += 1
                # print(i, min(v[0], thresholds[v[1]]), v[1], )

                final_thresh_dict[i] = (min(v[0], thresholds[v[1]]), v[1])

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
            
            ratio=discri_num_all / selected_num_all
            
            if ratio > dmax:
                dmax = ratio
                maxcomb=(p1, p2)
                
            if step and not(step % 10):
                print(dmax)