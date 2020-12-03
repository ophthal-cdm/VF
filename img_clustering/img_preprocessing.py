import glob
import cv2
import os

path = 

def remove_background(p,img):
    width = 60
    height=30
    top = int(height*0.15)
    bottom=top
    left = int(width*0.15)
    right=left
    dim = (width,height)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,dim,interpolation=cv2.INTER_NEAREST)
    blur = cv2.GaussianBlur(img,(3,3),0)
    # thr2 =cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,5)
    ret, temp = cv2.threshold(blur, int(cv2.mean(blur)[0]), 255,cv2.THRESH_TRUNC)
    # ret, thr2 = cv2.threshold(temp, int(cv2.mean(temp)[0]), 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, thr2 = cv2.threshold(temp, int(cv2.mean(temp)[0]), 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thr2,(3,3),0)
    im2 = cv2.copyMakeBorder(blur,top,bottom,left,right,cv2.BORDER_CONSTANT,None,[255,255,255])
    cv2.imwrite(path[:-4]+'_final_b_blur_processed.jpg',im2)


# detect된 border rects 넓이의 평균을 내서, 해당 평균을 넘으면 select

p = glob.glob('/home/ubuntu/data/Workspace/Soobin/data/ocr/testdataset/*')
p = [path for path in p if '_' in path]
print(p)
for path in p:
    os.remove(path)

def detect_digits(p): 
    p = glob.glob(p+'/*')

    for indexofp, path in enumerate(p):
        imgg = cv2.imread(path)
        imgg=cv2.cvtColor(imgg,cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(imgg, 20,50)
        contours,hierarchy = cv2.findContours(canny, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        boundRect = [None]*len(contours)
        for i, c in enumerate(contours):
            boundRect[i] = cv2.boundingRect(c)

        rects_area = [w*h for (x,y,w,h) in boundRect]
        rects_mean = sum(rects_area)/len(rects_area)
        rects_selected_idx = [idx for idx, a in enumerate (rects_area) if a>=rects_mean]
        # detect된 border rects 넓이의 평균을 내서, 해당 평균을 넘으면 select

        for idx,bound in enumerate(rects_selected_idx):
            if idx == 0:
                x,y,w,h = boundRect[bound]
                output = cv2.rectangle(imgg.copy(), (x, y), (x+w, y+h), (50,0,50), 2)
            else:
                x,y,w,h = boundRect[bound]
                output = cv2.rectangle(output, (x, y), (x+w, y+h), (50,0,50), 2)
            
            # digit만 crop하는 과정
            only_num_cropped = imgg[y-1:y+h+1,x-1:x+w+1]
            padding = 3
            only_num_cropped = cv2.copyMakeBorder(only_num_cropped,padding,padding,padding,padding,cv2.BORDER_CONSTANT,None,[255,255,255])
            cropped_path = '/home/ubuntu/data/Workspace/Soobin/data/ocr/test_dataset/'+str(indexofp)+'_'+str(idx)+'.jpg'
            cv2.imwrite(cropped_path,only_num_cropped)

        border_rect_path = '/home/ubuntu/data/Workspace/Soobin/data/ocr/test_dataset/'+str(indexofp)+'.jpg'
        cv2.imwrite(border_rect_path,output)
    return

# detect_digits('/home/ubuntu/data/Workspace/Soobin/data/ocr/test_dataset')



        #------ to find the largest rectangle boundary
        # rects = [w*h for (x,y,w,h) in boundRect]
        # border = np.argmax(rects)
        # x,y,w,h = boundRect[border]
        # expand_x_start = (x/imgg.shape[1])*img.shape[1]
        # expand_w = (w/imgg.shape[1])*img.shape[1]
        # expand_y_start = (y/imgg.shape[0])*img.shape[0]
        # expand_h = (h/imgg.shape[0])*img.shape[0]

        # tuning_x = 1.5
        # tuning_y = 2
        # tuned_x_start=int(expand_x_start-tuning_x)
        # tuned_x_end = tuned_x_start+int(expand_w+tuning_x)
        # tuned_y_start=int(expand_y_start-tuning_y)
        # tuned_y_end = tuned_y_start+int(expand_h+tuning_y)
        # output_width = img.shape[1]
        # output_height =img.shape[0]


        # output_path = '/home/ubuntu/data/Workspace/Soobin/data/test_dataset/'
        # output = cv2.rectangle(img.copy(), (tuned_x_start, tuned_y_start), (tuned_x_end, tuned_y_end), (50,0,50), 1)
        # cv2.imwrite(output_path+str(i)+'final_output.jpg',output)


        # if tuned_x_end > output_width:
        #     tuned_x_end = output_width
        # if tuned_y_end >output_height:
        #     tuned_y_end=output_height
        # if tuned_x_start<0:
        #     tuned_x_start=0
        # if tuned_y_start<0:
        #     tuned_y_start=0

        # cv2.imwrite(output_path+str(indexofp)+'_original.jpg',img)
        # output_path = '/home/ubuntu/data/Workspace/Soobin/data/test_dataset/'
        # only_num_cropped = img[tuned_y_start:tuned_y_end,tuned_x_start:tuned_x_end]
        # cv2.imwrite(output_path+str(indexofp)+'_output.jpg',only_num_cropped)

        # height, width = only_num_cropped.shape
        # n=3

        # first = only_num_cropped[:,:math.floor(width/3)]
        # second = only_num_cropped[:,math.floor(width/3):math.floor(2*width/3)]
        # third = only_num_cropped[:,math.floor(2*width/3):]

        # padding = int(height*0.15)
        # first = cv2.copyMakeBorder(first,padding,padding,padding,padding,cv2.BORDER_CONSTANT,None,[255,255,255])
        # second = cv2.copyMakeBorder(second,padding,padding,padding,padding,cv2.BORDER_CONSTANT,None,[255,255,255])
        # third = cv2.copyMakeBorder(third,padding,padding,padding,padding,cv2.BORDER_CONSTANT,None,[255,255,255])
        
        # cv2.imwrite(output_path+str(indexofp)+'_first.jpg',first)
        # cv2.imwrite(output_path+str(indexofp)+'_second.jpg',second)
        # cv2.imwrite(output_path+str(indexofp)+'_third.jpg',third)