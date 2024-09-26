import cv2
from imutils import contours
import numpy as np
import math



def segment(result,cntrs,saveToPath,name):
    import os
    tempImage=result.copy()
    list_of_digits=[]
    for i,c in enumerate(cntrs):
        area = cv2.contourArea(c)
        min_area_threshold = 500 
        if area > min_area_threshold:
            box = cv2.boundingRect(c)
            x,y,w,h = box
            temp=tempImage[y:y+h,x:x+w]
            temp=cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(os.path.join(saveToPath,name+f"{i+1}.png"),temp)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
            list_of_digits.append(temp)
    return list_of_digits


def makeContours(temp,kernalSize):
    gray=cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernalSize)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    cntrs = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
    return cntrs

def clearPath(path):
    import os
    import shutil
    for files in os.listdir(path):
        filepath = os.path.join(path, files)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)
    return

def split_digit_from_img(path):
    import os
    from imutils import contours
    img=cv2.imread(path["images"])
    # cropped=img[20:img.shape[0]-20,30:img.shape[1]-20]
    cntrs=makeContours(img,(150,3))
    (cnts, _) = contours.sort_contours(cntrs, method="left-to-right")
    result=segment(img,cnts,path["lines"],"line")
    return result


# path = {
#     "images": "data/test/digit_test.png",
#     "lines": "data/lines/",
#     "words": "data/words/",
#     "letter": "data/letter/"
# }

# split_digit_from_img(path)

def image_refiner(gray):
    org_size = 22
    img_size = 28
    rows,cols = gray.shape
    
    if rows > cols:
        factor = org_size/rows
        rows = org_size
        cols = int(round(cols*factor))        
    else:
        factor = org_size/cols
        cols = org_size
        rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))
    
    #get padding 
    colsPadding = (int(math.ceil((img_size-cols)/2.0)),int(math.floor((img_size-cols)/2.0)))
    rowsPadding = (int(math.ceil((img_size-rows)/2.0)),int(math.floor((img_size-rows)/2.0)))
    
    #apply padding 
    gray = np.pad(gray, (rowsPadding, colsPadding), 'constant')
    return gray

def put_label(t_img,label,x,y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    l_x = int(x) - 10
    l_y = int(y) + 10
    cv2.rectangle(t_img,(l_x,l_y+5),(l_x+35,l_y-35),(0,255,0),-1) 
    cv2.putText(t_img,str(label),(l_x,l_y), font,1.5,(255,0,0),1,cv2.LINE_AA)
    return t_img