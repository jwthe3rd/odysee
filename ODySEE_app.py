import cv2
import numpy as np
import math
import statistics as stats
import pandas as pd
import os
from tkinter import *
from tkinter.filedialog import askopenfilename

SHADER_THRESH = 240
SAFETY_W = 0.05
SAFETY_H = 0.02
BOX_SHIFT_LEFT = 0.1
BOX_SHIFT_UP = 0.1
SIGMA_T = []
MEAN_T = []
SIGMA_M = []
MEAN_M = []
SIGMA_B = []
MEAN_B = []
OKAY_FRAMING = [0]
SHAPE_REGION = 'ellipse'

coords = []


def detect_blink(prev_frame, frame, frame_num):
    #detection_range = frame[:][0:(math.floor(middle_fraction*frame.shape[1]))]
    abs_diff = cv2.absdiff(frame, prev_frame)

    if frame_num == 0:
        return False, frame, abs_diff

    return (np.mean(abs_diff))>=10, frame, abs_diff

def motion_detect_area(frame, frame_0):

    bool_img = (cv2.absdiff(frame, frame_0)) >= 10
    img_mask = bool_img.astype(int) * 255

    return cv2.boundingRect(img_mask)

def edge_detect_area(frame, count):
    # Convert to graycsale
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)

    sobelxy = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    #sobelxy = cv2.Canny(image=img_gray, threshold1=0, threshold2=255)
    ret, thresh = cv2.threshold(sobelxy, SHADER_THRESH, 255, 0)
    thresh = cv2.GaussianBlur(thresh, (51,51),0)
    cv2.imwrite(f"edges_{count}.png", thresh)
    thresh = cv2.cvtColor(cv2.imread(f"edges_{count}.png"), cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours = contours[0] if len(contours) == 2 else contours[1]
    index=0
    cntrs_info = []
    for cntr in contours:
        area = cv2.contourArea(cntr)
        #M = cv2.moments(cntr)
        #cx = int(M["m10"] / M["m00"])
        #cy = int(M["m01"] / M["m00"])
        cntrs_info.append((index,area))
        index = index + 1
        #print(index,area,cx,cy)

    # sort contours by area
    def takeSecond(elem):
        return elem[1]
    cntrs_info.sort(key=takeSecond, reverse=True)

    c = contours[cntrs_info[0][0]]


    return cv2.boundingRect(c)

def click_event(event, x, y, flags, params): 
  
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 

        coords.append((x,y))
        print(f"Mouse clicked at coordinates: ({x}, {y})")
    if event== cv2.EVENT_FLAG_CTRLKEY:
        cv2.destroyAllWindows()


def frame_check(event, x, y, flags, params):

    if event == cv2.EVENT_RBUTTONDOWN:
        OKAY_FRAMING.append(x)
        print("denied")
        cv2.destroyAllWindows()
  

def shader_detect_area(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, np.max(frame) - SHADER_THRESH, 255, 0)
    thresh = cv2.blur(thresh, (100,100))

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c = max(contours, key = cv2.contourArea)

    return cv2.boundingRect(c)

def generate_weak_labels(frame, og_frame):

    frame_blank = np.zeros(shape=(frame.shape[1],frame.shape[1],3))
    og_frame_blank = np.zeros(shape=(og_frame.shape[1],og_frame.shape[1]))

    frame_blank[0:frame.shape[0]][0:frame.shape[1]] = frame[0:frame.shape[0]][0:frame.shape[1]]
    og_frame_blank[0:og_frame.shape[0]][0:og_frame.shape[1]] = og_frame[0:og_frame.shape[0]][0:og_frame.shape[1]]
    #og_frame_blank = cv2.cvtColor(og_frame_blank, cv2.COLOR_RGB2GRAY)

    # Remember -> OpenCV stores things in BGR order
    lowerBound = np.asarray((0, 250, 0));
    upperBound = np.asarray((1, 255, 1));

    # this gives you the mask for those in the ranges you specified,
    # but you want the inverse, so we'll add bitwise_not...
    frame_blank = cv2.inRange(frame_blank, lowerBound, upperBound)

    contours, hierarchy = cv2.findContours(frame_blank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)

    x,y,w,h = cv2.boundingRect(c)

    for i in range(x,x+w):
        for j in range(y, y+h):
            #print(frame_blank[i][j])
            frame_blank[j][i] = 255

    print(og_frame_blank.shape)
    print(frame_blank.shape)

    frame_blank = cv2.GaussianBlur(frame_blank, (71,71), 0)
    catted = np.concatenate((frame_blank.T,og_frame_blank.T)).T


    return catted#feats_and_labels

def user_define_area(frame, count):

    if SHAPE_REGION == 'rect':

        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (0,math.floor(frame.shape[0]/6)) 
        fontScale = 1
        color = (0, 255, 100) 
        thickness = 2

        display_frame = cv2.putText(np.copy(frame), 'L-click 4 points for box ROI', org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 

        cv2.imshow(f"test{count}",display_frame)
        cv2.setMouseCallback(f"test{count}", click_event)
        cv2.waitKey()
        min_x = float("inf")
        min_y = float("inf")
        max_x = 0
        max_y = 0

        for coord in coords:
            if coord[0] > max_x:
                max_x = coord[0]
            if coord[0] < min_x:
                min_x = coord[0]
            if coord[1] > max_y:
                max_y = coord[1]
            if coord[1] < min_y:
                min_y = coord[1]
        w, h = abs(max_x - min_x), abs(max_y - min_y)

        coords.clear()

    elif SHAPE_REGION == 'ellipse':
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (0,math.floor(frame.shape[0]/6)) 
        fontScale = 1
        color = (0, 255, 100) 
        thickness = 2

        display_frame = cv2.putText(np.copy(frame), 'L-click 4 points for box ROI', org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 

        cv2.imshow(f"test{count}",display_frame)
        cv2.setMouseCallback(f"test{count}", click_event)
        cv2.waitKey()
        min_x = float("inf")
        min_y = float("inf")
        max_x = 0
        max_y = 0

        coord_x = 0
        coord_y = 0


        for coord in coords:
            coord_x += coord[0]
            coord_y += coord[1]
            if coord[0] > max_x:
                max_x = coord[0]
            if coord[0] < min_x:
                min_x = coord[0]
            if coord[1] > max_y:
                max_y = coord[1]
            if coord[1] < min_y:
                min_y = coord[1]
        w, h = abs(max_x - min_x), abs(max_y - min_y)
        min_x = math.floor((max_x+min_x)/2)
        min_y = math.floor((max_y+min_y)/2)
        print(w)
        print(h)

        coords.clear()



    return min_x, min_y, w, h

def conv_define_area():
    return 0

def get_stats(frame, bb):

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    print(bb)
    sigma_img = gray[bb[0][1]:bb[1][1],bb[0][0]:bb[1][0]]

    return np.std(sigma_img), np.mean(sigma_img)

def get_bounding_boxes(x, y, w, h, frame):

    if SHAPE_REGION == 'rect':

        safety_w = math.floor(w*SAFETY_W)
        safety_h = math.floor(h*SAFETY_H)

        box_wt = math.floor(w - 12*safety_w)
        box_wm = math.floor(w - 2*safety_w)
        box_h = math.floor(h/3)

        boxt = [(x+6*safety_w,y+safety_h),(box_wt,box_h)]
        boxm = [(x+safety_w,y+box_h+safety_h),(box_wm,box_h)]
        boxb = [(x+6*safety_w,y+2*box_h+safety_h),(box_wt,box_h)]

        bounding_box_t = [(boxt[0][0], boxt[0][1]),(boxt[0][0] + boxt[1][0], boxt[0][1]+boxt[1][1])]
        bounding_box_m = [(boxm[0][0], boxm[0][1]),(boxm[0][0] + boxm[1][0], boxm[0][1]+boxm[1][1])]
        bounding_box_b = [(boxb[0][0], boxb[0][1]),(boxb[0][0] + boxb[1][0], boxb[0][1]+boxb[1][1])]

        sigma_t, mean_t = get_stats(frame, bounding_box_t)
        sigma_m, mean_m = get_stats(frame, bounding_box_m)
        sigma_b, mean_b = get_stats(frame, bounding_box_b)

        # draw the biggest contour (c) in green
        placeholder_frame = np.copy(frame)
        cv2.rectangle(placeholder_frame,(x, y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(placeholder_frame,bounding_box_t[0],bounding_box_t[1],(0,255,0),2)
        cv2.rectangle(placeholder_frame,bounding_box_m[0],bounding_box_m[1],(0,255,0),2)
        cv2.rectangle(placeholder_frame,bounding_box_b[0],bounding_box_b[1],(0,255,0),2)
    
    elif SHAPE_REGION == 'ellipse':
        x-= math.floor(0.5*w)
        y-=math.floor(0.5*h)
        safety_w = math.floor(w*SAFETY_W)
        safety_h = math.floor(h*SAFETY_H)

        box_wt = math.floor(w - 12*safety_w)
        box_wm = math.floor(w - 2*safety_w)
        box_h = math.floor(h/3)

        boxt = [(x+6*safety_w,y+safety_h),(box_wt,box_h)]
        boxm = [(x+safety_w,y+box_h+safety_h),(box_wm,box_h)]
        boxb = [(x+6*safety_w,y+2*box_h+safety_h),(box_wt,box_h)]

        bounding_box_t = [(boxt[0][0], boxt[0][1]),(boxt[0][0] + boxt[1][0], boxt[0][1]+boxt[1][1])]
        bounding_box_m = [(boxm[0][0], boxm[0][1]),(boxm[0][0] + boxm[1][0], boxm[0][1]+boxm[1][1])]
        bounding_box_b = [(boxb[0][0], boxb[0][1]),(boxb[0][0] + boxb[1][0], boxb[0][1]+boxb[1][1])]

        sigma_t, mean_t = get_stats(frame, bounding_box_t)
        sigma_m, mean_m = get_stats(frame, bounding_box_m)
        sigma_b, mean_b = get_stats(frame, bounding_box_b)

        # draw the biggest contour (c) in green
        placeholder_frame = np.copy(frame)
        print(w)
        print(h)
        cv2.ellipse(placeholder_frame,(x+math.floor(0.5*w), y+math.floor(0.5*h)),(w//2,h//2),0.,0.,360.,(0,255,0))
        cv2.rectangle(placeholder_frame,bounding_box_t[0],bounding_box_t[1],(0,255,0),2)
        cv2.rectangle(placeholder_frame,bounding_box_m[0],bounding_box_m[1],(0,255,0),2)
        cv2.rectangle(placeholder_frame,bounding_box_b[0],bounding_box_b[1],(0,255,0),2)


    return placeholder_frame, mean_t, sigma_t, mean_m, sigma_m, mean_b, sigma_b

def test_recursive_selection(frame, count, area_definer):

    x, y, w, h = area_definer(frame, count)
    OKAY_FRAMING.clear()

    placeholder_frame, mean_t, sigma_t, mean_m, sigma_m, mean_b, sigma_b = get_bounding_boxes(x,y,w,h,frame)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (0,math.floor(placeholder_frame.shape[0]/6)) 
    fontScale = 1
    color = (0, 255, 100) 
    thickness = 2

    display_frame = cv2.putText(np.copy(placeholder_frame), 'R-click to retry', org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 
    display_frame = cv2.putText(display_frame, 'Enter to accept boxes', (math.floor(3*placeholder_frame.shape[1]/6),math.floor(15*placeholder_frame.shape[0]/16)), font,  
                    fontScale, color, thickness, cv2.LINE_AA) 
    cv2.imshow("f",display_frame)

    cv2.setMouseCallback("f", frame_check)

    cv2.waitKey()

    return x,y,w,h,placeholder_frame, mean_t, sigma_t, mean_m, sigma_m, mean_b, sigma_b


def grab_pixel_data(video_dir, name, check_counts, area_definer):
    cap = cv2.VideoCapture(f'{video_dir}')
    count = 0
    counter = 0
    frame_number = 0
    name = '_'.join(name)
    
    while cap.isOpened():
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        count+=1

        if not(ret):
            break

        if count == 0:
            frame_0 = frame

        if count in check_counts:

            if frame_number % 5 == 0:

                while OKAY_FRAMING:
                    x,y,w,h, placeholder_frame, mean_t, sigma_t, mean_m, sigma_m, mean_b, sigma_b = test_recursive_selection(frame, count, area_definer)
            else:
                placeholder_frame, mean_t, sigma_t, mean_m, sigma_m, mean_b, sigma_b = get_bounding_boxes(x,y,w,h,frame)

            SIGMA_T.append(sigma_t)
            MEAN_T.append(mean_t)
            SIGMA_M.append(sigma_m)
            MEAN_M.append(mean_m)
            
            SIGMA_B.append(sigma_b)
            MEAN_B.append(mean_b)

            placeholder_frame = generate_weak_labels(placeholder_frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            
            cv2.imwrite(f"pics/{name}_{count}.png", placeholder_frame)

            OKAY_FRAMING.append(0)
                


            
            cv2.destroyAllWindows()
            frame_number+=1

    df = pd.DataFrame({"Top Mean": MEAN_T, "Top Stdev": SIGMA_T,
                  "Middle Mean": MEAN_M, "Middle Stdev": SIGMA_M,
                  "Bottom Mean": MEAN_B, "Bottom Stdev": SIGMA_B})
    
    
    df.to_csv(f"data/{name}_data.csv")
            

    cap.release()
    return 0

def detect_blink_frames(video_folder):
    cap = cv2.VideoCapture(f'{video_folder}')
    frame_avg = 0
    count = 0
    blinks = []
    diffs = []
    reading_video = True
    while reading_video:
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not(ret):
            reading_video = False
        if count == 0:
            prev_frame = frame
            frame_0 = frame
        if ret:
            blink, prev_frame, abs_diff = detect_blink(prev_frame,frame,count)
            if blink:
                blinks.append(count)
                diffs.append(frame)
            count+=1

    cap.release()

    check_count = []
    diff_frames = []
    frames = []

    for i, val in enumerate(blinks):
        if i > 0:
            if (val - blinks[i-1]) > math.floor(5*fps)+2:

                check_count.append(blinks[i-1]+fps)
                check_count.append(blinks[i-1]+2*fps)
                check_count.append(blinks[i-1]+3*fps)
                check_count.append(blinks[i-1]+4*fps)
                check_count.append(blinks[i-1]+5*fps)
                curr_max = 0
                
                frames = []
            else:
                frames.append(val)

    return check_count


if __name__ == "__main__":

    root = Tk()
    video_folder = []
    area_function = []

    root.title("ODySEE")
    root.geometry('800x200')

    def click_vid_select():
        video_folder.append(askopenfilename())
        lbl_file =Label(root, text = f'{os.path.basename(video_folder[-1])}')
        lbl_file.grid(column=2,row=1)
    
    def run_task():
        check_count = detect_blink_frames(video_folder[-1])
        SIGMA_T.clear()
        MEAN_T.clear()
        SIGMA_M.clear()
        MEAN_M.clear()
        SIGMA_B.clear()
        MEAN_B.clear()
        grab_pixel_data(video_folder[-1],os.path.basename(video_folder[-1]).split('.')[0].split(),check_count, area_function[-1])
    
    def manual_area():
        area_function.append(user_define_area)
        btn = Button(root, text="Manual Area Detection", fg="blue",command=manual_area)
        btn.grid(column=1, row=2)
        btn = Button(root, text="ConvNet Area Detection (Not Available)", fg="black",command=ai_area)
        btn.grid(column=2, row=2)
    
    def ai_area():
        area_function.append(conv_define_area)
        btn = Button(root, text="Manual Area Detection", fg="black",command=manual_area)
        btn.grid(column=1, row=2)
        btn = Button(root, text="ConvNet Area Detection (Not Available)", fg="blue",command=ai_area)
        btn.grid(column=2, row=2)

    lbl = Label(root, text = "Welcome to ODySEE!")
    lbl.grid()
    lbl2 = Label(root, text = "Please select a video file")
    lbl2.grid()

    btn = Button(root, text="Select video file for analysis", fg="black",command=click_vid_select)
    btn.grid(column=1, row=1)
    lbl3 = Label(root, text = "Please select method of area definition")
    lbl3.grid()
    btn = Button(root, text="Manual Area Detection", fg="black",command=manual_area)
    btn.grid(column=1, row=2)
    btn = Button(root, text="ConvNet Area Detection (Not Available)", fg="black",command=ai_area)
    btn.grid(column=2, row=2)
    lbl4 = Label(root, text = "Press to start analysis")
    lbl4.grid()
    btn = Button(root, text="Run analysis", fg="black",command=run_task)
    btn.grid(column=1, row=3)
    root.mainloop()