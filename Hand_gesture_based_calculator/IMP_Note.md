A Note on some IMP methods

capture_histogram()  # this method captures histogram and saves in hist (Not required if ure gonna only use
background subtraction to get mask)

find_object2(frame, object_hist)  # Uses histogram to get the mask  (not required if ure only gonna use back sub)

find_object_backsub(imb, imf,filtert)  # finds mask using custom background subtraction

find_object_hsv(frame,vals)   # finds mask using hsv color extraction ( not required if ure only gonna use back sub)

If you want line by line explanation of this program then you should join my Classical computer vision course
fb.com/bleed-ai

