import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import random 

#colors = [(114, 180, 126), (201, 241, 101), (112, 122, 135), (205, 91, 70), (90, 235, 189), (176, 189, 163), (82, 74, 70), (238, 238, 231), (102, 250, 116), (79, 99, 108), (123, 152, 208), (170, 142, 163), (189, 79, 88), (239, 74, 122), (179, 125, 117), (225, 233, 229), (227, 206, 225), (211, 103, 105), (105, 103, 149), (130, 123, 195), (248, 191, 134), (164, 225, 220), (70, 108, 245), (182, 239, 174), (160, 150, 79), (243, 137, 156), (97, 199, 155), (186, 78, 226), (204, 109, 114), (199, 197, 148), (238, 135, 120), (163, 202, 159), (181, 108, 98), (126, 180, 119), (133, 113, 248), (70, 101, 77), (197, 251, 78), (198, 96, 129), (191, 122, 103), (183, 132, 250), (200, 230, 88), (161, 132, 102), (73, 248, 159), (155, 152, 95), (184, 247, 128), (124, 113, 99), (242, 99, 142), (131, 94, 188), (209, 183, 95), (226, 212, 70), (161, 160, 176), (168, 211, 177), (122, 155, 112), (149, 183, 243), (145, 129, 253), (130, 184, 244), (215, 211, 137), (98, 118, 163), (199, 92, 137), (176, 203, 156), (215, 237, 239), (120, 100, 132), (175, 253, 140), (166, 93, 92), (96, 212, 187), (143, 118, 131), (80, 248, 188), (235, 126, 168), (120, 175, 202), (95, 135, 211), (125, 89, 198), (101, 76, 86), (77, 129, 75), (85, 103, 200), (177, 239, 87), (195, 94, 85), (99, 249, 96), (85, 198, 245), (201, 130, 114), (106, 193, 191), (122, 113, 116), (231, 249, 184), (129, 177, 195), (97, 221, 92), (152, 70, 235), (224, 237, 241), (172, 102, 81), (73, 229, 248), (112, 92, 169), (162, 215, 244), (177, 108, 140), (90, 160, 116), (189, 181, 223), (145, 203, 185), (177, 233, 204), (215, 194, 82), (139, 82, 135), (197, 71, 209), (132, 104, 126), (173, 215, 115), (233, 216, 84), (213, 167, 91), (105, 116, 158), (247, 194, 86), (165, 154, 168), (176, 118, 157), (169, 170, 129), (214, 76, 203), (82, 148, 222), (119, 186, 248), (74, 247, 130), (112, 103, 110), (114, 223, 73), (131, 208, 180), (105, 138, 130), (94, 242, 160), (164, 192, 98), (133, 180, 232), (211, 107, 150), (72, 183, 245), (205, 120, 78), (234, 196, 244), (132, 219, 145), (160, 128, 93), (98, 148, 244), (236, 220, 139), (224, 201, 254), (92, 156, 85), (220, 80, 252), (201, 81, 203), (113, 105, 133), (78, 206, 116), (223, 255, 109), (152, 129, 229), (190, 163, 187), (145, 221, 254), (143, 189, 120), (93, 236, 134), (252, 194, 199), (94, 93, 132), (205, 130, 230), (210, 223, 216), (237, 167, 240), (94, 79, 202), (173, 77, 205), (151, 164, 111), (250, 118, 215), (234, 120, 113), (211, 148, 127), (80, 73, 154), (110, 176, 215), (227, 219, 127), (162, 235, 109), (76, 151, 229), (160, 179, 251), (216, 120, 83), (80, 237, 99), (241, 227, 151), (215, 136, 162), (241, 131, 80), (101, 136, 169), (170, 118, 153), (255, 89, 119), (208, 149, 200), (143, 224, 127), (93, 193, 236), (196, 250, 96), (77, 178, 214), (221, 148, 249), (106, 112, 184), (167, 108, 205), (126, 101, 177), (229, 201, 172), (111, 110, 181), (179, 142, 244), (135, 235, 208), (251, 81, 244), (85, 91, 236), (97, 146, 182), (74, 136, 228), (190, 160, 155), (128, 70, 174), (219, 102, 173), (210, 188, 153), (232, 163, 250), (218, 251, 184), (129, 184, 231), (223, 115, 247), (144, 147, 78), (99, 149, 126), (148, 248, 242), (166, 185, 140), (221, 128, 166), (226, 92, 216), (182, 183, 108), (132, 82, 163), (110, 254, 198), (121, 147, 231), (190, 240, 189), (255, 129, 117)]

colors = [(r,g,b) for r in range(70,200) for g in range(70,200) for b in range(70,200) ]

def create_possible_center_coordinates(image):
    h,w,_ = image.shape
    h_range_start = int(h/3)
    h_range_end = int(h-h_range_start)
    center_w = int(w/2)
    return [ (center_w,y) for y in range(h_range_start,h_range_end,5) ]

def create_possible_center_coordinates_for_rect(image):
    h,w,_ = image.shape
    h_range_start = int(h/3)
    h_range_end = int(h-h_range_start)
    #center_w = int(w/2)

    #w_range_start =  int(w/3)
    #w_range_end = int(w/2)
    center_w = int(w/2)
    return [ (center_w,y) for y in range(h_range_start,h_range_end,5) ]
    #return [ (x,y) for y in range(h_range_start,h_range_end,5) for x in range(w_range_start,w_range_end) ]



def create_circle(image ,images_with_defect_on_left_side_count,images_with_defect_on_right_side_count,images_without_defect_count, images_without_defect_with_dot_count):
    


    h,w,_ = image.shape

    minor_axis_length_min = int(h/5)
    minor_axis_length_max = int(h/4)
    major_axis_lenght_min = int(w/3) 
    major_axis_lenght_max = int(w/2)

    possible_axis_lengths = [ (x,y) for x in range(major_axis_lenght_min, major_axis_lenght_max, 5) for y in range(minor_axis_length_min, minor_axis_length_max,5)]

    possible_center_coordinates = create_possible_center_coordinates(image)

    images_without_defect = []
    images_with_defect_on_left = []
    images_with_defect_on_right = []
    images_without_defect_with_dot = []

    print("# defect on right_side only")
    for count in range(0,images_with_defect_on_right_side_count):
        center_coordinates = random.choice(possible_center_coordinates)
        color = random.choice(colors)
        created = cv2.ellipse(image.copy(), center_coordinates, random.choice(possible_axis_lengths), 0.0, 0.0, 360.0, color, -1);
        crop_left = created.copy()[:,0:int(w/2)]
        crop_right = created.copy()[:, int(w/2):]

        checking_for_pixel = True
        pix_x, pix_y = 0,0
        while(checking_for_pixel):
            pix_x = random.choice(range(0,crop_right.shape[1]))
            pix_y =  random.choice(range(0,crop_right.shape[0]))
            if np.all(crop_right[pix_x, pix_y] == color ):
                checking_for_pixel = False
                if pix_x >=10:
                    pix_x = pix_x+10
                if pix_y >= 10:
                    pix_y = pix_y - 10
                if np.all(crop_right[pix_x,pix_y-10] != color):
                    checking_for_pixel =True
                if np.all(crop_right[pix_x,pix_y+10] != color):
                    checking_for_pixel =True
        crop_right = cv2.circle(crop_right, (pix_y,pix_x), 10, (0,255,0), -1) 
        images_with_defect_on_right.append( (crop_left, crop_right,3) )
    
    
    print("# defect on left_side only")
    for count in range(0,images_with_defect_on_left_side_count):
        center_coordinates = random.choice(possible_center_coordinates)
        color = random.choice(colors)
        created = cv2.ellipse(image.copy(), center_coordinates, random.choice(possible_axis_lengths), 0.0, 0.0, 360.0, color, -1);
        crop_left = created.copy()[:,0:int(w/2)]
        crop_right = created.copy()[:, int(w/2):]

        checking_for_pixel = True
        pix_x, pix_y = 0,0
        while(checking_for_pixel):
            pix_x = random.choice(range(0,crop_right.shape[1]))
            pix_y =  random.choice(range(0,crop_right.shape[0]))
            if np.all(crop_right[pix_x, pix_y] == color ):
                checking_for_pixel = False
                if pix_x >=10:
                    pix_x = pix_x+10
                if pix_y >= 10:
                    pix_y = pix_y - 10
                if np.all(crop_right[pix_x,pix_y-10] != color):
                    checking_for_pixel =True
                if np.all(crop_right[pix_x,pix_y+10] != color):
                    checking_for_pixel =True
        crop_left = cv2.circle(crop_right.copy(), (pix_y,pix_x), 10, (0,255,0), -1) 
        crop_left = cv2.flip(crop_left, 1)
        images_with_defect_on_left.append( (crop_left, crop_right,2) )


    print("# defect on both side")
    for count in range(0,images_without_defect_with_dot_count):
        center_coordinates = random.choice(possible_center_coordinates)
        color = random.choice(colors)
        created = cv2.ellipse(image.copy(), center_coordinates, random.choice(possible_axis_lengths), 0.0, 0.0, 360.0, color, -1);
        crop_left = created.copy()[:,0:int(w/2)]
        crop_right = created.copy()[:, int(w/2):]

        checking_for_pixel = True
        pix_x, pix_y = 0,0
        while(checking_for_pixel):
            pix_x = random.choice(range(0,crop_right.shape[1]))
            pix_y =  random.choice(range(0,crop_right.shape[0]))
            if np.all(crop_right[pix_x, pix_y] == color ):
                checking_for_pixel = False
                if pix_x >=10:
                    pix_x = pix_x+10
                if pix_y >= 10:
                    pix_y = pix_y - 10
                if np.all(crop_right[pix_x,pix_y-10] != color):
                    checking_for_pixel =True
                if np.all(crop_right[pix_x,pix_y+10] != color):
                    checking_for_pixel =True
        crop_right = cv2.circle(crop_right.copy(), (pix_y,pix_x), 10, (0,255,0), -1) 
        crop_left = cv2.flip(crop_right, 1)
        images_without_defect_with_dot.append( (crop_left, crop_right,1) )

    print("# No defect on both side")
    for count in range(0,images_without_defect_count):
        center_coordinates = random.choice(possible_center_coordinates)
        color = random.choice(colors)
        created = cv2.ellipse(image.copy(), center_coordinates, random.choice(possible_axis_lengths), 0.0, 0.0, 360.0, color, -1);
        crop_left = created.copy()[:,0:int(w/2)]
        crop_right = created.copy()[:, int(w/2):]

        checking_for_pixel = True
        pix_x, pix_y = 0,0
        while(checking_for_pixel):
            pix_x = random.choice(range(0,crop_right.shape[1]))
            pix_y =  random.choice(range(0,crop_right.shape[0]))
            if np.all(crop_right[pix_x, pix_y] == color ):
                checking_for_pixel = False
                if pix_x >=10:
                    pix_x = pix_x+10
                if pix_y >= 10:
                    pix_y = pix_y - 10
                if np.all(crop_right[pix_x,pix_y-10] != color):
                    checking_for_pixel =True
                if np.all(crop_right[pix_x,pix_y+10] != color):
                    checking_for_pixel =True
        #crop_right = cv2.circle(crop_right.copy(), (pix_y,pix_x), 10, (0,255,0), -1) 
        #crop_left = cv2.flip(crop_right, 1)
        images_without_defect.append( (crop_left, crop_right,0) )

    return images_without_defect,images_without_defect_with_dot,images_with_defect_on_left,images_with_defect_on_right


def create_rectangle(image ,images_with_defect_on_left_side_count,images_with_defect_on_right_side_count,images_without_defect_count, images_without_defect_with_dot_count):
    


    h,w,_ = image.shape

    minor_axis_length_min = int(h/5)
    minor_axis_length_max = int(h/4)
    major_axis_lenght_min = int(w/3) 
    major_axis_lenght_max = int(w/2)

    possible_axis_lengths = [ (int(x), int(y)) for x in range(major_axis_lenght_min, major_axis_lenght_max, 5) for y in range(minor_axis_length_min, minor_axis_length_max,5)]

    possible_center_coordinates = create_possible_center_coordinates_for_rect(image)

    images_without_defect = []
    images_with_defect_on_left = []
    images_with_defect_on_right = []
    images_without_defect_with_dot = []

    print("# defect on right_side only")
    for count in range(0,images_with_defect_on_right_side_count):
        center_coordinates = random.choice(possible_center_coordinates)
        color = random.choice(colors)
        #created = cv2.ellipse(image.copy(), center_coordinates, random.choice(possible_axis_lengths), 0.0, 0.0, 360.0, color, -1);
        
        x,y = random.choice(possible_axis_lengths)
        x = random.choice( [ int(x/1.4), int(x/1.3), int(x/1.2), int(x/1.1) ] )
        start_point = (center_coordinates[0] - x,center_coordinates[1] - y)
        end_point = (center_coordinates[0] + x,center_coordinates[1] + y)
        created = cv2.rectangle(image.copy(), start_point, end_point, color, -1) 
 

        crop_left = created.copy()[:,0:int(w/2)]
        crop_right = created.copy()[:, int(w/2):]

        checking_for_pixel = True
        pix_x, pix_y = 0,0
        while(checking_for_pixel):
            pix_x = random.choice(range(0,crop_right.shape[1]))
            pix_y =  random.choice(range(0,crop_right.shape[0]))
            if np.all(crop_right[pix_x, pix_y] == color ):
                checking_for_pixel = False
                if pix_x >=10:
                    pix_x = pix_x+10
                if pix_y >= 10:
                    pix_y = pix_y - 10
                if np.all(crop_right[pix_x,pix_y-10] != color):
                    checking_for_pixel =True
                if np.all(crop_right[pix_x,pix_y+10] != color):
                    checking_for_pixel =True
        crop_right = cv2.circle(crop_right, (pix_y,pix_x), 10, (0,255,0), -1) 
        images_with_defect_on_right.append( (crop_left, crop_right,3) )
    
    
    print("# defect on left_side only")
    for count in range(0,images_with_defect_on_left_side_count):
        center_coordinates = random.choice(possible_center_coordinates)
        color = random.choice(colors)
        #created = cv2.ellipse(image.copy(), center_coordinates, random.choice(possible_axis_lengths), 0.0, 0.0, 360.0, color, -1);
        
        x,y = random.choice(possible_axis_lengths)
        x = random.choice( [ int(x/1.4), int(x/1.3), int(x/1.2), int(x/1.1) ] )
        start_point = (center_coordinates[0] - x,center_coordinates[1] - y)
        end_point = (center_coordinates[0] + x,center_coordinates[1] + y)
        created = cv2.rectangle(image.copy(), start_point, end_point, color, -1) 
 
        crop_left = created.copy()[:,0:int(w/2)]
        crop_right = created.copy()[:, int(w/2):]

        checking_for_pixel = True
        pix_x, pix_y = 0,0
        while(checking_for_pixel):
            pix_x = random.choice(range(0,crop_right.shape[1]))
            pix_y =  random.choice(range(0,crop_right.shape[0]))
            if np.all(crop_right[pix_x, pix_y] == color ):
                checking_for_pixel = False
                if pix_x >=10:
                    pix_x = pix_x+10
                if pix_y >= 10:
                    pix_y = pix_y - 10
                if np.all(crop_right[pix_x,pix_y-10] != color):
                    checking_for_pixel =True
                if np.all(crop_right[pix_x,pix_y+10] != color):
                    checking_for_pixel =True
        crop_left = cv2.circle(crop_right.copy(), (pix_y,pix_x), 10, (0,255,0), -1) 
        crop_left = cv2.flip(crop_left, 1)
        images_with_defect_on_left.append( (crop_left, crop_right,2) )


    print("# defect on both side")
    for count in range(0,images_without_defect_with_dot_count):
        center_coordinates = random.choice(possible_center_coordinates)
        color = random.choice(colors)
        
        #created = cv2.ellipse(image.copy(), center_coordinates, random.choice(possible_axis_lengths), 0.0, 0.0, 360.0, color, -1);
        
        x,y = random.choice(possible_axis_lengths)
        x = random.choice( [ int(x/1.4), int(x/1.3), int(x/1.2), int(x/1.1) ] )
        start_point = (center_coordinates[0] - x,center_coordinates[1] - y)
        end_point = (center_coordinates[0] + x,center_coordinates[1] + y)
        created = cv2.rectangle(image.copy(), start_point, end_point, color, -1) 
 
        crop_left = created.copy()[:,0:int(w/2)]
        crop_right = created.copy()[:, int(w/2):]

        checking_for_pixel = True
        pix_x, pix_y = 0,0
        while(checking_for_pixel):
            pix_x = random.choice(range(0,crop_right.shape[1]))
            pix_y =  random.choice(range(0,crop_right.shape[0]))
            if np.all(crop_right[pix_x, pix_y] == color ):
                checking_for_pixel = False
                if pix_x >=10:
                    pix_x = pix_x+10
                if pix_y >= 10:
                    pix_y = pix_y - 10
                if np.all(crop_right[pix_x,pix_y-10] != color):
                    checking_for_pixel =True
                if np.all(crop_right[pix_x,pix_y+10] != color):
                    checking_for_pixel =True
        crop_right = cv2.circle(crop_right.copy(), (pix_y,pix_x), 10, (0,255,0), -1) 
        crop_left = cv2.flip(crop_right, 1)
        images_without_defect_with_dot.append( (crop_left, crop_right,1) )

    print("# No defect on both side")
    for count in range(0,images_without_defect_count):
        center_coordinates = random.choice(possible_center_coordinates)
        color = random.choice(colors)
        #created = cv2.ellipse(image.copy(), center_coordinates, random.choice(possible_axis_lengths), 0.0, 0.0, 360.0, color, -1);
        
        x,y = random.choice(possible_axis_lengths)
        x = random.choice( [ int(x/1.4), int(x/1.3), int(x/1.2), int(x/1.1) ] )
        start_point = (center_coordinates[0] - x,center_coordinates[1] - y)
        end_point = (center_coordinates[0] + x,center_coordinates[1] + y)
        created = cv2.rectangle(image.copy(), start_point, end_point, color, -1) 
 
        
        crop_left = created.copy()[:,0:int(w/2)]
        crop_right = created.copy()[:, int(w/2):]

        checking_for_pixel = True
        pix_x, pix_y = 0,0
        while(checking_for_pixel):
            pix_x = random.choice(range(0,crop_right.shape[1]))
            pix_y =  random.choice(range(0,crop_right.shape[0]))
            if np.all(crop_right[pix_x, pix_y] == color ):
                checking_for_pixel = False
                if pix_x >=10:
                    pix_x = pix_x+10
                if pix_y >= 10:
                    pix_y = pix_y - 10
                if np.all(crop_right[pix_x,pix_y-10] != color):
                    checking_for_pixel =True
                if np.all(crop_right[pix_x,pix_y+10] != color):
                    checking_for_pixel =True
        #crop_right = cv2.circle(crop_right.copy(), (pix_y,pix_x), 10, (0,255,0), -1) 
        #crop_left = cv2.flip(crop_right, 1)
        images_without_defect.append( (crop_left, crop_right,0) )

    return images_without_defect,images_without_defect_with_dot,images_with_defect_on_left,images_with_defect_on_right


def create_triangle(image ,images_with_defect_on_left_side_count,images_with_defect_on_right_side_count,images_without_defect_count, images_without_defect_with_dot_count):
    
    h,w,_ = image.shape

    minor_axis_length_min = int(h/3.5)
    minor_axis_length_max = int(h/3)
    major_axis_lenght_min = int(w/3) 
    major_axis_lenght_max = int(w/2)

    possible_axis_lengths = [ (int(x), int(y)) for x in range(major_axis_lenght_min, major_axis_lenght_max, 5) for y in range(minor_axis_length_min, minor_axis_length_max,5)]

    possible_center_coordinates = create_possible_center_coordinates_for_rect(image)

    images_without_defect = []
    images_with_defect_on_left = []
    images_with_defect_on_right = []
    images_without_defect_with_dot = []

    print("# defect on right_side only")
    for count in range(0,images_with_defect_on_right_side_count):
        center_coordinates = random.choice(possible_center_coordinates)
        color = random.choice(colors)
        #created = cv2.ellipse(image.copy(), center_coordinates, random.choice(possible_axis_lengths), 0.0, 0.0, 360.0, color, -1);
        
        x,y = random.choice(possible_axis_lengths)
        x = random.choice( [ int(x/1.4), int(x/1.3), int(x/1.2), int(x/1.1) ] )
        point_one = (center_coordinates[0],center_coordinates[1] - y)
        point_two  = (center_coordinates[0] - x,center_coordinates[1] )
        point_three  = (center_coordinates[0] + x,center_coordinates[1] )
        triangle = np.array([point_one, point_two, point_three,point_one])
        created = cv2.fillConvexPoly(image.copy(), triangle, color)


        crop_left = created.copy()[:,0:int(w/2)]
        crop_right = created.copy()[:, int(w/2):]

        checking_for_pixel = True
        pix_x, pix_y = 0,0
        while(checking_for_pixel):
            pix_x = random.choice(range(0,crop_right.shape[1]))
            pix_y =  random.choice(range(0,crop_right.shape[0]))
            if np.all(crop_right[pix_x, pix_y] == color ):
                checking_for_pixel = False
                if pix_x >=10:
                    pix_x = pix_x+10
                if pix_y >= 10:
                    pix_y = pix_y - 10
                if np.all(crop_right[pix_x,pix_y-15] != color):
                    checking_for_pixel =True
                if np.all(crop_right[pix_x,pix_y+15] != color):
                    checking_for_pixel =True
        crop_right = cv2.circle(crop_right, (pix_y,pix_x), 6, (0,255,0), -1) 
        images_with_defect_on_right.append( (crop_left, crop_right,3) )
    
    
    print("# defect on left_side only")
    for count in range(0,images_with_defect_on_left_side_count):
        center_coordinates = random.choice(possible_center_coordinates)
        color = random.choice(colors)
        #created = cv2.ellipse(image.copy(), center_coordinates, random.choice(possible_axis_lengths), 0.0, 0.0, 360.0, color, -1);
        
        x,y = random.choice(possible_axis_lengths)
        x = random.choice( [ int(x/1.4), int(x/1.3), int(x/1.2), int(x/1.1) ] )
        point_one = (center_coordinates[0],center_coordinates[1] - y)
        point_two  = (center_coordinates[0] - x,center_coordinates[1] )
        point_three  = (center_coordinates[0] + x,center_coordinates[1] )
        triangle = np.array([point_one, point_two, point_three,point_one])
        created = cv2.fillConvexPoly(image.copy(), triangle, color)

        crop_left = created.copy()[:,0:int(w/2)]
        crop_right = created.copy()[:, int(w/2):]

        checking_for_pixel = True
        pix_x, pix_y = 0,0
        while(checking_for_pixel):
            pix_x = random.choice(range(0,crop_right.shape[1]))
            pix_y =  random.choice(range(0,crop_right.shape[0]))
            if np.all(crop_right[pix_x, pix_y] == color ):
                checking_for_pixel = False
                if pix_x >=10:
                    pix_x = pix_x+10
                if pix_y >= 10:
                    pix_y = pix_y - 10
                if np.all(crop_right[pix_x,pix_y-15] != color):
                    checking_for_pixel =True
                if np.all(crop_right[pix_x,pix_y+15] != color):
                    checking_for_pixel =True
        crop_left = cv2.circle(crop_right.copy(), (pix_y,pix_x), 6, (0,255,0), -1) 
        crop_left = cv2.flip(crop_left, 1)
        images_with_defect_on_left.append( (crop_left, crop_right,2) )


    print("# defect on both side")
    for count in range(0,images_without_defect_with_dot_count):
        center_coordinates = random.choice(possible_center_coordinates)
        color = random.choice(colors)
        
        #created = cv2.ellipse(image.copy(), center_coordinates, random.choice(possible_axis_lengths), 0.0, 0.0, 360.0, color, -1);
        
        x,y = random.choice(possible_axis_lengths)
        x = random.choice( [ int(x/1.4), int(x/1.3), int(x/1.2), int(x/1.1) ] )
        point_one = (center_coordinates[0],center_coordinates[1] - y)
        point_two  = (center_coordinates[0] - x,center_coordinates[1] )
        point_three  = (center_coordinates[0] + x,center_coordinates[1] )
        triangle = np.array([point_one, point_two, point_three,point_one])
        created = cv2.fillConvexPoly(image.copy(), triangle, color)

        crop_left = created.copy()[:,0:int(w/2)]
        crop_right = created.copy()[:, int(w/2):]

        checking_for_pixel = True
        pix_x, pix_y = 0,0
        while(checking_for_pixel):
            pix_x = random.choice(range(0,crop_right.shape[1]))
            pix_y =  random.choice(range(0,crop_right.shape[0]))
            if np.all(crop_right[pix_x, pix_y] == color ):
                checking_for_pixel = False
                if pix_x >=10:
                    pix_x = pix_x+10
                if pix_y >= 10:
                    pix_y = pix_y - 10
                if np.all(crop_right[pix_x,pix_y-15] != color):
                    checking_for_pixel =True
                if np.all(crop_right[pix_x,pix_y+15] != color):
                    checking_for_pixel =True
        crop_right = cv2.circle(crop_right.copy(), (pix_y,pix_x), 6, (0,255,0), -1) 
        crop_left = cv2.flip(crop_right, 1)
        images_without_defect_with_dot.append( (crop_left, crop_right,1) )

    print("# No defect on both side")
    for count in range(0,images_without_defect_count):
        center_coordinates = random.choice(possible_center_coordinates)
        color = random.choice(colors)
        #created = cv2.ellipse(image.copy(), center_coordinates, random.choice(possible_axis_lengths), 0.0, 0.0, 360.0, color, -1);
        
        x,y = random.choice(possible_axis_lengths)
        x = random.choice( [ int(x/1.4), int(x/1.3), int(x/1.2), int(x/1.1) ] )
        point_one = (center_coordinates[0],center_coordinates[1] - y)
        point_two  = (center_coordinates[0] - x,center_coordinates[1] )
        point_three  = (center_coordinates[0] + x,center_coordinates[1] )
        triangle = np.array([point_one, point_two, point_three,point_one])
        created = cv2.fillConvexPoly(image.copy(), triangle, color)

        
        crop_left = created.copy()[:,0:int(w/2)]
        crop_right = created.copy()[:, int(w/2):]

        checking_for_pixel = True
        pix_x, pix_y = 0,0
        while(checking_for_pixel):
            pix_x = random.choice(range(0,crop_right.shape[1]))
            pix_y =  random.choice(range(0,crop_right.shape[0]))
            if np.all(crop_right[pix_x, pix_y] == color ):
                checking_for_pixel = False
                if pix_x >=10:
                    pix_x = pix_x+10
                if pix_y >= 10:
                    pix_y = pix_y - 10
                if np.all(crop_right[pix_x,pix_y-10] != color):
                    checking_for_pixel =True
                if np.all(crop_right[pix_x,pix_y+10] != color):
                    checking_for_pixel =True
        #crop_right = cv2.circle(crop_right.copy(), (pix_y,pix_x), 10, (0,255,0), -1) 
        #crop_left = cv2.flip(crop_right, 1)
        images_without_defect.append( (crop_left, crop_right,0) )

    return images_without_defect,images_without_defect_with_dot,images_with_defect_on_left,images_with_defect_on_right




if __name__ == "__main__":

    height = 300
    width = 600

    ##Defining variables 
    images_with_defect_on_left_side_count = 1
    images_with_defect_on_right_side_count = 1
    images_without_defect_count = 1
    images_without_defect_with_dot_count = 1

    #creating a blank image
    blank_image = np.zeros((height,width,3), np.uint8)

    #creating ellipse
    ellipse_0, ellipse_1, ellips_2, ellipse_3 =  create_circle(blank_image.copy(),    images_with_defect_on_left_side_count,images_with_defect_on_right_side_count,images_without_defect_count, images_without_defect_with_dot_count)

    #creating rectangle
    rectangle_0, rectangle_1, rectangle_2, rectangle_3 =  create_rectangle(blank_image.copy(),    images_with_defect_on_left_side_count,images_with_defect_on_right_side_count,images_without_defect_count, images_without_defect_with_dot_count)

    #creating triangle
    triangle_0, triangle_1, triangle_2, triangle_3 =  create_triangle(blank_image.copy(),    images_with_defect_on_left_side_count,images_with_defect_on_right_side_count,images_without_defect_count, images_without_defect_with_dot_count)



    #createing dataset
    dataset = ellipse_0 + ellipse_1 + ellips_2 + ellipse_3 + rectangle_0 + rectangle_1 + rectangle_2 + rectangle_3 + triangle_0 + triangle_1 + triangle_2 + triangle_3


    print(len(dataset))

    for data in dataset:
        if data[2] == 0 :
            print("no dot")
        elif data[2] == 1 :
            print("dot on both side")
        elif data[2] == 2 :
            print("dot on right side")
        elif data[2] == 3:
            print("dot on left side")
        
        plt.imshow(data[0])
        plt.show()
        plt.imshow(data[1])
        plt.show()