import cv2
from PIL import Image
import numpy as np
import random
import pdb


def mask(src_img, boxes, att_map, mask_max_size, mask_dir):
    src_img = cv2.cvtColor(np.asarray(src_img),cv2.COLOR_RGB2BGR)   
    src_shape = src_img.shape
    pick_mask = random.randint(1,16)

    mask_img = cv2.imread(mask_dir+'/'+str(pick_mask)+'.jpg')
    mask_shape = mask_img.shape
    # distinguish background
    if pick_mask>13:
        background_label = 0 # background is black
    else:
        background_label = 1
    
    re_ = random.uniform(1, mask_max_size)
    re_size = (int(re_*mask_shape[0]), int(re_*mask_shape[1]))
    mask_img = cv2.resize(mask_img, re_size, interpolation=cv2.INTER_LINEAR)
    while re_size[0]>=src_shape[0] or re_size[1]>=src_shape[1]:
        re_ = random.uniform(1, mask_max_size)
        re_size = [int(re_*mask_shape[0]), int(re_*mask_shape[1])]
    mask_shape = re_size
    crop_pos = [random.randint(0,src_shape[1]-re_size[1]), 
                random.randint(0,src_shape[0]-re_size[0])]

    for h in range(re_size[0]):
        for w in range(re_size[1]):
            if background_label == 1:
                padding = [255, 255, 255]
            else:
                padding = [0, 0, 0]

            if np.ndarray.tolist(mask_img[w, h, :]) != padding:
                src_img[crop_pos[1]+w, crop_pos[0]+h, :] = mask_img[w, h, :]
    image = Image.fromarray(cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)) 
    return image
    #print("already output masked image")
    #cv2.imwrite(save_name, src_img)



def masked_att(src_img, boxes, att_map, mask_max_size, mask_dir):
    src_img = cv2.cvtColor(np.asarray(src_img),cv2.COLOR_RGB2BGR)   
    src_shape = src_img.shape
    pick_mask = random.randint(1,16)

    mask_img = cv2.imread(mask_dir+'/'+str(pick_mask)+'.jpg')
    mask_shape = mask_img.shape
    # distinguish background
    if pick_mask>13:
        background_label = 0 # background is black
    else:
        background_label = 1
    att_padding = 0

    for sub_box in boxes:
        sub_box = [int(x) for x in sub_box]
        box_wid = sub_box[2] - sub_box[0]
        box_hei = sub_box[3] - sub_box[1]
        if box_wid <30 or box_hei<30:
            continue
        crop_pos = [random.randint(sub_box[0], sub_box[2]),
                    random.randint(sub_box[1], sub_box[3])]
        re_ = random.uniform(0.1, mask_max_size)
        re_size = ((int(re_*box_wid)), int(re_*box_hei))

        crop_radio = [(min(crop_pos[0]+re_size[0], sub_box[2])-crop_pos[0])/box_wid,
                      (min(crop_pos[1]+re_size[1], sub_box[3])-crop_pos[1])/box_hei]
        #print(mask_max_size)
        #print('first crop', re_)
        #print('src_shape', src_shape)
        
        while max(crop_radio)>0.8:
            #print('re_:',re_)
            #print('crop_radio:', crop_radio)
            #print('crop again')
            crop_pos = [random.randint(sub_box[0], sub_box[2]),
                        random.randint(sub_box[1], sub_box[3])]
            re_ = random.uniform(0.1, mask_max_size)
            re_size = (int(re_*box_wid), int(re_*box_hei))
            crop_radio = [(min(crop_pos[0]+re_size[0], sub_box[2])-crop_pos[0])/box_wid,
                          (min(crop_pos[1]+re_size[1], sub_box[3])-crop_pos[1])/box_hei]
        #print('mask shape', mask_img.shape)
        #print('resize shape', re_size)
        re_mask_img = cv2.resize(mask_img, (re_size[1], re_size[0]), interpolation=cv2.INTER_LINEAR)
        if background_label == 1:
            padding = [255, 255, 255]
        else:
            padding = [0, 0, 0]
            
        # h+pos[1]    w+pos[0]
        for h in range(re_size[0]): 
            for w in range(re_size[1]):
                if crop_pos[0]+h>=src_shape[0] or crop_pos[1]+w>=src_shape[1]:
                    break
                if np.ndarray.tolist(re_mask_img[h, w, :]) != padding:
                    src_img[crop_pos[0]+h, crop_pos[1]+w, :] = np.array(re_mask_img[h, w, :])
                    if att_map[crop_pos[0]+h, crop_pos[1]+w] != att_padding:
                        att_map[crop_pos[0]+h, crop_pos[1]+w] = att_padding
    image = Image.fromarray(cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)) 
    return att_map, image
    


    '''
    re_ = random.uniform(1, mask_max_size)
    re_size = (int(re_*mask_shape[0]), int(re_*mask_shape[1]))
    while re_size[0]>=src_shape[0] or re_size[1]>=src_shape[1]:
        re_ = random.uniform(1, mask_max_size)
        re_size = (int(re_*mask_shape[0]), int(re_*mask_shape[1]))
    re_mask_img = cv2.resize(mask_img, (re_size[1], re_size[0]), interpolation=cv2.INTER_LINEAR)
    crop_pos = [random.randint(0,src_shape[0]-re_size[0]), 
                random.randint(0,src_shape[1]-re_size[1])]

    if background_label == 1:
        padding = [255, 255, 255]
    else:
        padding = [0, 0, 0]
        
    att_padding = 0

    # h+pos[1]    w+pos[0]
    for h in range(re_size[0]): 
        for w in range(re_size[1]):
            if np.ndarray.tolist(re_mask_img[h, w, :]) != padding:
                src_img[crop_pos[0]+h, crop_pos[1]+w, :] = np.array(re_mask_img[h, w, :])
                if att_map[crop_pos[0]+h, crop_pos[1]+w] != att_padding:
                    att_map[crop_pos[0]+h, crop_pos[1]+w] = att_padding               
    image = Image.fromarray(cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)) 
    return att_map, image
    '''



if __name__ == '__main__':
    #img = cv2.imread('src.jpg')
    img = Image.open('src.jpg')
    if img.mode != 'RGB':
        img = img.convert('RGB')
    src_shape = img.size
    att_map = np.zeros([src_shape[1], src_shape[0]])
    pdb.set_trace()
    att_map, img_ = masked_att(img, att_map, 20, './')
    #cv2.imwrite('att_test.jpg', att_map)
    att_ = Image.fromarray(att_map)
    if att_.mode != 'RGB':
        att_ = att_.convert('RGB')
    att_.save('att_test.jpg')
    img_.save('src_test.jpg')





        
    




