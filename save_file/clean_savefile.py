import importlib
import sys 
import os 

save_dir = os.getcwd()
file_list = os.listdir(save_dir)
rm_flag = False

for file_loop in file_list:
    if file_loop[:9] == 'save_file':
        file_open = open(file_loop, encoding='ascii', errors='ignore')
        file_line = file_open.readlines()
        if len(file_line)<20:
            rm_flag = True
        file_open.close()
        if rm_flag:
            os.remove(file_loop)
        rm_flag = False 

