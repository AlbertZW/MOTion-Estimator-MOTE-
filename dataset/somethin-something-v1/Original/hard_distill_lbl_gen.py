import os

relbl_path = './relabeled.txt'


relbl_file = open(relbl_path)
lbl_info_list = relbl_file.readlines()

lbl_list = dict()

for lbl_line in lbl_info_list:
    lbl_line = lbl_line.replace('\n', '')
    video_idx = lbl_line.split(' ')[0]
    video_lbl = lbl_line.split(' ')[1]
    lbl_list[video_idx] = video_lbl

src_path = './train_videofolder.txt'
hard_distill_file = './hard_distill_lbl.txt'

src_file = open(src_path)
dest_file = open(hard_distill_file, 'a')
info_list = src_file.readlines()

for info in info_list:
    video_idx = info.split(' ')[0]
    frame_num = info.split(' ')[1]

    if video_idx in lbl_list.keys():
        info = ' '.join([video_idx, frame_num, lbl_list[video_idx]]) + '\n'
    dest_file.write(info)

