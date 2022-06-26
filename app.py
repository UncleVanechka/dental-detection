import streamlit as st
from utils import read_files
import os


def main():
    st.title('Детекция кариеса')

    st.subheader("Загрузка файлов")
    filename = st.file_uploader('Выберите или ператащите сюда снимки', type=['png', 'jpeg', 'jpg'])
    filename2 = st.camera_input('Сделайте фото')

    if st.button('Загрузить') and (filename or filename2):
        if filename2:
            fn = filename2
        else:
            fn = filename
        paths, folder_name = read_files([fn])
        path = paths[0][0]
        os.system('python yolo5/detect.py --weights yolov5m6 --source images/')
        st.image('segmentation.png')


if __name__ == '__main__':
    main()
