import streamlit as st
import numpy as np
import pandas as pd
import lightfm as lf
import nmslib
import pickle
import scipy.sparse as sparse

def nearest_items_ids(item_id, index, n=10):
    """Функция для поиска ближайших соседей, возвращает построенный индекс"""
    nn = index.knnQuery(item_embeddings[item_id], k=n)
    return nn
          
def load_embeddings():
    """
    Функция для загрузки векторных представлений
    """
    with open('item_embeddings.pickle', 'rb') as f:
        item_embeddings = pickle.load(f)
    # Тут мы используем nmslib, чтобы создать наш быстрый knn
    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    nms_idx.addDataPointBatch(item_embeddings)
    nms_idx.createIndex(print_progress=True)
    return item_embeddings,nms_idx

#Загружаем данные
item_embeddings,nms_idx = load_embeddings()

#Форма для ввода текста
id_num = st.text_input('Введите item_id', '')
val_index = int(id_num) 

#Ищем рекомендации
index = nearest_items_ids(val_index, nms_idx)

#Выводим рекомендации к товару

st.write('Рекомендованные товары (item_id)', index[0][1:])