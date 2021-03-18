import sys

import pandas as pd
import pythainlp
from math import exp
from math import floor

class MaxEnt:
    def __init__(self, csv_file_name):
        self.model_params = pd.read_csv(csv_file_name, index_col=0)
    def compute_probability(self, thai_text_string):
        token_list = pythainlp.word_tokenize(thai_text_string)
        name_of_feature = self.get_all_possible_features()
        labels = self.get_all_possible_labels()
        dict_final = {}
        keep_ans_label = {}
        for label in labels:
            #ดู label เป็นหลัก
            keep_value = []
            for feature in token_list:
                #ต่อมาก็ไล่คำ
                if feature in name_of_feature:
                    keep_value.append(self.model_params.loc[feature][label]) #เอาค่าของคำนั้นใน label นั้นๆมา แล้วก็เปลี่ยนคำไปเรื่อยๆ ถ้ามีคำเดียวมันก็มีอแค่ตัวเดียว
            print(keep_value) 
            keep_ans_label[label] = sum(keep_value) #อันนี้คือคำนวณแล้ว ก็คือเอามา + กันทั้งหมดใน keep value นั้นแล้วก็เก็บค่าลงไปใน label ใหญ่
        print(keep_ans_label) 
        for i in labels:
            keep_expo = 0
            for j in keep_ans_label.values():
                keep_expo += exp(j) #หา expo รวมๆกัน
            # print(keep_expo)
            # ans = floor((exp(keep_ans_label[i])/keep_expo)*100) #ปรับให้มันจุดทศนิยมสองตัวเฉยๆ *100 ปัดทิ้ง แล้วไป /100
            dict_final[i] = exp(keep_ans_label[i])/keep_expo
        print(dict_final)
        return dict_final
    def get_all_possible_features(self):
        return self.model_params.index.tolist()
    def get_all_possible_labels(self):
        return self.model_params.columns.tolist()
    def classify(self, thai_text_string):
        dict_final = self.compute_probability(thai_text_string)
        ans = max(dict_final, key=dict_final.get)
        return ans


if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print('usage:\tmaxent.py <model_file>')
        sys.exit(0)
    model_file_name = sys.argv[1]
    model = MaxEnt(model_file_name)


