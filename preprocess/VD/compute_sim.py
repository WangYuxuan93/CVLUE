import json
import numpy as np

# 计算问题相似度
# 输入是问题的隐含层表示
# 输出是相似度矩阵

question_hid_file = 'vd_question_hid.json'
sim_file = 'vd_question_sim.json'

with open(question_hid_file, 'r', encoding='utf-8') as f:
    string = f.read()
    question_hidden_states = json.loads(string)

hid_matrix = np.mat(question_hidden_states)

sim_matrix = np.dot(hid_matrix, hid_matrix.T)

# print('hid_matrix: ', hid_matrix)
print('sim_matrix_size: ', sim_matrix.shape)

sim_matrix = sim_matrix.tolist()
with open(sim_file, 'w',encoding='utf-8') as f:
    json.dump(sim_matrix, f,ensure_ascii=False,indent=2)