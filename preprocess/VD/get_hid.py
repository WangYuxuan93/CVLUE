import json
import torch
from tqdm import tqdm


from transformers import BertModel
from torch.utils.data import DataLoader

# 基于 roberta 计算答案的隐层表示，并进行保存

model_path = '/models/chinese-roberta-wwm-ext'
cache_dir = '/tmp'
question_file = 'vd_question.json'
result_path = 'vd_question_hid.json'

from transformers import AutoTokenizer

model = BertModel.from_pretrained(
            model_path,
            cache_dir=cache_dir,
        )

tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            cache_dir=cache_dir
        )

with open(question_file, 'r', encoding='utf-8') as f:
    string = f.read()
    questions = json.loads(string)

question_list = [question for question in questions.values()]
# print('question_list: ', question_list)

questions_inputs = tokenizer(question_list, padding="longest", truncation=True)
# print('questions_inputs: ', questions_inputs)

batch_size = 32
question_len = len(question_list)
result = []

model = model.cuda()

for i in tqdm(range(0,question_len,batch_size)):
    temp_questions_input_ids = questions_inputs['input_ids'][i:min(i+batch_size,question_len)]
    temp_questions_token_type_ids = questions_inputs['token_type_ids'][i:min(i+batch_size,question_len)]
    temp_questions_attention_mask = questions_inputs['attention_mask'][i:min(i+batch_size,question_len)]

    temp_questions_input_ids = torch.tensor(temp_questions_input_ids).cuda()
    temp_questions_token_type_ids = torch.tensor(temp_questions_token_type_ids).cuda()
    temp_questions_attention_mask = torch.tensor(temp_questions_attention_mask).cuda()

    outputs = model(input_ids = temp_questions_input_ids, token_type_ids = temp_questions_token_type_ids, attention_mask = temp_questions_attention_mask)
    last_hidden_states = outputs.last_hidden_state

    cls = last_hidden_states[:,0,:]

    # print('last_hidden_states: ', last_hidden_states)
    # print('last_hidden_states.shape: ', last_hidden_states.size())

    # print('cls: ', cls)
    # print('cls.shape: ', cls.size())

    result += cls.cpu().detach().numpy().tolist()

# print('all_cls: ', all_cls.size())

# print('result: ', result)
print('len(result): ', len(result))
print('len(result[0]): ', len(result[0]))



with open(result_path, 'w',encoding='utf-8') as f:
    json.dump(result, f,ensure_ascii=False,indent=2)