import pytextrank
import spacy
import numpy as np
# 按行存储，第一位是数字，有关键词就放关键词的个数，没有就放0，
# 后面放关键词以 ';'分割，最后放关键词所在句子

# 这里的datacut是对整个文本字符长度作的初步筛选，以后可考虑使用源数据集，再对摘要split后的长度即词数进行筛选
source_path = 'C:/Users/you/Desktop/paper_rec/data/segment/datacut.txt'
source_path = 'C:/Users/you/Desktop/paper_rec/data/testdata.txt'

keywords_num = 10
dataset_path = 'C:/Users/you/Desktop/paper_rec/data/nenn_dataset.txt'

fw = open(dataset_path, 'w',encoding='utf-8')

nlp = spacy.load("en_core_web_sm")
# 这里的关键词获取方式是pytextrank，以后是否有对比实验的改进空间
nlp.add_pipe("textrank", config={ "stopwords": { "words": ["NOUN"] } })

all_text_list = []
temp_str = ""
with open(source_path,'r',encoding='utf8') as f:
    for line in f.readlines():
        temp_str += line
        if(line == "\n" and temp_str.find("#index") > -1 and temp_str.find("#!") > -1):
            # 如果是数据全集需要再加一个判断长度的if，下面内容除了tempstr赋空以外放在那个if里
            all_text_list.append(temp_str)

            all_abs_text = temp_str.split('\n')[-3].replace('#! ','')# 字符串
            all_abs_sentence = temp_str.split('\n')[-3].replace('#! ','').split('. ')# list
            all_user_name = temp_str.split('\n')[2].replace('#@ ','').split(';')# list
            all_paper_id= temp_str.split('\n')[0].replace('#index ','')# 字符串
            
            # 获取关键词
            doc = nlp(all_abs_text)
            count_i = np.zeros((len(all_abs_sentence)), dtype=int)
            count_keyword = []
            for i in range(len(all_abs_sentence)):
                count_keyword.append('')
            keyword = []
            # 这里之前使用的np.zeros把数据换成str但是有长度的限制
            for phrase in doc._.phrases[:keywords_num]:                        
                # 句子级查找是否有关键词
                # print(phrase.text,'\n') 这里关键词句级共现还挺多的
                fw.write(phrase.text)
                fw.write('; ')
                for i in range(len(all_abs_sentence)):                    
                    count_i[i] += all_abs_sentence[i].count(phrase.text)
                    if(all_abs_sentence[i].count(phrase.text) > 0):
                        for n in range(all_abs_sentence[i].count(phrase.text)):
                            count_keyword[i] += phrase.text
                            count_keyword[i] += '; '#这里的分割符还可以放更严谨的
                            
            
            fw.write('\n')
            fw.write(str(len(all_abs_sentence)))# 指代文章分句总长，之后可扩展添加其他信息
            fw.write('\n')

            for i in range(len(all_abs_sentence)):
                fw.writelines([str(count_i[i]),'; ',count_keyword[i],all_abs_sentence[i],'\n'])
            temp_str = ""
f.close()
fw.close()
with open(dataset_path,'r+',encoding='utf8') as fa:
    content = fa.read()        
    fa.seek(0, 0)
    print('文章个数为：',str(len(all_text_list)),'关键词数量为：',keywords_num)
    fa.write(str(len(all_text_list))+'\n'+content)
fa.close()
