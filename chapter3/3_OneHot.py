# One Hot Encoding
    # vectorization : 각 토큰을 유한한 개수의 범주형 자료로 표현했기 때문에 컴퓨터가 인식을 제대로 x
    # 문장의 각 단어를 총 문장 길이에서 해당 문장 위치 단어만 1로 표현하고 나머지는 0으로 표현하는 방법
    
# EX1
sentences : str = ["나는 책상 위에 사과를 먹었다",
                 "알고 보니 그 사과는 영철이 것이었다",
                 "그래서 영철이에게 사과를 했다"]

token2idx = {}
index = 0
for sentence in sentences:
    for token in sentence.split():
        if token2idx.get(token) == None:
            token2idx[token] = index
            index+=1
            
def indexed_sentence(sentence):
    return [token2idx[x] for x in sentence.split()]

result1 = indexed_sentence(sentences[0])
result2 = indexed_sentence(sentences[1])
result3 = indexed_sentence(sentences[2])

# 파이썬 리스트를 활용한 one hot encoding
V = len(token2idx)
for token, idx  in token2idx.items():
    result = []
    for i in range(V):
        if idx == i:
            result.append(1)
        else:
            result.append(0)
    print(f"{str(token):10s}\t{str(idx):2s}\t{str(result):30s}")

# numpy를 이용한 방법
import numpy as np
for sentence in sentences:
    onehot_s = []
    tokens = sentence.split()
    for token in tokens:
        if token2idx.get(token) != None:
            vector = np.zeros((1,V))
            vector[:, token2idx[token]] = 1
            onehot_s.append(vector)
        else:
            print("unk")
            
    print(f'{sentence}')
    print(np.concatenate(onehot_s, axis=0))
    print("\n\n")