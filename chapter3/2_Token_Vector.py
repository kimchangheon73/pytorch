# TEXT to Number : 모델 학습을 위해 문장을 숫자로 표현

# 작업순서
    # 1. Tokenizaion : 문장을 의미 있는 부분으로 나눈다 
    # 2. 의미 있는 부분을 숫자로 매칭해 단어사전을 만든다
    # 3. 단어사전을 통해 문장을 숫자로 표현한다


# 문장 준비 
sentences : str = ["나는 책상 위에 사과를 먹었다",
                 "알고 보니 그 사과는 영철이 것이었다",
                 "그래서 영철이에게 사과를 했다"]

# 단어사전 구축
token2idx = {}
index = 0
for sentence in sentences:
    for token in sentence.split():
        if token2idx.get(token) == None:
            token2idx[token] = index
            index+=1
            
# Text to number
def indexed_sentence(sentence):
    return [token2idx[x] for x in sentence.split()]

# result 
result1 = indexed_sentence(sentences[0])
result2 = indexed_sentence(sentences[1])
result3 = indexed_sentence(sentences[2])
print(result1)
print(result2)
print(result3)


# 2.1 corpus & OOV(out of voca)
    # oov : 단어사전에 없는 단어가 문장에 나오는 현상 


# 기존 token 사전에 <unk> token 추가 
token2idx = {t : i+1 for t, i in token2idx.items()}
token2idx['<unk>'] = 0

# 토큰이 없을 경우 <unk> token으로 치환하는 새로운 함수 지정
sentences.append("나는 책상 위에 배를 먹었다")
def indexed_sentence(sentence):
    return [token2idx.get(token, token2idx['<unk>']) for token in sentence.split()]
# result 
result4 = indexed_sentence(sentences[3])
print(result4)






# 2.2 N gram Tokenization 
    # 글자의 특정 연속성을 고려해 n개의 단어로 token을 만드는 방법
# uni gram : 1개씩 
print([sentences[0][i:i+1] for i in range(len(sentences[0]))])
# bi gram : 2개씩 
print([sentences[0][i:i+2] for i in range(len(sentences[0]))])
# tri gram : 3개씩 
print([sentences[0][i:i+3] for i in range(len(sentences[0]))])


# 2.3 BPE (Byte Pair Encoding)
    # 반복적으로 나오는 데이터의 연속 패턴을 치환하는 방식 
    # ex : ab -> x,  cd -> y