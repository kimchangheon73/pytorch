# NLP(자연어처리) : TEXT데이터를 분석하고 모델링 하는 분야
    # NLU(자연어이해) 
    # NLG(자연어생성)
    
# NLP예시
    # 1. 감정분석 : 문장에 대한 특정 감정을 분류해내는 문제
    # 2. 요약 : 주어진 Text에서 중요한 부분을 찾아 추출/요약하는 문제 
    # 3. 기계번역 : 다른언어로 번역하고 새로운 언어의 문장을 생성하는 문제
    # 4. 질문응답 : 문서를 이해하고 문서 속 정보에 대한 질문에 답을 내는 문제 
    
# Torch Dataset 
from torchtext import data, datasets

TEXT = data.Field(lower=True, batch_first=True)     # batch_first : 배치크기를 shape의 가장 첫부분으로 설정,  lower : 문장을 모두 소문자화 
LABEL = data.Field(sequential=False)

train, test = datasets.IMDB.splits(root="./data",text_field=TEXT, label_field=LABEL)